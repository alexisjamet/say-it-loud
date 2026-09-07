// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib

public class ASR {
    let moshi: LM
    let vocab: [Int: String]
    let mimi: Mimi
    var prevTextToken: Int = 0
    let sampler: Sampler = Sampler(temp: 0.0)
    let cb: Callbacks

    public init(
        _ moshi: LM, _ mimi: Mimi, vocab: [Int: String], cb: Callbacks = EmptyCallbacks()
    ) {
        self.moshi = moshi
        self.mimi = mimi
        self.vocab = vocab
        self.cb = cb
    }

    public func reset() {
        mimi.resetState()
        moshi.resetCache()
        prevTextToken = self.moshi.cfg.textInitToken()
        let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
        let audioIds = (0..<self.moshi.cfg.audioCodebooks).map { _ in
            MLXArray([moshi.cfg.audioPaddingToken()])
        }
        let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
        let (textToken, _) = sampler(logits: textLogits)
        let textTokenI: Int = textToken[0].item()
        prevTextToken = textTokenI
        cb.onReset()
    }

    public func onPcmInput(_ pcm: MLXArray) -> [String] {
        var tokens: [String] = []
        let codebooks = moshi.cfg.audioCodebooks
        cb.onEvent(.beginEncode)
        let codes = mimi.encodeStep(StreamArray(pcm))
        codes.eval()
        cb.onEvent(.endEncode)
        if let codes = codes.asArray() {
            cb.onInputAudioTokens(codes)
            let (_, _, steps) = codes.shape3
            for step in 0..<steps {
                let textIds = MLXArray([prevTextToken]).reshaped([1, 1])
                let audioIds = (0..<codebooks).map { codes[0..., $0, step].reshaped(1, 1) }
                cb.onEvent(.beginStep)
                let (_, textLogits) = moshi.stepMain(textIds: textIds, audioIds: audioIds)
                eval(textLogits)
                cb.onEvent(.endStep)
                let (textToken, _) = sampler(logits: textLogits)
                let textTokenI: Int = textToken[0].item()
                cb.onOutputTextToken(textTokenI)
                if textTokenI != 0 && textTokenI != 3 {
                    if var v = vocab[textTokenI] {
                        v.replace("▁", with: " ")
                        tokens.append(v)
                    }
                }
                prevTextToken = textTokenI
            }
        }
        return tokens
    }
}

/// Long-form dictation on top of `ASR`.
///
/// The model keeps a 60 s attention window plus its own text stream as context. On
/// continuous speech its output drifts after two to three minutes and eventually
/// locks into a loop (the same character over and over) — Kyutai's PyTorch
/// reference does the same, see delayed-streams-modeling issues #172 and #175.
/// Every minute or so this wrapper waits for a short pause, drains the words still
/// in flight with a little silence, and starts the model over, replaying the last
/// seconds of audio (output muted) so it gets the speaker's voice back before new
/// words arrive; a detected loop is cut out of the text and triggers an immediate
/// restart.
public final class StreamingASR {
    public let frameSize = 1920  // 80 ms at 24 kHz, one Mimi frame
    /// Silence fed before speech; the model expects some.
    public var prefixFrames = 13
    /// Silence fed at the end: words come out ~0.5 s late.
    public var flushFrames = 25
    /// Restart at the first pause once this many frames were seen since the last start...
    public var restartAfter = 12 * 60
    /// ...and no later than this, pause or not.
    public var forceRestartAfter = 12 * 120
    /// Consecutive quiet frames that count as a pause.
    public var pauseFrames = 4
    /// The same token this many times in a row is a loop, not speech.
    public var loopLength = 8
    /// Audio replayed after a restart, as context.
    public var replayFrames = 36

    public private(set) var text = ""
    public private(set) var restarts = 0

    private let asr: ASR
    private let silence: MLXArray
    private var pending: [Float] = []
    private var steps = 0
    private var quietFrames = 0
    private var noiseFloor: Float = 1
    private var lastToken = ""
    private var repeats = 0
    private var recent: [[Float]] = []
    private var muted = false

    public init(_ asr: ASR) {
        self.asr = asr
        self.silence = MLXArray([Float](repeating: 0, count: frameSize))[.newAxis, .newAxis]
    }

    public func start() {
        text = ""
        restarts = 0
        pending = []
        recent = []
        noiseFloor = 1
        restartModel()
    }

    /// Feeds captured audio (24 kHz mono); any chunk size works.
    public func feed(_ pcm: [Float]) {
        pending.append(contentsOf: pcm)
        while pending.count >= frameSize {
            let frame = Array(pending[0..<frameSize])
            pending.removeFirst(frameSize)
            step(frame)
        }
    }

    /// Call once the recording is over to get the last words out.
    public func finish() {
        if !pending.isEmpty {
            let frame = pending + [Float](repeating: 0, count: frameSize - pending.count)
            pending = []
            step(frame)
        }
        for _ in 0..<flushFrames { emit(asr.onPcmInput(silence)) }
    }

    private func step(_ frame: [Float]) {
        let rms = (frame.reduce(0) { $0 + $1 * $1 } / Float(frame.count)).squareRoot()
        noiseFloor = min(rms, noiseFloor * 1.02)
        quietFrames = rms < max(0.006, noiseFloor * 3) ? quietFrames + 1 : 0
        emit(asr.onPcmInput(MLXArray(frame)[.newAxis, .newAxis]))
        steps += 1
        recent.append(frame)
        if recent.count > replayFrames { recent.removeFirst() }
        if repeats >= loopLength {
            // Drop the loop and start over; whatever was in flight is garbage too.
            text = String(text.dropLast(lastToken.count * repeats))
            restartModel()
        } else if steps >= forceRestartAfter || (steps >= restartAfter && quietFrames >= pauseFrames) {
            for _ in 0..<8 { emit(asr.onPcmInput(silence)) }
            restartModel()
            // The replayed words were already emitted before the restart: keep them out.
            muted = true
            for frame in recent { emit(asr.onPcmInput(MLXArray(frame)[.newAxis, .newAxis])) }
            for _ in 0..<8 { emit(asr.onPcmInput(silence)) }
            muted = false
        }
    }

    private func emit(_ tokens: [String]) {
        if muted { return }
        for t in tokens {
            text += t
            if t == lastToken {
                repeats += 1
            } else {
                lastToken = t
                repeats = 1
            }
        }
    }

    private func restartModel() {
        asr.reset()
        for _ in 0..<prefixFrames { _ = asr.onPcmInput(silence) }
        steps = 0
        quietFrames = 0
        lastToken = ""
        repeats = 0
        restarts += 1
    }
}
