// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import ArgumentParser
import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib
import Synchronization
import Tokenizers

/// Runs an async job from the synchronous command entry points.
func runBlocking<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let result = Mutex<Result<UnsafeSendable<T>, Error>?>(nil)
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            let value = try await body()
            result.withLock { $0 = .success(UnsafeSendable(value: value)) }
        } catch {
            result.withLock { $0 = .failure(error) }
        }
        semaphore.signal()
    }
    semaphore.wait()
    return try result.withLock { $0! }.get().value
}

struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
}

func downloadFromHub(id: String, filename: String) throws -> URL {
    let targetURL = HubApi().localRepoLocation(Hub.Repo(id: id)).appending(path: filename)
    if FileManager.default.fileExists(atPath: targetURL.path) {
        print("using cached file \(targetURL.path)")
        return targetURL
    }
    let url = try runBlocking {
        try await Hub.snapshot(from: Hub.Repo(id: id), matching: filename) { progress in
            let pct = Int(progress.fractionCompleted * 100)
            print("\rretrieving \(filename): \(pct)%", terminator: "")
        }
    }
    print("\rretrieved \(filename)")
    return url.appending(path: filename)
}

func maybeDownloadFromHub(filename: String) throws -> URL {
    // dowloadFromHub(id: id, filename: filename)
    let prefix = "hf://"
    if filename.hasPrefix(prefix) {
        let rest = filename.dropFirst(prefix.count)
        let components = rest.split(separator: "/", omittingEmptySubsequences: false)
        let id = components[0..<components.count - 1].joined(separator: "/")
        let filename = String(components.last!)
        return try downloadFromHub(id: id, filename: filename)
    } else {
        return URL(fileURLWithPath: filename)
    }
}

func makeTokenizer(hfRepo: String) throws -> any Tokenizer {
    try runBlocking { try await AutoTokenizer.from(pretrained: hfRepo) }
}

@main
struct Moshi: ParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [
            Run.self, RunHelium.self, RunMimi.self, AudioToCodes.self, CodesToAudio.self,
            RunAsr.self, RunRewrite.self,
        ]
    )
}

public enum Config: String, CaseIterable, ExpressibleByArgument {
    case moshi1b
    case moshi7b
}

struct Run: ParsableCommand {
    @Argument(help: "the model to run")
    var model: String

    @Option(help: "the file to process, use 'mic' for the microphone")
    var input: String?

    @Option(help: "the audio delay to apply")
    var audioDelay: Int = 2

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    @Option(help: "the config size")
    var config: Config = .moshi1b

    mutating func run() throws {
        let model = URL(fileURLWithPath: model)
        let cfg =
            switch config {
            case .moshi1b: LmConfig.moshi1b(audioDelay: audioDelay)
            case .moshi7b: LmConfig.moshi_2024_07()
            }

        switch input {
        case .none:
            try runMoshi(model, cfg: cfg, audioFile: nil)
        case .some("mic"): try runMoshiMic(model, cfg: cfg)
        case .some(let input):
            let audioFile = URL(fileURLWithPath: input)
            try runMoshi(
                model, cfg: cfg, audioFile: audioFile, channel: channel)
        }
    }
}

struct RunMimi: ParsableCommand {
    @Option(help: "whether to use the streaming mode or not")
    var streaming: Bool = false

    @Option(help: "the file to process")
    var input: String?

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    mutating func run() throws {
        let audioFile = input.flatMap { URL(fileURLWithPath: $0) }
        try runMimi(streaming: streaming, audioFile: audioFile, channel: channel)
    }
}

public enum HeliumConfig: String, CaseIterable, ExpressibleByArgument {
    case q4
    case q6
    case q8
    case bf16
}

struct RunHelium: ParsableCommand {
    @Option(help: "the config")
    var config: HeliumConfig = .q4

    mutating func run() throws {
        let cfg = LmConfig.helium2b()
        let filename =
            switch config {
            case .q4: "helium-1-preview-2b-q4.safetensors"
            case .q6: "helium-1-preview-2b-q6.safetensors"
            case .q8: "helium-1-preview-2b-q8.safetensors"
            case .bf16: "helium-1-preview-2b-bf16.safetensors"
            }
        let url = try downloadFromHub(id: "kyutai/helium-1-preview-2b-mlx", filename: filename)
        try runHelium(url, cfg: cfg)
    }
}

struct AudioToCodes: ParsableCommand {
    mutating func run() throws {
        try runAudioToCodes()
    }
}

struct CodesToAudio: ParsableCommand {
    @Option(help: "whether to write the output file or not")
    var writeFile: Bool = false

    mutating func run() throws {
        try runCodesToAudio(writeFile: writeFile)
    }
}

struct RunAsr: ParsableCommand {
    @Argument(help: "the model to run")
    var model: String

    @Option(help: "the file to process, use 'mic' for the microphone")
    var input: String?

    @Option(help: "the audio channel from the input file to be used")
    var channel: Int = 0

    mutating func run() throws {
        let model = try maybeDownloadFromHub(filename: model)
        let weights = try loadArrays(url: model)
        let cfg =
            switch weights["out_norm.weight"]?.shape {
            case .none: fatalError("no out_norm.weight tensor in \(model)")
            case .some([1024]): LmConfig.asr300m()
            case .some([2048]): LmConfig.asr1b()
            case .some([2560]): LmConfig.asr2b()
            case .some(let s): fatalError("unexpected shape for out_norm.weight \(s)")
            }
        print("here2")
        switch input {
        case .none:
            try runAsr(model, cfg, audioFile: nil, channel: channel)
        case .some("mic"): try runAsrMic(model, cfg)
        case .some(let input):
            let audioFile = URL(fileURLWithPath: input)
            try runAsr(model, cfg, audioFile: audioFile, channel: channel)
        }
    }
}

struct RunRewrite: ParsableCommand {
    @Option(help: "the Hugging Face repo of the mlx-community Ministral 3 checkpoint")
    var hfRepo: String = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"

    @Option(help: "fix, friend, email, bullets, english, french, or a free instruction")
    var task: String = "fix"

    @Argument(help: "the text to rewrite")
    var text: String

    mutating func run() throws {
        for f in ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"] {
            _ = try downloadFromHub(id: hfRepo, filename: f)
        }
        let folder = HubApi().localRepoLocation(Hub.Repo(id: hfRepo))
        let tokenizer = try LlmTokenizer.load(from: folder)
        let task: RewriteTask =
            switch self.task {
            case "fix": .fix
            case "friend": .friend
            case "email": .email
            case "bullets": .bullets
            case "english": .toEnglish
            case "french": .toFrench
            case let s: .custom(s)
            }
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try LlmModel.load(from: folder)
        print("model loaded in \(CFAbsoluteTimeGetCurrent() - t0)s")
        let prompt = tokenizer.encode(text: task.prompt(for: text), addSpecialTokens: false)
        let gen = LlmGenerator(model)
        var out: [Int] = []
        var shown = ""
        let t1 = CFAbsoluteTimeGetCurrent()
        gen.generate(prompt: prompt) { tok in
            out.append(tok)
            let text = tokenizer.decode(tokens: out, skipSpecialTokens: true)
            if text.hasPrefix(shown) {
                print(text.dropFirst(shown.count), terminator: "")
                shown = text
            }
            fflush(stdout)
            return true
        }
        let dt = CFAbsoluteTimeGetCurrent() - t1
        print("\n\(prompt.count) prompt tokens, \(out.count) generated in \(dt)s (\(Double(out.count) / dt) tok/s)")
    }
}

