// Downloads the weights from Hugging Face into the app's container and builds
// the speech model. (The original multi-model demo was removed; see git history.)

import Foundation
import Hub
import MLX
import MLXNN
import MoshiLib
import Observation

struct CustomError: Error {
    let message: String
    init(_ s: String) { message = s }
}

/// Hands a non-Sendable MLX model to the thread that will own it. The caller
/// promises the value is not touched from anywhere else meanwhile.
struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
}

@Observable
@MainActor
final class Evaluator {
    static let mimiFilename = "tokenizer-dbaa9758-checkpoint125.safetensors"
    static let sttFilename = "moshi-70f8f0ea@500.q8.safetensors"

    /// Bytes received / expected for the file currently downloading, nil otherwise.
    var download: (done: Int64, total: Int64)? = nil
    private var asr: ASR?

    func unloadAsr() {
        asr = nil
        GPU.clearCache()
    }

    /// The speech model, downloaded and built on first call and kept afterwards.
    func loadAsr() async throws -> ASR {
        if let asr { return asr }
        let url = try await downloadFromHub(id: "lmz/moshi-swift", filename: Self.sttFilename)
        let mimiURL = try await downloadFromHub(id: "lmz/moshi-swift", filename: Self.mimiFilename)
        let cfg = LmConfig.asr1b()
        let vocab = try await loadVocab(cfg)
        // Loading ~1.4 GB of weights and warming up takes seconds: keep the UI responsive.
        let built = try await Task.detached(priority: .userInitiated) {
            let moshi = try Self.makeMoshi(url, cfg)
            let mimi = try Self.buildMimi(mimiURL, numCodebooks: 32)
            mimi.warmup()
            moshi.warmup()
            return UnsafeSendable(value: (moshi, mimi))
        }.value
        let (moshi, mimi) = built.value
        let asr = ASR(moshi, mimi, vocab: vocab)
        self.asr = asr
        return asr
    }

    func downloadFromHub(id: String, filename: String) async throws -> URL {
        guard
            let downloadDir = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw CustomError("cannot find the app support directory")
        }
        let api = HubApi(downloadBase: downloadDir)
        let repo = Hub.Repo(id: id)
        let targetURL = api.localRepoLocation(repo).appending(path: filename)
        if FileManager.default.fileExists(atPath: targetURL.path) {
            return targetURL
        }
        // The Hub client only reports progress per file; stream the bytes ourselves so
        // a 1 GB checkpoint shows a moving bar instead of 0 % then 100 %.
        let remote = URL(string: "https://huggingface.co/\(id)/resolve/main/\(filename)")!
        try FileManager.default.createDirectory(
            at: targetURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        download = (0, 0)
        defer { download = nil }
        try await FileDownloader.download(remote, to: targetURL) { done, total in
            Task { @MainActor in self.download = (done, total) }
        }
        return targetURL
    }

    private func loadVocab(_ cfg: LmConfig) async throws -> [Int: String] {
        let filename =
            switch cfg.textOutVocabSize {
            case 48000: "tokenizer_spm_48k_multi6_2.json"
            case 32000: "tokenizer_spm_32k_3.json"
            case 8000: "tokenizer_spm_8k_0.json"
            case 4000: "test_en_audio_4000.json"
            case let other: throw CustomError("unexpected text vocab size \(other)")
            }
        let fileURL = try await downloadFromHub(id: "lmz/moshi-swift", filename: filename)
        return try JSONDecoder().decode([Int: String].self, from: Data(contentsOf: fileURL))
    }

    nonisolated static func makeMoshi(_ url: URL, _ cfg: LmConfig) throws -> LM {
        let weights = try loadArrays(url: url)
        let parameters = ModuleParameters.unflattened(weights)
        let model = LM(cfg, bSize: 1)
        if url.lastPathComponent.hasSuffix(".q4.safetensors") {
            quantize(model: model, groupSize: 32, bits: 4)
        } else if url.lastPathComponent.hasSuffix(".q6.safetensors") {
            quantize(model: model, groupSize: 64, bits: 6)
        } else if url.lastPathComponent.hasSuffix(".q8.safetensors") {
            quantize(model: model, groupSize: 64, bits: 8)
        }
        try model.update(parameters: parameters, verify: [.all])
        eval(model)
        return model
    }

    nonisolated static func buildMimi(_ url: URL, numCodebooks: Int) throws -> Mimi {
        let cfg = MimiConfig.mimi_2024_07(numCodebooks: numCodebooks)
        let model = Mimi(cfg, bSize: 1)
        let origWeights = try loadArrays(url: url)
        var weights: [String: MLXArray] = [:]
        for (var key, var weight) in origWeights {
            if key.hasPrefix("encoder.model") {
                key.replace("encoder.model.", with: "encoder.")
            }
            if key.hasPrefix("decoder.model") {
                key.replace("decoder.model.", with: "decoder.")
            }
            if key.hasSuffix(".in_proj_weight") {
                key.replace(".in_proj_weight", with: ".in_proj.weight")
            }
            if key.hasSuffix(".linear1.weight") {
                key.replace(".linear1.weight", with: ".gating.linear1.weight")
            }
            if key.hasSuffix(".linear2.weight") {
                key.replace(".linear2.weight", with: ".gating.linear2.weight")
            }
            // Hardcoded matching between the pytorch layers and their mlx equivalent.
            for (layerIdx, decoderIdx) in [2, 5, 8, 11].enumerated() {
                key.replace("decoder.\(decoderIdx).", with: "decoder.layers.\(layerIdx).upsample.")
                key.replace(
                    "decoder.\(decoderIdx + 1).", with: "decoder.layers.\(layerIdx).residuals.0.")
            }
            for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
                key.replace(
                    "encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
                key.replace(
                    "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
            }
            key.replace("decoder.0.", with: "decoder.init_conv1d.")
            key.replace("decoder.14.", with: "decoder.final_conv1d.")
            key.replace("encoder.0.", with: "encoder.init_conv1d.")
            key.replace("encoder.14.", with: "encoder.final_conv1d.")
            key.replace(".block.1.", with: ".block.0.")
            key.replace(".block.3.", with: ".block.1.")
            // PyTorch layout for conv weights is outC, inC, kSize, for MLX it's outC, kSize, inC
            if key.hasSuffix(".conv.weight") || key.hasSuffix(".output_proj.weight")
                || key.hasSuffix(".input_proj.weight")
            {
                weight = weight.swappedAxes(-1, -2)
            }
            // PyTorch layout for conv-transposed weights is inC, outC, kSize, for MLX it's outC, kSize, inC
            if key.hasSuffix(".convtr.weight") {
                weight = weight.transposed(axes: [1, 2, 0])
            }
            weights[key] = weight
        }
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        return model
    }
}
