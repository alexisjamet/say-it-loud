// Ministral 3 decoder, used to rewrite / translate transcripts on device.
//
// Llama layout with YaRN RoPE and the Llama-4 style attention temperature.
// Weights come from mlx-community checkpoints (already quantized); they also
// carry a vision tower, which is dropped at load time.

import Foundation
import MLX
import MLXFast
import MLXNN
import Synchronization

public struct LlmQuantization {
    public var groupSize: Int
    public var bits: Int
}

public struct LlmConfig {
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var headDim: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    /// YaRN parameters (Ministral 3); nil means plain RoPE.
    public var yarn: Yarn?
    /// Llama-4 attention temperature: `1 + beta * log(1 + floor(pos / originalMaxPositions))`.
    public var llama4ScalingBeta: Float?
    public var tieWordEmbeddings: Bool
    public var vocabSize: Int
    public var quantization: LlmQuantization?

    public struct Yarn {
        public var factor: Float
        public var originalMaxPositions: Int
        public var betaFast: Float
        public var betaSlow: Float
        public var mscale: Float
        public var mscaleAllDim: Float
    }

    /// Reads an mlx-community `config.json`; Mistral 3 nests the decoder under `text_config`.
    public static func load(from folder: URL) throws -> LlmConfig {
        let data = try Data(contentsOf: folder.appending(path: "config.json"))
        guard let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LlmError("config.json is not a JSON object")
        }
        let modelType = top["model_type"] as? String ?? ""
        let quant = (top["quantization"] as? [String: Any]).map {
            LlmQuantization(groupSize: $0["group_size"] as? Int ?? 64, bits: $0["bits"] as? Int ?? 4)
        }
        func int(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = d[k] as? Int else { throw LlmError("missing \(k) in config.json") }
            return v
        }
        func float(_ d: [String: Any], _ k: String) throws -> Float {
            guard let v = d[k] as? Double else { throw LlmError("missing \(k) in config.json") }
            return Float(v)
        }
        switch modelType {
        case "mistral3", "ministral3":
            let text = top["text_config"] as? [String: Any] ?? top
            let rope = text["rope_parameters"] as? [String: Any] ?? [:]
            let heads = try int(text, "num_attention_heads")
            let hidden = try int(text, "hidden_size")
            var yarn: Yarn? = nil
            if (rope["rope_type"] as? String ?? rope["type"] as? String) == "yarn" {
                yarn = Yarn(
                    factor: try float(rope, "factor"),
                    originalMaxPositions: try int(rope, "original_max_position_embeddings"),
                    betaFast: (rope["beta_fast"] as? Double).map(Float.init) ?? 32,
                    betaSlow: (rope["beta_slow"] as? Double).map(Float.init) ?? 1,
                    mscale: (rope["mscale"] as? Double).map(Float.init) ?? 1,
                    mscaleAllDim: (rope["mscale_all_dim"] as? Double).map(Float.init) ?? 0)
            }
            return LlmConfig(
                // Tekken tokenizer: <s> = 1, </s> = 2.
                bosTokenId: 1, eosTokenId: 2,
                headDim: text["head_dim"] as? Int ?? hidden / heads,
                hiddenSize: hidden, intermediateSize: try int(text, "intermediate_size"),
                numAttentionHeads: heads, numHiddenLayers: try int(text, "num_hidden_layers"),
                numKeyValueHeads: try int(text, "num_key_value_heads"),
                rmsNormEps: try float(text, "rms_norm_eps"),
                ropeTheta: try (rope["rope_theta"] as? Double).map(Float.init) ?? float(text, "rope_theta"),
                yarn: yarn,
                llama4ScalingBeta: (rope["llama_4_scaling_beta"] as? Double).map(Float.init),
                tieWordEmbeddings: text["tie_word_embeddings"] as? Bool ?? true,
                vocabSize: try int(text, "vocab_size"), quantization: quant)
        default:
            throw LlmError("unsupported model_type \"\(modelType)\"")
        }
    }
}

public struct LlmError: Error, CustomStringConvertible {
    public let description: String
    init(_ s: String) { description = s }
}

/// RoPE with YaRN-interpolated frequencies, after mlx-lm's `YarnRoPE`.
private final class YarnRoPE {
    let dims: Int
    let mscale: Float
    let freqs: MLXArray

    init(dims: Int, base: Float, yarn: LlmConfig.Yarn) {
        self.dims = dims
        func correctionDim(_ rotations: Float) -> Float {
            Float(dims) * log(Float(yarn.originalMaxPositions) / (rotations * 2 * .pi)) / (2 * log(base))
        }
        func getMscale(_ scale: Float, _ m: Float) -> Float {
            scale <= 1 ? 1 : 0.1 * m * log(scale) + 1
        }
        let low = max(floor(correctionDim(yarn.betaFast)), 0)
        var high = min(ceil(correctionDim(yarn.betaSlow)), Float(dims - 1))
        if low == high { high += 0.001 }
        mscale = getMscale(yarn.factor, yarn.mscale) / getMscale(yarn.factor, yarn.mscaleAllDim)
        let freqExtra = pow(MLXArray(base), MLXArray(stride(from: 0, to: Int32(dims), by: 2)).asType(.float32) / Float(dims))
        let freqInter = freqExtra * yarn.factor
        let ramp = clip((MLXArray(0..<Int32(dims / 2)).asType(.float32) - low) / (high - low), min: 0, max: 1)
        let mask = 1 - ramp
        freqs = (freqInter * freqExtra) / (freqInter * mask + freqExtra * (1 - mask))
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let x = mscale != 1 ? x * mscale : x
        return MLXFast.RoPE(x, dimensions: dims, traditional: false, base: nil, scale: 1, offset: offset, freqs: freqs)
    }
}

private class Mlp: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ cfg: LlmConfig) {
        self._gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class Attention: Module {
    let headDim: Int
    let scale: Float
    let rope: RoPE?
    let yarnRope: YarnRoPE?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ cfg: LlmConfig) {
        let headDim = cfg.headDim
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qProj.wrappedValue = Linear(cfg.hiddenSize, cfg.numAttentionHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(cfg.hiddenSize, cfg.numKeyValueHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(cfg.hiddenSize, cfg.numKeyValueHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(cfg.numAttentionHeads * headDim, cfg.hiddenSize, bias: false)
        if let yarn = cfg.yarn {
            self.yarnRope = YarnRoPE(dims: headDim, base: cfg.ropeTheta, yarn: yarn)
            self.rope = nil
        } else {
            self.rope = RoPE(dimensions: headDim, traditional: false, base: cfg.ropeTheta)
            self.yarnRope = nil
        }
    }

    private func applyRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let yarnRope { return yarnRope(x, offset: offset) }
        return rope!(x, offset: offset)
    }

    func callAsFunction(_ x: MLXArray, attnScale: MLXArray?, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, T, -1, headDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        queries = applyRope(queries, offset: offset)
        keys = applyRope(keys, offset: offset)
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }
        if let attnScale { queries = queries * attnScale }
        var mask = mask
        if let m = mask {
            let maskLen = m.dim(-1)
            if keys.dim(2) < maskLen {
                mask = m[0..., (maskLen - keys.dim(2))...]
            }
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, T, -1)  // heads * headDim, not always hiddenSize
        return oProj(out)
    }
}

private class Layer: Module {
    @ModuleInfo(key: "mlp") var mlp: Mlp
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: LlmConfig) {
        self._mlp.wrappedValue = Mlp(cfg)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._postAttnNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._selfAttn.wrappedValue = Attention(cfg)
    }

    func callAsFunction(_ x: MLXArray, attnScale: MLXArray?, mask: MLXArray?, cache: KVCache) -> MLXArray {
        var x = x + selfAttn(inputNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        x = x + mlp(postAttnNorm(x))
        return x
    }
}

public class LlmModel: Module {
    public let cfg: LlmConfig
    private let norm: RMSNorm
    private let layers: [Layer]
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ cfg: LlmConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numHiddenLayers).map { _ in Layer(cfg) }
        self.norm = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self._embedTokens.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self._lmHead.wrappedValue = cfg.tieWordEmbeddings ? nil : Linear(cfg.hiddenSize, cfg.vocabSize, bias: false)
    }

    /// `x` is `[B, T]` of token ids; returns the logits of the last position, `[B, vocab]`.
    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embedTokens(x)
        let mask = cache.first?.createAttentionMask(h: h)
        var attnScale: MLXArray? = nil
        if let beta = cfg.llama4ScalingBeta, let yarn = cfg.yarn {
            let offset = cache.first?.offset ?? 0
            let positions = MLXArray(Int32(offset)..<Int32(offset + x.dim(1))).asType(.float32)
            let scaling = 1 + beta * log(1 + floor(positions / Float(yarn.originalMaxPositions)))
            attnScale = scaling.reshaped(1, 1, -1, 1).asType(h.dtype)
        }
        for (layer, c) in zip(layers, cache) {
            h = layer(h, attnScale: attnScale, mask: mask, cache: c)
        }
        h = norm(h[0..., -1, 0...])
        if let lmHead { return lmHead(h) }
        return embedTokens.asLinear(h)
    }

    public func makeCache() -> [KVCache] {
        (0..<cfg.numHiddenLayers).map { _ in
            KVCacheSimple(headDim: .init(cfg.headDim), kvHeads: cfg.numKeyValueHeads)
        }
    }

    /// Builds the model from an mlx-community checkpoint folder (`config.json` +
    /// `model.safetensors`). Vision weights of multimodal checkpoints are ignored.
    public static func load(from folder: URL) throws -> LlmModel {
        let cfg = try LlmConfig.load(from: folder)
        let raw = try loadArrays(url: folder.appending(path: "model.safetensors"))
        var weights: [String: MLXArray] = [:]
        for (key, value) in raw {
            if key.hasPrefix("vision_tower.") || key.hasPrefix("multi_modal_projector.") { continue }
            var k = key
            for prefix in ["language_model.model.", "language_model.", "model."] where k.hasPrefix(prefix) {
                k = String(k.dropFirst(prefix.count))
                break
            }
            weights[k] = value
        }
        let model = LlmModel(cfg)
        if let q = cfg.quantization {
            // Only the layers that carry `scales` are quantized (mlx_lm keeps some, like
            // the Ministral embedding, in bf16).
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}

/// Greedy decoding with a KV cache. Prompt tokens are processed in one pass.
public final class LlmGenerator {
    let model: LlmModel
    public var maxTokens: Int
    public var stopTokens: Set<Int>
    private let stopped = Mutex(false)

    public init(_ model: LlmModel, maxTokens: Int = 1024) {
        self.model = model
        self.maxTokens = maxTokens
        self.stopTokens = [model.cfg.eosTokenId, model.cfg.bosTokenId]
    }

    /// Thread-safe: ends `generate` after the current token.
    public func stop() { stopped.withLock { $0 = true } }

    /// Calls `onToken` for each generated token (stop tokens excluded); returns false to stop.
    public func generate(prompt: [Int], onToken: (Int) -> Bool) {
        stopped.withLock { $0 = false }
        let cache = model.makeCache()
        var logits = model(MLXArray(prompt.map { Int32($0) }).reshaped(1, -1), cache: cache)
        for _ in 0..<maxTokens {
            let next = logits.argMax(axis: -1)
            eval(next)
            let token = next.item(Int32.self)
            if stopTokens.contains(Int(token)) || stopped.withLock({ $0 }) { break }
            if !onToken(Int(token)) { break }
            logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
        }
        GPU.clearCache()
    }
}
