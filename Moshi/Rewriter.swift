// Rewrites a transcript with a local LLM (Ministral 3 3B, MLX): fix it, shorten
// it, turn it into an email or a list, translate it. The model is downloaded on
// first use.

import Foundation
import Hub
import MLX
import MoshiLib
import Tokenizers

/// The presets offered in the UI, in display order. Labels come from `Strings`.
enum RewritePreset: String, CaseIterable, Identifiable {
    case fix, friend, email, bullets, toEnglish, toFrench
    var id: String { rawValue }

    var task: RewriteTask {
        switch self {
        case .fix: .fix
        case .friend: .friend
        case .email: .email
        case .bullets: .bullets
        case .toEnglish: .toEnglish
        case .toFrench: .toFrench
        }
    }

    var symbol: String {
        switch self {
        case .fix: "text.badge.checkmark"
        case .friend: "message"
        case .email: "envelope"
        case .bullets: "list.bullet"
        case .toEnglish, .toFrench: "character.book.closed"
        }
    }

    func label(_ s: Strings) -> String {
        switch self {
        case .fix: s.presetFix
        case .friend: s.presetFriend
        case .email: s.presetEmail
        case .bullets: s.presetBullets
        case .toEnglish: s.presetEnglish
        case .toFrench: s.presetFrench
        }
    }
}

@Observable
@MainActor
final class Rewriter {
    private static let customKey = "rewriteCustomInstruction"
    /// mlx-community 4-bit export of Ministral 3 3B Instruct.
    static let repo = "mlx-community/Ministral-3-3B-Instruct-2512-4bit"
    static let files = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]
    /// Approximate download size, for the hint under the progress bar.
    static let sizeGB = "2.7"
    /// Last free-form instruction typed by the user, kept between launches.
    var customInstruction: String {
        didSet { UserDefaults.standard.set(customInstruction, forKey: Self.customKey) }
    }

    private var loaded: (model: LlmModel, tokenizer: any Tokenizer)?
    private var generator: LlmGenerator?

    init() {
        customInstruction = UserDefaults.standard.string(forKey: Self.customKey) ?? ""
    }

    var isLoaded: Bool { loaded != nil }

    /// Frees the model (iOS does this before recording: the speech model needs the memory).
    func unload() {
        loaded = nil
        generator = nil
        GPU.clearCache()
    }

    /// Downloads (through `ev`, so its progress shows in the status line) and builds the model.
    func load(ev: Evaluator) async throws {
        if loaded != nil { return }
        var folder: URL?
        for file in Self.files {
            folder = try await ev.downloadFromHub(id: Self.repo, filename: file).deletingLastPathComponent()
        }
        guard let folder else { throw CustomError("no model folder") }
        Self.removeOtherCheckpoints(next: folder)
        let tokenizer = try LlmTokenizer.load(from: folder)
        let model = try LlmModel.load(from: folder)
        loaded = (model, tokenizer)
    }

    /// Earlier versions offered other models; their weights (gigabytes) are dropped
    /// once the current one is in place.
    private static func removeOtherCheckpoints(next folder: URL) {
        let fm = FileManager.default
        let parent = folder.deletingLastPathComponent()  // …/models/mlx-community
        guard let siblings = try? fm.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil) else { return }
        for url in siblings where url.lastPathComponent != folder.lastPathComponent {
            try? fm.removeItem(at: url)
        }
    }

    /// Runs `task` on `text`. `onPartial` receives the text decoded so far, on the main
    /// actor; the returned string is the final result, trimmed.
    func run(_ task: RewriteTask, on text: String, onPartial: @escaping @MainActor (String) -> Void) async throws -> String {
        guard let (model, tokenizer) = loaded else { throw CustomError("model not loaded") }
        let promptTokens = tokenizer.encode(text: task.prompt(for: text), addSpecialTokens: false)
        let generator = LlmGenerator(model, maxTokens: min(2048, promptTokens.count * 2 + 256))
        self.generator = generator
        defer { self.generator = nil }
        let owned = UnsafeSendable(value: (generator, tokenizer))
        return await withCheckedContinuation { cont in
            // MLX work stays off the cooperative pool, like the transcription loop.
            let thread = Thread {
                let (generator, tokenizer) = owned.value
                var out: [Int] = []
                var lastShown = Date.distantPast
                generator.generate(prompt: promptTokens) { token in
                    out.append(token)
                    if Date().timeIntervalSince(lastShown) > 0.08 {
                        lastShown = Date()
                        let partial = tokenizer.decode(tokens: out, skipSpecialTokens: true)
                        Task { @MainActor in onPartial(partial) }
                    }
                    return true
                }
                let result = tokenizer.decode(tokens: out, skipSpecialTokens: true)
                cont.resume(returning: Self.clean(result))
            }
            thread.name = "rewrite"
            thread.qualityOfService = .userInitiated
            thread.start()
        }
    }

    func cancel() { generator?.stop() }

    /// Strips a thinking block if the model produced one anyway, a trailing
    /// parenthesized note ("(Note: …)"), markdown bold, and surrounding quotes.
    nonisolated static func clean(_ s: String) -> String {
        var t = s
        if let r = t.range(of: "</think>") { t = String(t[r.upperBound...]) }
        t = t.replacingOccurrences(of: "**", with: "")
        t = t.trimmingCharacters(in: .whitespacesAndNewlines)
        if let last = t.split(separator: "\n", omittingEmptySubsequences: false).last {
            let line = last.trimmingCharacters(in: .whitespaces)
            let inner = line.trimmingCharacters(in: CharacterSet(charactersIn: "*_"))
            if inner.count > 2, inner.hasPrefix("("), inner.hasSuffix(")"), t.count > line.count {
                t = String(t.dropLast(line.count)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        if t.count > 2, let f = t.first, let l = t.last, "\"“«".contains(f), "\"”»".contains(l) {
            t = String(t.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return t
    }
}
