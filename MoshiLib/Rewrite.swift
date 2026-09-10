// What the rewrite model is asked to do with a transcript. The prompt is built
// here so the CLI and the app share it; the user-facing labels live in the app.

import Foundation

public enum RewriteTask: Equatable, Sendable {
    /// Punctuation, spelling, filler words; wording otherwise unchanged.
    case fix
    /// Short and casual.
    case friend
    /// Polished email body.
    case email
    /// Bulleted key points.
    case bullets
    case toEnglish
    case toFrench
    /// Free instruction typed by the user.
    case custom(String)

    public var instruction: String {
        switch self {
        case .fix:
            "Correct the punctuation, capitalization, spelling and grammar, and remove filler words and hesitations (euh, hum, um, uh, bah, repeated words, false starts). Keep every sentence, the wording, the order and the length otherwise. Output plain prose, no list."
        case .friend:
            "Rewrite the text as a short, casual message to a friend, in the same language. Keep it natural, direct and to the point, and fix the mistakes."
        case .email:
            "Rewrite the text as a polished, well-structured email body in the same language, with a polite but natural tone, and fix the mistakes. Do not add a subject line, a closing formula or a signature that was not dictated."
        case .bullets:
            "Turn the text into a bulleted list in the same language, one line per point, each line starting with \"- \". Cover every fact, decision and request in the text; leave nothing out."
        case .toEnglish:
            "Translate the text into English. Reply with the English translation only."
        case .toFrench:
            "Translate the text into French. Reply with the French translation only."
        case .custom(let s):
            s
        }
    }

    public static let system = """
        You are a text rewriting tool, not a chat assistant. You receive a text that the user dictated, followed by an instruction. Apply the instruction to the text and output the resulting text only.
        Rules: plain text without markdown, bold or headings; no numbered list unless the instruction asks for a list; no notes, comments or explanations before or after the text; never add information, facts, times, names or details that are not in the text; keep the language of the text and its form of address (tu / vous, first names) unless the instruction says otherwise; never add a signature.
        """

    /// A worked example shown as a previous exchange: small models copy the format
    /// (no notes, no markdown) much more reliably than they follow rules. The example
    /// is in the language of the input, so it does not pull the answer into French.
    public func example(for text: String) -> (text: String, output: String)? {
        let fr = Self.looksFrench(text)
        switch self {
        case .fix:
            return fr
                ? ("euh salut c'est moi euh je passe te voir vers dix-huit heures euh ok",
                   "Salut, c'est moi, je passe te voir vers dix-huit heures, OK ?")
                : ("um hi it's me uh I'll come by to see you around six um ok",
                   "Hi, it's me, I'll come by to see you around six, OK?")
        case .friend:
            return fr
                ? ("euh salut c'est moi euh je voulais te dire que je passe te voir vers dix-huit heures si ça te va euh ok",
                   "Salut, je passe te voir vers 18h si ça te va, OK ?")
                : ("um hi it's me uh I wanted to tell you that I'll come by to see you around six if that works for you um ok",
                   "Hi, I'll come by around six if that works for you, OK?")
        case .email:
            return fr
                ? ("bonjour euh je vous envoie le rapport euh ce soir et euh on en parle demain merci",
                   "Bonjour,\n\nJe vous envoie le rapport ce soir et nous pourrons en parler demain.\n\nMerci.")
                : ("hello um I'll send you the report uh tonight and um we can talk about it tomorrow thanks",
                   "Hello,\n\nI will send you the report tonight and we can discuss it tomorrow.\n\nThanks.")
        case .bullets:
            return fr
                ? ("alors il faut acheter du pain et euh appeler le plombier ah et aussi payer la facture",
                   "- Acheter du pain\n- Appeler le plombier\n- Payer la facture")
                : ("so we need to buy bread and uh call the plumber oh and also pay the bill",
                   "- Buy bread\n- Call the plumber\n- Pay the bill")
        case .toEnglish:
            return ("euh je passe te voir vers dix-huit heures ok",
                    "I'll come by to see you around 6 pm, OK?")
        case .toFrench:
            return ("um I'll come by to see you around 6 pm ok",
                    "Je passe te voir vers 18 h, OK ?")
        case .custom:
            return nil
        }
    }

    /// Crude French / English detection on common words; French wins ties (the app's home).
    static func looksFrench(_ text: String) -> Bool {
        let french: Set<String> = ["le", "la", "les", "de", "des", "du", "et", "que", "qui", "je", "tu", "vous", "nous", "pas", "pour", "une", "un", "est", "il", "elle", "ce", "ça", "euh", "avec", "dans", "sur", "mais", "on"]
        let english: Set<String> = ["the", "and", "to", "is", "you", "we", "that", "of", "it", "in", "for", "i", "um", "uh", "yeah", "this", "with", "on", "be", "are", "not", "so"]
        var f = 0, e = 0
        for w in text.lowercased().split(whereSeparator: { !$0.isLetter && $0 != "'" }) {
            let word = String(w)
            if french.contains(word) { f += 1 }
            if english.contains(word) { e += 1 }
        }
        return f >= e
    }

    private func userMessage(_ text: String) -> String {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch self {
        case .toEnglish, .toFrench:
            return "Text:\n\(t)\n\nInstruction: \(instruction)"
        default:
            // Naming the language keeps a French-leaning model from translating English input.
            let lang = Self.looksFrench(t) ? "French" : "English"
            return "Text (in \(lang)):\n\(t)\n\nInstruction: \(instruction) Answer in \(lang)."
        }
    }

    /// The full prompt in Mistral's chat format, with the example as a first
    /// exchange. The instruction comes last in each message: a small model follows
    /// it better that way.
    public func prompt(for text: String) -> String {
        var p = "<s>[SYSTEM_PROMPT]\(Self.system)[/SYSTEM_PROMPT]"
        if let example = example(for: text) {
            p += "[INST]\(userMessage(example.text))[/INST]\(example.output)</s>"
        }
        return p + "[INST]\(userMessage(text))[/INST]"
    }
}
