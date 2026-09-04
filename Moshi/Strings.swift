// In-app language: English or French, chosen with a small switch. With no explicit
// choice we follow the system language, falling back to English.

import Foundation
import Observation

enum AppLanguage: String, CaseIterable, Identifiable {
    case english = "en"
    case french = "fr"
    var id: String { rawValue }
    /// Shown in its own language, so it stays readable whichever one is active.
    var name: String { self == .french ? "Français" : "English" }
    var locale: Locale { Locale(identifier: rawValue) }

    /// What the system is set to, if we support it.
    static var system: AppLanguage {
        let code = Locale.preferredLanguages.first ?? Locale.current.identifier
        return code.lowercased().hasPrefix("fr") ? .french : .english
    }
}

@Observable
@MainActor
final class Lang {
    static let shared = Lang()
    private static let key = "language"

    /// Explicit choice, nil means "same as the system".
    var choice: AppLanguage? {
        didSet { UserDefaults.standard.set(choice?.rawValue, forKey: Self.key) }
    }

    private init() {
        choice = UserDefaults.standard.string(forKey: Self.key).flatMap(AppLanguage.init(rawValue:))
    }

    var language: AppLanguage {
        get { choice ?? .system }
        set { choice = newValue }
    }

    var t: Strings { language == .french ? .fr : .en }
}

/// Every user-facing text, in one place per language.
struct Strings {
    let appName = "Say It Loud"
    static let repository = URL(string: "https://github.com/alexisjamet/say-it-loud")!

    // Hands-free flow (⌘F6) and notifications
    var recording: String
    var recordingBody: String
    var nothingHeard: String
    var nothingHeardBody: String
    var copiedToClipboard: String
    var copied: String

    // Status line
    var loadingModel: String
    var finishing: String
    var micDenied: String
    var unexpectedModel: String
    var micFailed: String
    var listening: String
    var failed: (String) -> String
    var downloading: (String, String) -> String
    var connecting: String
    var downloadHint: String

    // Buttons
    var clear: String
    var copy: String
    var share: String
    var history: String
    var closeHistory: String
    var more: String
    var quit: String
    var record: String
    var stop: String
    var delete: String
    var cancel: String

    // History
    var noTranscripts: String

    // Uninstall (macOS)
    var uninstall: String
    var uninstallTitle: String
    var uninstallConfirm: String
    var uninstallMessage: (String) -> String

    var privacy: String
    var language: String
    var sourceCode: String

    static let en = Strings(
        recording: "Recording",
        recordingBody: "Say It Loud is listening. Press ⌘F6 again to stop and copy.",
        nothingHeard: "Nothing heard",
        nothingHeardBody: "No speech was detected, the clipboard was left unchanged.",
        copiedToClipboard: "Copied to clipboard",
        copied: "Copied",
        loadingModel: "Loading model…",
        finishing: "Finishing…",
        micDenied: "Microphone access denied — allow it in Settings",
        unexpectedModel: "Unexpected model",
        micFailed: "Could not open the microphone",
        listening: "Listening",
        failed: { "Failed: \($0)" },
        downloading: { "Downloading model — \($0) / \($1) MB" },
        connecting: "Connecting to Hugging Face…",
        downloadHint: "1.4 GB in total, downloaded once — depending on your connection this can take several minutes.",
        clear: "Clear",
        copy: "Copy",
        share: "Share",
        history: "History",
        closeHistory: "Close history",
        more: "More",
        quit: "Quit Say It Loud",
        record: "Record",
        stop: "Stop",
        delete: "Delete",
        cancel: "Cancel",
        noTranscripts: "No transcripts yet",
        uninstall: "Uninstall Say It Loud…",
        uninstallTitle: "Uninstall Say It Loud?",
        uninstallConfirm: "Delete and Quit",
        uninstallMessage: {
            "This deletes the speech model and all your transcripts (\($0)), then quits and shows the app in the Finder so you can move it to the Trash."
        },
        privacy: "Everything stays on your device: the model and your transcripts are never uploaded, and no one else can access them.",
        language: "Language",
        sourceCode: "Source code on GitHub"
    )

    static let fr = Strings(
        recording: "Enregistrement",
        recordingBody: "Say It Loud vous écoute. Appuyez à nouveau sur ⌘F6 pour arrêter et copier.",
        nothingHeard: "Rien entendu",
        nothingHeardBody: "Aucune parole détectée, le presse-papiers n'a pas été modifié.",
        copiedToClipboard: "Copié dans le presse-papiers",
        copied: "Copié",
        loadingModel: "Chargement du modèle…",
        finishing: "Finalisation…",
        micDenied: "Accès au micro refusé — autorisez-le dans Réglages",
        unexpectedModel: "Modèle inattendu",
        micFailed: "Impossible d'ouvrir le micro",
        listening: "À l'écoute",
        failed: { "Échec : \($0)" },
        downloading: { "Téléchargement du modèle — \($0) / \($1) Mo" },
        connecting: "Connexion à Hugging Face…",
        downloadHint: "1,4 Go au total, téléchargé une seule fois — selon votre connexion, cela peut prendre plusieurs minutes.",
        clear: "Effacer",
        copy: "Copier",
        share: "Partager",
        history: "Historique",
        closeHistory: "Fermer l'historique",
        more: "Plus",
        quit: "Quitter Say It Loud",
        record: "Enregistrer",
        stop: "Arrêter",
        delete: "Supprimer",
        cancel: "Annuler",
        noTranscripts: "Aucune transcription pour l'instant",
        uninstall: "Désinstaller Say It Loud…",
        uninstallTitle: "Désinstaller Say It Loud ?",
        uninstallConfirm: "Supprimer et quitter",
        uninstallMessage: {
            "Cela supprime le modèle vocal et toutes vos transcriptions (\($0)), puis quitte l'app et l'affiche dans le Finder pour que vous la mettiez à la corbeille."
        },
        privacy: "Tout reste sur votre appareil : le modèle et vos transcriptions ne sont jamais envoyés, et personne d'autre n'y a accès.",
        language: "Langue",
        sourceCode: "Code source sur GitHub"
    )
}
