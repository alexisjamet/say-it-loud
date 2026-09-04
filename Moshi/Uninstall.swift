// macOS only: the sandbox container survives dragging the app to the Trash, so the
// model (about 1.4 GB) and the transcripts would stay behind. This wipes them, then
// shows the app in the Finder so the user can trash it, and quits.

#if os(macOS)
    import AppKit
    import Foundation

    enum Uninstaller {
        /// Files the app created, all inside its own container.
        static var dataURLs: [URL] {
            let fm = FileManager.default
            var urls: [URL] = []
            if let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
                urls.append(support.appending(path: "models"))
                urls.append(support.appending(path: "transcripts.json"))
            }
            if let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
                urls.append(caches)
            }
            return urls
        }

        /// Bytes used by `dataURLs`, for the confirmation text.
        static func dataSize() -> Int64 {
            let fm = FileManager.default
            var total: Int64 = 0
            for url in dataURLs {
                guard let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
                    if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize { total += Int64(size) }
                    continue
                }
                for case let f as URL in e {
                    if let size = try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize { total += Int64(size) }
                }
            }
            return total
        }

        /// Asks with an app-modal AppKit alert. A SwiftUI `.alert` inside the menu bar
        /// panel closes with the panel and its button action never runs.
        @MainActor
        static func confirm(_ s: Strings) {
            let alert = NSAlert()
            alert.messageText = s.uninstallTitle
            alert.informativeText = s.uninstallMessage(
                ByteCountFormatter.string(fromByteCount: dataSize(), countStyle: .file))
            alert.alertStyle = .critical
            alert.addButton(withTitle: s.uninstallConfirm).hasDestructiveAction = true
            alert.addButton(withTitle: s.cancel)
            NSApp.activate(ignoringOtherApps: true)
            guard alert.runModal() == .alertFirstButtonReturn else { return }
            run()
        }

        /// Deletes the data, reveals the app bundle in the Finder, quits.
        @MainActor
        static func run() {
            let fm = FileManager.default
            for url in dataURLs where fm.fileExists(atPath: url.path) {
                do {
                    try fm.removeItem(at: url)
                    NSLog("uninstall: deleted %@", url.path)
                } catch {
                    NSLog("uninstall: could not delete %@: %@", url.path, String(describing: error))
                }
            }
            // The sandbox forbids trashing our own bundle in /Applications: hand over to the Finder.
            NSWorkspace.shared.activateFileViewerSelecting([Bundle.main.bundleURL])
            // Leave the current event (menu action / modal) before terminating.
            DispatchQueue.main.async { NSApp.terminate(nil) }
        }
    }
#endif
