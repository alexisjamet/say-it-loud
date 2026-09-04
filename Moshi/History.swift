// Local history of transcripts, persisted as JSON in the app's Application Support.

import Foundation

struct Transcript: Identifiable, Codable, Equatable {
    let id: UUID
    let date: Date
    var text: String
}

@Observable
@MainActor
final class HistoryStore {
    static let shared = HistoryStore()
    private(set) var items: [Transcript] = []  // newest first
    private let maxItems = 500

    private let fileURL: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return dir.appending(path: "transcripts.json")
    }()

    private init() {
        load()
    }

    @discardableResult
    func add(_ text: String) -> Transcript {
        let t = Transcript(id: UUID(), date: Date(), text: text)
        items.insert(t, at: 0)
        if items.count > maxItems { items.removeLast(items.count - maxItems) }
        save()
        return t
    }

    func update(_ id: UUID, text: String) {
        guard let i = items.firstIndex(where: { $0.id == id }), items[i].text != text else { return }
        items[i].text = text
        save()
    }

    func delete(_ id: UUID) {
        items.removeAll { $0.id == id }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL) else { return }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        items = (try? decoder.decode([Transcript].self, from: data)) ?? []
    }

    private func save() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(items).write(to: fileURL, options: .atomic)
        } catch {
            print("could not save history: \(error)")
        }
    }
}
