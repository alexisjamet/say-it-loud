// Dictation screen: one big microphone button, the transcript shows up as
// editable text once recording stops. Mirrors the web page served by stt/server.py.

import AVFoundation
import MLX
import MoshiLib
import SwiftUI
import Synchronization
#if os(iOS)
    import UIKit
#endif

@Observable
@MainActor
final class Transcriber {
    enum Phase { case idle, loading, recording, finishing }

    /// One instance for the whole app: the view, the menu-bar icon and the hot key share it.
    static let shared = Transcriber()

    var phase: Phase = .idle
    var text = ""
    var status = ""
    let ev = Evaluator()
    let history = HistoryStore.shared
    /// History entry the editor is currently showing, if any.
    private(set) var current: Transcript?

    private var mic: MicrophoneCapture?
    private var copyWhenDone = false

    // 12.5 frames of 80 ms per second. The model expects a second of silence up
    // front; it emits words ~0.5 s late, so two seconds of silence flush them out.
    private let frameSize = 1920
    private let prefixFrames = 13
    private let flushFrames = 25

    var isBusy: Bool { phase == .loading || phase == .finishing }
    var download: (done: Int64, total: Int64)? { ev.download }

    func toggle() {
        switch phase {
        case .idle: start()
        case .recording: stop()
        default: break
        }
    }

    /// Hands-free flow (global shortcut): start, or stop and copy the result.
    func hotKeyToggle() {
        switch phase {
        case .idle:
            start()
            notify(Lang.shared.t.recording, Lang.shared.t.recordingBody)
        case .recording:
            copyWhenDone = true
            stop()
        default: break
        }
    }

    /// iOS opens straight into a recording; the download/permission steps run first if needed.
    func startOnLaunch() {
        guard phase == .idle, text.isEmpty else { return }
        start()
    }

    /// Stops a recording in progress (e.g. when the app goes to the background).
    func stopIfRecording() {
        if phase == .recording { stop() }
    }

    private func notify(_ title: String, _ body: String) {
        #if os(macOS)
            Notifier.shared.post(title, body)
        #endif
    }

    func clear() {
        text = ""
        current = nil  // the entry stays in the history
        if phase == .idle { status = "" }
    }

    /// Called by the view when the user edits the text.
    func textEdited() {
        guard let current else { return }
        history.update(current.id, text: text)
    }

    func open(_ transcript: Transcript) {
        guard phase == .idle else { return }
        text = transcript.text
        current = transcript
        status = ""
    }

    func delete(_ transcript: Transcript) {
        history.delete(transcript.id)
        if current?.id == transcript.id { current = nil }
    }

    func copy() {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        #if os(iOS)
            UIPasteboard.general.string = t
        #else
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(t, forType: .string)
        #endif
        flash(Lang.shared.t.copied)
    }

    private func flash(_ s: String) {
        status = s
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            if status == s && phase == .idle { status = "" }
        }
    }

    private func start() {
        phase = .loading
        status = Lang.shared.t.loadingModel
        text = ""
        current = nil
        Task { await run() }
    }

    private func stop() {
        guard phase == .recording else { return }
        phase = .finishing
        status = Lang.shared.t.finishing
        // Closing the mic queues an empty sentinel *after* everything it captured:
        // the loop keeps transcribing until it reaches it, so nothing said is dropped.
        mic?.close()
    }

    private func run() async {
        defer {
            phase = .idle
            setKeepAwake(false)
        }
        #if os(iOS)
            if AVAudioApplication.shared.recordPermission == .undetermined {
                _ = await AVAudioApplication.requestRecordPermission()
            }
            guard AVAudioApplication.shared.recordPermission == .granted else {
                status = Lang.shared.t.micDenied
                return
            }
        #endif
        do {
            let state = try await ev.load(.asr)
            guard let model = await state.perform({ $0 as? AsrModel }) else {
                status = Lang.shared.t.unexpectedModel
                return
            }
            let mic = MicrophoneCapture()
            self.mic = mic
            guard mic.startCapturing() else {
                status = Lang.shared.t.micFailed
                return
            }
            phase = .recording
            status = Lang.shared.t.listening
            setKeepAwake(true)

            let transcript = await runCaptureLoop(model.asr, mic: mic)
            text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { current = history.add(text) }
            status = ""
            if copyWhenDone {
                copyWhenDone = false
                if text.isEmpty {
                    status = Lang.shared.t.nothingHeard
                    notify(Lang.shared.t.nothingHeard, Lang.shared.t.nothingHeardBody)
                } else {
                    copy()
                    notify(Lang.shared.t.copiedToClipboard, String(text.prefix(140)))
                }
            }
        } catch {
            status = Lang.shared.t.failed("\(error)")
        }
        mic = nil
    }

    /// Runs the model on a dedicated thread: `mic.receive()` blocks, and MLX work
    /// should stay off the cooperative pool anyway.
    private func runCaptureLoop(_ asr: ASR, mic: MicrophoneCapture) async -> String {
        let frameSize = self.frameSize
        let prefixFrames = self.prefixFrames
        let flushFrames = self.flushFrames
        return await withCheckedContinuation { cont in
            let thread = Thread {
                var out = ""
                let silence = MLXArray([Float](repeating: 0, count: frameSize))[.newAxis, .newAxis]
                asr.reset()
                for _ in 0..<prefixFrames {
                    _ = asr.onPcmInput(silence)
                }
                var step = 0
                // Runs until the sentinel queued by `close()`, i.e. after the last captured buffer.
                while let pcm = mic.receive(), !pcm.isEmpty {
                    out += asr.onPcmInput(MLXArray(pcm)[.newAxis, .newAxis]).joined()
                    step += 1
                    if step % 128 == 0 { GPU.clearCache() }
                }
                for _ in 0..<flushFrames {
                    out += asr.onPcmInput(silence).joined()
                }
                GPU.clearCache()
                cont.resume(returning: out)
            }
            thread.name = "stt-capture"
            thread.qualityOfService = .userInteractive
            thread.start()
        }
    }

    private func setKeepAwake(_ on: Bool) {
        #if os(iOS)
            UIApplication.shared.isIdleTimerDisabled = on
        #endif
    }
}

struct SttView: View {
    @State private var t = Transcriber.shared
    @State private var lang = Lang.shared
    private var s: Strings { lang.t }
    @State private var showHistory = false
    @FocusState private var editing: Bool

    private var hasText: Bool { !t.text.isEmpty && t.phase == .idle }

    var body: some View {
        VStack(spacing: 20) {
            header

            if showHistory {
                HistoryList(store: t.history, current: t.current) { item in
                    t.open(item)
                    showHistory = false
                } onDelete: { item in
                    t.delete(item)
                }
                .frame(maxWidth: 640, maxHeight: .infinity)
            } else if hasText {
                TextEditor(text: $t.text)
                    .font(.system(size: 17))
                    .scrollContentBackground(.hidden)
                    .padding(12)
                    .padding(.trailing, 28)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color.field)
                    )
                    .overlay(alignment: .topTrailing) {
                        Button(action: { t.clear() }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 22))
                                .foregroundStyle(.secondary)
                                .padding(10)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(s.clear)
                    }
                    .focused($editing)
                    .frame(maxWidth: 640, maxHeight: .infinity)
            } else {
                Spacer()
            }

            if !showHistory {
            HStack(spacing: 24) {
                sideButton("doc.on.doc", label: s.copy, enabled: hasText) { t.copy() }
                micButton
                ShareLink(item: t.text.trimmingCharacters(in: .whitespacesAndNewlines)) {
                    sideIcon("square.and.arrow.up")
                }
                .buttonStyle(.plain)
                .disabled(!hasText)
                .opacity(hasText ? 1 : 0.3)
                .accessibilityLabel(s.share)
            }

            statusLine
                .frame(minHeight: 44)
            }


            if !hasText && !showHistory { Spacer() }

            privacyNote
        }
        .padding(16)
        .environment(\.locale, lang.language.locale)
        .onChange(of: t.text) { _, _ in t.textEdited() }
        #if os(iOS)
            .task { t.startOnLaunch() }
        #endif
        .animation(.easeInOut(duration: 0.2), value: hasText)
        .animation(.easeInOut(duration: 0.2), value: t.phase)
        .onTapGesture { editing = false }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image("Logo")
                .resizable()
                .scaledToFit()
                .frame(width: 44, height: 44)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            Text("Say It Loud")
                .font(.headline)
            Spacer()
            Button(action: { showHistory.toggle(); editing = false }) {
                Image(systemName: showHistory ? "xmark.circle.fill" : "clock.arrow.circlepath")
                    .font(.system(size: 20))
                    .foregroundStyle(.secondary)
                    .frame(width: 36, height: 36)
            }
            .buttonStyle(.plain)
            .disabled(t.phase != .idle)
            .accessibilityLabel(showHistory ? s.closeHistory : s.history)

            Menu {
                Picker(s.language, selection: $lang.language) {
                    ForEach(AppLanguage.allCases) { Text($0.name).tag($0) }
                }
                .pickerStyle(.inline)
                #if os(macOS)
                    Divider()
                    Button(s.uninstall, role: .destructive) { Uninstaller.confirm(s) }
                    Divider()
                    Button(s.quit) { NSApplication.shared.terminate(nil) }
                        .keyboardShortcut("q")
                #endif
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.system(size: 20))
                    .foregroundStyle(.secondary)
                    .frame(width: 36, height: 36)
            }
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .disabled(t.phase != .idle)
            .accessibilityLabel(s.more)
        }
        .frame(maxWidth: 640)
    }

    /// Everything happens on this device; say it, since users assume the opposite.
    private var privacyNote: some View {
        let compact = hasText || showHistory
        return HStack(alignment: .top, spacing: 6) {
            Image(systemName: "lock.shield")
            Text(s.privacy)
        }
        .font(compact ? .footnote : .callout)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.leading)
        .fixedSize(horizontal: false, vertical: true)  // never truncate, grow instead
        .frame(maxWidth: 480)
        .padding(.top, 4)
    }

    private var micButton: some View {
        #if os(macOS)
            let size: CGFloat = hasText ? 72 : 140
        #else
            let size: CGFloat = hasText ? 84 : 180
        #endif
        let recording = t.phase == .recording
        return Button(action: { t.toggle() }) {
            ZStack {
                Circle()
                    .fill(recording ? Color.red : Color.field)
                    .overlay(Circle().strokeBorder(recording ? Color.red : Color.hairline, lineWidth: 2))
                Image(systemName: "mic.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: size * 0.36)
                    .foregroundStyle(recording ? .white : .primary)
            }
            .frame(width: size, height: size)
            .opacity(t.isBusy ? 0.5 : 1)
            .modifier(Pulse(active: recording))
        }
        .buttonStyle(.plain)
        .disabled(t.isBusy)
        .accessibilityLabel(recording ? s.stop : s.record)
    }

    private func sideIcon(_ symbol: String) -> some View {
        Image(systemName: symbol)
            .font(.system(size: 22))
            .frame(width: 64, height: 64)
            .overlay(Circle().strokeBorder(Color.hairline, lineWidth: 1))
    }

    private func sideButton(_ symbol: String, label: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            sideIcon(symbol)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.3)
        .accessibilityLabel(label)
    }

    @ViewBuilder
    private var statusLine: some View {
        if let d = t.download, t.phase == .loading {
            VStack(spacing: 6) {
                if d.total > 0 {
                    ProgressView(value: Double(d.done), total: Double(d.total))
                        .frame(maxWidth: 260)
                    Text(s.downloading(mb(d.done), mb(d.total)))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView()
                        .frame(maxWidth: 260)
                    Text(s.connecting)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Text(s.downloadHint)
                    .multilineTextAlignment(.center)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        } else {
            Text(t.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

extension Color {
    /// Background of the editor and of the idle mic button.
    static var field: Color {
        #if os(iOS)
            Color(.secondarySystemBackground)
        #else
            Color(nsColor: .controlBackgroundColor)
        #endif
    }
    static var hairline: Color {
        #if os(iOS)
            Color(.separator)
        #else
            Color(nsColor: .separatorColor)
        #endif
    }
}

struct HistoryList: View {
    @State private var lang = Lang.shared
    private var s: Strings { lang.t }
    let store: HistoryStore
    let current: Transcript?
    let onOpen: (Transcript) -> Void
    let onDelete: (Transcript) -> Void

    var body: some View {
        if store.items.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 36))
                    .foregroundStyle(.tertiary)
                Text(s.noTranscripts)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List {
                ForEach(store.items) { item in
                    HStack(spacing: 8) {
                        Button(action: { onOpen(item) }) {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(item.date, format: .dateTime.day().month().year().hour().minute())
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text(item.text)
                                    .lineLimit(2)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)

                        Button(role: .destructive, action: { onDelete(item) }) {
                            Image(systemName: "trash")
                                .foregroundStyle(.red)
                                .frame(width: 32, height: 32)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(s.delete)
                    }
                    .listRowBackground(item.id == current?.id ? Color.accentColor.opacity(0.12) : nil)
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) { onDelete(item) } label: {
                            Label(s.delete, systemImage: "trash")
                        }
                    }
                    .contextMenu {
                        Button(role: .destructive) { onDelete(item) } label: {
                            Label(s.delete, systemImage: "trash")
                        }
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .background(RoundedRectangle(cornerRadius: 14).fill(Color.field))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
    }
}

private func mb(_ bytes: Int64) -> String {
    (Double(bytes) / 1_000_000).formatted(.number.precision(.fractionLength(0)))
}

/// Soft red halo that breathes while recording.
private struct Pulse: ViewModifier {
    let active: Bool
    @State private var on = false

    func body(content: Content) -> some View {
        content
            .background(
                Circle()
                    .fill(Color.red.opacity(active ? (on ? 0.18 : 0.04) : 0))
                    .padding(-18)
            )
            .onChange(of: active) { _, a in
                if a {
                    withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) { on = true }
                } else {
                    withAnimation(.easeOut(duration: 0.2)) { on = false }
                }
            }
    }
}

#Preview {
    SttView()
}
