// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
#if os(macOS)
    import Carbon.HIToolbox
#endif
import Foundation
import SwiftUI

func requestMicrophoneAccess() {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return
    case .notDetermined:
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            print("granted", granted)
        }
    case .denied:  // The user has previously denied access.
        return
    case .restricted:  // The user can't grant access due to restrictions.
        return
    case _:
        return
    }
}

@main
struct moshiApp: App {
    @Environment(\.scenePhase) var scenePhase
    #if os(macOS)
        @State private var transcriber = Transcriber.shared
        // ⌘F6 anywhere: start recording, or stop and copy the transcript.
        private static let hotKey = HotKey(keyCode: kVK_F6, modifiers: cmdKey) {
            Task { @MainActor in Transcriber.shared.hotKeyToggle() }
        }
    #endif

    init() {
        requestMicrophoneAccess()
        #if os(macOS)
            _ = Self.hotKey
            Notifier.shared.setup()
        #endif
    }

    #if os(macOS)
        private var menuBarSymbol: String {
            switch transcriber.phase {
            case .recording: "waveform.circle.fill"
            case .loading, .finishing: "ellipsis.circle"
            case .idle: "mic.fill"
            }
        }
    #endif

    var body: some Scene {
        #if os(macOS)
            // Lives in the menu bar: no window, no Dock icon (LSUIElement is set for macOS).
            MenuBarExtra {
                SttView()
                    .frame(width: 380, height: 520)
            } label: {
                Image(systemName: menuBarSymbol)
            }
            .menuBarExtraStyle(.window)
        #else
            WindowGroup {
                ContentView()
                    .environment(DeviceStat())
            }
        #endif
    }
}
