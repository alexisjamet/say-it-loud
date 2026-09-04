// Global keyboard shortcut for the macOS menu-bar app. Carbon hot keys work
// inside the sandbox and need no Accessibility permission, unlike NSEvent monitors.

#if os(macOS)
    import Carbon.HIToolbox
    import Foundation

    final class HotKey {
        private var hotKeyRef: EventHotKeyRef?
        private var handlerRef: EventHandlerRef?
        private let action: () -> Void

        /// `keyCode` is a kVK_* virtual key, `modifiers` a Carbon mask (cmdKey, optionKey, …).
        init(keyCode: Int, modifiers: Int, action: @escaping () -> Void) {
            self.action = action
            var spec = EventTypeSpec(
                eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
            InstallEventHandler(
                GetApplicationEventTarget(),
                { _, _, userData in
                    guard let userData else { return OSStatus(eventNotHandledErr) }
                    Unmanaged<HotKey>.fromOpaque(userData).takeUnretainedValue().action()
                    return noErr
                },
                1, &spec, Unmanaged.passUnretained(self).toOpaque(), &handlerRef)
            let id = EventHotKeyID(signature: OSType(0x53494C21) /* 'SIL!' */, id: 1)
            let status = RegisterEventHotKey(
                UInt32(keyCode), UInt32(modifiers), id, GetApplicationEventTarget(), 0, &hotKeyRef)
            if status != noErr {
                print("could not register the hot key (status \(status))")
            }
        }

        deinit {
            if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
            if let handlerRef { RemoveEventHandler(handlerRef) }
        }
    }
#endif
