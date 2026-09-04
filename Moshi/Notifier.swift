// System notifications for the hands-free (⌘F6) flow on macOS: the panel is
// closed, so this is the only feedback the user gets.

#if os(macOS)
    import Foundation
    import UserNotifications

    @MainActor
    final class Notifier: NSObject, UNUserNotificationCenterDelegate {
        static let shared = Notifier()

        func setup() {
            let center = UNUserNotificationCenter.current()
            center.delegate = self
            center.requestAuthorization(options: [.alert, .sound]) { _, error in
                if let error { print("notification authorization: \(error)") }
            }
        }

        func post(_ title: String, _ body: String) {
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            let request = UNNotificationRequest(
                identifier: UUID().uuidString, content: content, trigger: nil)
            UNUserNotificationCenter.current().add(request)
        }

        // Show banners even when the app counts as "in the foreground" (panel open).
        nonisolated func userNotificationCenter(
            _ center: UNUserNotificationCenter, willPresent notification: UNNotification
        ) async -> UNNotificationPresentationOptions {
            [.banner, .sound]
        }
    }
#endif
