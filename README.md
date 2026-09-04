# Say It Loud

On-device speech-to-text for macOS and iOS, built on Kyutai's
[`stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr) model running with
[MLX Swift](https://github.com/ml-explore/mlx-swift). English and French.

Everything stays on your device: the model (about 1.4 GB, downloaded once from
Hugging Face) and your transcripts are never uploaded.

## macOS

Say It Loud lives in the menu bar, with no window and no Dock icon.

- Press **⌘F6** anywhere to start recording. Press it again to stop: the
  transcript is copied to the clipboard and a notification confirms it.
- Click the menu bar icon to see the live transcript, edit it, copy or share
  it, and browse your history.

## iOS

Same engine, as a plain dictation app: record, review, edit, copy or share.
Transcripts are kept in a local history.

## Building

Open `moshi.xcodeproj` in Xcode and run the `Moshi` scheme on a Mac or an
iPhone. Signing uses your own team; the bundle identifier is
`com.alexisjamet.sil`, change it to yours.

The iOS build is deliberately not offered to Macs ("Designed for iPad" is
off), the native menu bar app is the Mac version.

## Credits

This is a fork of [kyutai-labs/moshi-swift](https://github.com/kyutai-labs/moshi-swift),
which provides the streaming Mimi codec and Moshi model implementations in MLX
Swift. The original README is in the git history. Licensed under the MIT
license, see [LICENSE](LICENSE).
