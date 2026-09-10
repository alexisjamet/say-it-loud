# Say It Loud

On-device speech-to-text for macOS and iOS, built on Kyutai's
[`stt-1b-en_fr`](https://huggingface.co/kyutai/stt-1b-en_fr) model running with
[MLX Swift](https://github.com/ml-explore/mlx-swift). English and French.

Everything stays on your device: the model (about 1.4 GB, downloaded once from
Hugging Face) and your transcripts are never uploaded.

## Rewriting with a local LLM

Once a transcript is on screen, a row of presets rewrites it on device with a
small LLM (MLX, 4-bit): fix punctuation and filler words, shorten it for a
friend, turn it into an email or a bulleted list, translate to English or
French, or type your own instruction. Undo restores the dictated text. The
model, Ministral 3 3B (about 2.7 GB, French-native), is downloaded on first
use. On iPhone the speech model is unloaded while the LLM runs, and reloaded
for the next recording.

The same pipeline can be tried from the command line:

```
xcodebuild -scheme moshi-cli -derivedDataPath build
build/Build/Products/Release/MoshiCLI run-rewrite --task email "euh bonjour je voulais dire que…"
```

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
iPhone. The project builds in Swift 6 language mode (strict concurrency).
Signing uses your own team; the bundle identifier is
`com.alexisjamet.sil`, change it to yours.

The iOS build is deliberately not offered to Macs ("Designed for iPad" is
off), the native menu bar app is the Mac version.

## Credits

This is a fork of [kyutai-labs/moshi-swift](https://github.com/kyutai-labs/moshi-swift),
which provides the streaming Mimi codec and Moshi model implementations in MLX
Swift. The original README is in the git history. Licensed under the MIT
license, see [LICENSE](LICENSE).
