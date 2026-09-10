// Loads a Hugging Face `tokenizer.json` + `tokenizer_config.json` pair with
// swift-transformers. Newer checkpoints (transformers v5, e.g. Ministral 3) name
// a `tokenizer_class` the library does not know; every model we ship uses a
// byte-level BPE, so the class is forced to the BPE implementation.

import Foundation
import Hub
import Tokenizers

public enum LlmTokenizer {
    public static func load(from folder: URL) throws -> any Tokenizer {
        func json(_ name: String) throws -> [NSString: Any] {
            let data = try Data(contentsOf: folder.appending(path: name))
            guard let dict = try JSONSerialization.jsonObject(with: data) as? [NSString: Any] else {
                throw LlmError("\(name) is not a JSON object")
            }
            return dict
        }
        var config = try json("tokenizer_config.json")
        config["tokenizer_class"] = "PreTrainedTokenizer"
        return try AutoTokenizer.from(tokenizerConfig: Config(config), tokenizerData: Config(try json("tokenizer.json")))
    }
}
