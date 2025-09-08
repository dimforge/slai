# slai-chat

This is a basic chat interface for testing **slai**. Currently only supports:
- Models of the Llama family.
- Qwen 2.
- Segment Anything.

In order to compile and run `slai-chat`, be sure to define the `SLANG_DIR` environment variable:
1. Download the Slang compiler libraries for your platform: https://github.com/shader-slang/slang/releases/tag/v2025.16
2. Unzip the downloaded directory, and use its path as value to the `SLANG_DIR` environment variable: `SLANG_DIR=/path/to/slang`.
   Note that the variable must point to the root of the slang installation (i.e. the directory that contains `bin` and `lib`).

To run the GUI natively:
```bash
dx run --release --features desktop
```

To run the CLI version natively (the CLI doesn’t support segment-anything currently):
```bash
cargo run --release --features desktop -- --headless --inspect '/path/to/model.gguf'
```
