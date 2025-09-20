# slai − on-device GPU LLM inference on every platform

<p align="center">
  <img src="./crates/slai-chat/assets/slai-logo.png" alt="crates.io" height="200px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

-----

**slai** is a set of [Rust](https://www.rust-lang.org/) libraries exposing [Slang](https://shader-slang.org/) shaders
and kernels for local Large Language Models (LLMs) inference on the GPU. It is cross-platform and runs on the web.
**slai** can be used as a rust library to assemble your own transformer from the provided operators (and write your
owns on top of it).

Aside from the library, two binary crates are provided:
- **slai-bench** is a basic benchmarking utility for measuring calculation times for matrix multiplication with various
  quantization formats.
- **slai-chat** is a basic chat GUI application for loading GGUF files and chat with the model. It can be run natively
  or on the browser. Check out its [README](./crates/slai-chat/README.md) for details on how to run it.

⚠️ **slai** is still under heavy development and might be lacking some important features. Contributions  are welcome!

----