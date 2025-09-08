use async_std::sync::RwLock;
// use crate::chat_gpt2::ChatGpt2;
use crate::chat_llama2::ChatLlama2;
use crate::chat_template::ChatTemplate;
use crate::cli::Cli;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use crate::segment_anything::SegmentAnything;
use slai::gguf::{Gguf, GgufMetadataValue};
use slai::models::gpt2::Gpt2Tokenizer;
use slai::models::llama2::{LlamaModelType, LlamaTokenizer};
use slang_hal::backend::Backend;
use slang_hal::re_exports::minislang::SlangCompiler;

pub enum ChatLlm<B: Backend> {
    Llama(ChatLlama2<B>),
    Qwen(ChatLlama2<B>),
    Sam(RwLock<SegmentAnything<B>>),
}

impl<B: Backend> ChatLlm<B> {
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "llama",
            Self::Qwen(_) => "qwen2",
            Self::Sam(_) => "segment-anything",
        }
    }
}

impl<B: Backend> ChatLlm<B> {
    pub async fn from_gguf(
        backend: &B,
        compiler: &SlangCompiler,
        gguf: &Gguf,
        _cli: Option<&Cli>,
    ) -> anyhow::Result<Self> {
        let Some(GgufMetadataValue::String(name)) = gguf.metadata.get("general.architecture")
        else {
            anyhow::bail!("Unrecognized model")
        };

        if name.to_lowercase().contains("llama") {
            Ok(Self::Llama(ChatLlama2::from_gguf(backend, compiler, gguf)?))
        } else if name.to_lowercase().contains("qwen2") {
            Ok(Self::Qwen(ChatLlama2::from_gguf_with_model_type(
                backend,
                compiler,
                gguf,
                LlamaModelType::Qwen2,
            )?))
        } else if name.to_lowercase().contains("sam") {
            Ok(Self::Sam(RwLock::new(SegmentAnything::from_gguf(
                backend, compiler, gguf,
            )?)))
        } else {
            anyhow::bail!("Unrecognized model")
        }
    }

    pub async fn forward(
        &self,
        backend: &B,
        prompt: Prompt,
        sampler_params: SamplerParams,
        chat_template: ChatTemplate,
        next_pos: usize,
        out: impl Fn(ChatEvent) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        match self {
            Self::Llama(llm) => {
                llm.forward(
                    backend,
                    prompt,
                    sampler_params,
                    chat_template,
                    next_pos,
                    out,
                )
                .await
            }
            Self::Qwen(llm) => {
                llm.forward(
                    backend,
                    prompt,
                    sampler_params,
                    chat_template,
                    next_pos,
                    out,
                )
                .await
            }
            Self::Sam(_llm) => {
                todo!()
            }
        }
    }
}

pub enum AnyTokenizer {
    Llama(LlamaTokenizer),
    Gpt2(Gpt2Tokenizer),
}

impl AnyTokenizer {
    pub fn from_gguf(gguf: &Gguf) -> anyhow::Result<Self> {
        let tokenizer_type = gguf
            .metadata
            .get("tokenizer.ggml.model")
            .ok_or(anyhow::anyhow!("Missing tokenizer.ggml.model"))?
            .as_string();
        if tokenizer_type == "gpt2" {
            Ok(AnyTokenizer::Gpt2(Gpt2Tokenizer::from_gguf(gguf)))
        } else if tokenizer_type == "llama" {
            Ok(AnyTokenizer::Llama(LlamaTokenizer::from_gguf(gguf)))
        } else {
            anyhow::bail!("Unrecognized tokenizer type: {}", tokenizer_type)
        }
    }

    pub fn eos(&self) -> usize {
        match self {
            Self::Llama(t) => t.eos(),
            Self::Gpt2(t) => t.eos(),
        }
    }

    #[allow(dead_code)]
    pub fn bos(&self) -> usize {
        match self {
            Self::Llama(t) => t.bos(),
            Self::Gpt2(t) => t.bos(),
        }
    }

    pub fn bos_str(&self) -> &str {
        match self {
            Self::Llama(t) => t.bos_str(),
            Self::Gpt2(t) => t.bos_str(),
        }
    }

    pub fn eos_str(&self) -> &str {
        match self {
            Self::Llama(t) => t.eos_str(),
            Self::Gpt2(t) => t.eos_str(),
        }
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        match self {
            Self::Llama(t) => t.decode(prev_token, token),
            Self::Gpt2(t) => t.decode(&[token as u32]),
        }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        match self {
            Self::Llama(t) => t.encode(text, bos, eos),
            // TODO: auto-instert bos/eos based on the flag?
            Self::Gpt2(t) => t.encode(text),
        }
    }
}
