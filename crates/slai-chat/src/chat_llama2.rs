use crate::chat_template::ChatTemplate;
use crate::llm::AnyTokenizer;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::{sample_next_token, SamplerParams};
use async_std::sync::RwLock;
// use dioxus::hooks::UnboundedSender;
use nalgebra::DVector;
use slang_hal::backend::Backend;
// use slang_hal::re_exports::bytemuck;
use slang_hal::re_exports::minislang::SlangCompiler;
// use slang_hal::re_exports::Device;
use slai::context::{LlmContext, LlmOps};
use slai::gguf::Gguf;
use slai::models::llama2::cpu::Llama2Config;
use slai::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType};
use slai::re_exports::stensor::shapes::ViewShapeBuffers;
use slai::tensor_cache::TensorCache;

pub struct ChatLlama2<B: Backend> {
    ops: LlmOps<B>,
    transformer: Llama2<B>,
    weights: Llama2Weights<B>,
    tokenizer: AnyTokenizer,
    config: Llama2Config,
    state: RwLock<Llama2State<B>>,
    shapes: RwLock<ViewShapeBuffers<B>>,
    tensor_cache: RwLock<TensorCache<B>>,
}

impl<B: Backend> ChatLlama2<B> {
    pub fn from_gguf(
        backend: &B,
        compiler: &SlangCompiler,
        gguf: &Gguf,
    ) -> anyhow::Result<ChatLlama2<B>> {
        Self::from_gguf_with_model_type(backend, compiler, gguf, LlamaModelType::Llama)
    }

    pub fn from_gguf_with_model_type(
        backend: &B,
        compiler: &SlangCompiler,
        gguf: &Gguf,
        model_type: LlamaModelType,
    ) -> anyhow::Result<ChatLlama2<B>> {
        let ops = LlmOps::new(backend, compiler)?;
        let transformer = Llama2::new(backend, compiler, model_type)?;
        let config = Llama2Config::from_gguf_with_model_type(gguf, model_type);
        let weights = Llama2Weights::from_gguf(backend, &config, gguf)?;
        let tokenizer = AnyTokenizer::from_gguf(gguf)?;
        let state = Llama2State::new(backend, &config)?;

        Ok(Self {
            ops,
            transformer,
            weights,
            tokenizer,
            config,
            state: RwLock::new(state),
            shapes: RwLock::new(ViewShapeBuffers::new(backend)),
            tensor_cache: RwLock::new(TensorCache::default()),
        })
    }

    pub async fn forward(
        &self,
        backend: &B,
        prompt: Prompt,
        sampler_params: SamplerParams,
        template: ChatTemplate,
        start_pos: usize,
        out: impl Fn(ChatEvent) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        log::info!("Original prompt:\n{}", prompt);

        let bos_str = self.tokenizer.bos_str();
        let eos_str = self.tokenizer.eos_str();
        println!("eos_str: {}, bos_str: {}", eos_str, bos_str);
        let prompt_str = template.apply(&prompt, bos_str, eos_str);
        println!("Forwarding prompt: ’’’{}’’’", prompt_str);
        if out(ChatEvent::TemplatedPrompt(prompt_str.clone())).is_err() {
            return Ok(());
        }

        let (mut sampler, mut sampler_res) = sampler_params.sampler();
        let prompt_toks = self.tokenizer.encode(&prompt_str, false, false);
        // let prompt_toks =
        //     self.tokenizer
        //         .encode(&prompt_str, !prompt_str.starts_with(&bos_str), false);
        log::info!("Promp tokens: {:?}", prompt_toks);

        let prompt_toks_map: Vec<_> = prompt_toks
            .iter()
            .map(|tok| {
                let tok_str = self.tokenizer.decode(0, *tok);
                (*tok, tok_str)
            })
            .collect();
        if out(ChatEvent::PromptTokens(prompt_toks_map)).is_err() {
            return Ok(());
        }

        // Skip the first token in the tok/s timing since it is particularly slow due to gpu initialization.
        let timing_delay = 1;

        let mut token = prompt_toks[start_pos];
        let mut start = None;
        let mut logits = DVector::zeros(self.config.vocab_size);

        for pos in start_pos.. {
            if pos == start_pos + timing_delay {
                start = Some(web_time::Instant::now());
            }

            // let t0 = std::time::Instant::now();
            self.forward_logits(backend, pos as u32, token as u32, &mut logits)
                .await?;
            // let elapsed = t0.elapsed().as_secs_f64();
            // println!("Logits time: {} (= {:.3} tok/s)", elapsed, 1.0 / elapsed);

            // let t0 = std::time::Instant::now();
            let next = sample_next_token(
                &mut sampler,
                &mut sampler_res,
                &mut logits,
                &prompt_toks,
                pos,
            );
            let token_string = self.tokenizer.decode(token, next);
            // println!("Sampling time: {}", t0.elapsed().as_secs_f64());
            token = next;

            if pos + 1 >= prompt_toks.len() {
                if token == self.tokenizer.eos() {
                    break;
                } else {
                    let (token_count, token_time) = if let Some(start) = &start {
                        (
                            pos - start_pos - timing_delay,
                            start.elapsed().as_secs_f64(),
                        )
                    } else {
                        (0, 0.0)
                    };

                    if out(ChatEvent::Token {
                        string: token_string,
                        next_pos: pos,
                        token_count,
                        token_time,
                    })
                    .is_err()
                    {
                        // Early-exit if an error was returned.
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    async fn forward_logits(
        &self,
        backend: &B,
        pos: u32,
        token: u32,
        out: &mut DVector<f32>,
    ) -> anyhow::Result<()> {
        let mut shapes = self.shapes.write().await;
        let mut tensor_cache = self.tensor_cache.write().await;
        let mut state = self.state.write().await;
        shapes.clear_tmp();
        tensor_cache.clear();

        let (rope_config, rms_norm_config, attn_params) = self.config.derived_configs(pos);

        // Run the transformer.
        let mut encoder = backend.begin_encoding();
        backend.write_buffer(state.rope_config_mut().buffer_mut(), &[rope_config])?;
        backend.write_buffer(state.rms_norm_config_mut().buffer_mut(), &[rms_norm_config])?;
        backend.write_buffer(state.attn_params_mut().buffer_mut(), &[attn_params])?;
        state
            .x
            .copy_from_view(&mut encoder, self.weights.token_embd.column(token))?;
        backend.submit(encoder)?;

        // let t0 = std::time::Instant::now();
        // let mut pass = encoder.begin_pass();

        let mut ctxt = LlmContext {
            backend,
            shapes: &mut *shapes,
            cache: &mut *tensor_cache,
            pass: None,
            encoder: None,
            ops: &self.ops,
        };
        ctxt.begin_submission();
        self.transformer.launch(
            &mut ctxt,
            &mut state,
            &self.weights,
            &self.config,
            &attn_params,
            pos,
        )?;
        // println!("queue time: {}", t0.elapsed().as_secs_f64());
        drop(ctxt.pass.take());

        let (logits, readback) = state.logits_and_readback_mut();
        readback.copy_from_view(ctxt.encoder.as_mut().unwrap(), logits)?;

        // DEBUG: uncomment if useful for debugging the transformer
        // self.state.xb_read.copy_from(&mut encoder, &self.state.xb);
        // self.state.q_read.copy_from(&mut encoder, &self.state.q);

        // let t0 = std::time::Instant::now();
        ctxt.submit();
        backend.synchronize()?; // TODO: is this needed?

        // println!("submit & sync time: {}", t0.elapsed().as_secs_f64());

        // let t0 = std::time::Instant::now();
        backend
            .read_buffer(state.logits_readback().buffer(), out.as_mut_slice())
            .await?;
        // println!("readback time: {}", t0.elapsed().as_secs_f64());

        // if pos == 1 {
        //     let debug = self.state.q_read.read(backend).await.unwrap();
        //     println!("debug: {:?}", &debug[120..130]);
        //     std::process::exit(0);
        // }

        Ok(())
    }
}
