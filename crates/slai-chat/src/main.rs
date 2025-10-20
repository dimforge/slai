use crate::chat_template::ChatTemplate;
use crate::cli::Cli;
use crate::llm::ChatLlm;
use crate::prompt::Prompt;
use crate::sampler::SamplerParams;
use components::{Chat, Home};
use dioxus::prelude::*;
use slang_hal::backend::Backend;
use slang_hal::backend::WebGpu;
use slang_hal::re_exports::minislang::SlangCompiler;
use std::rc::Rc;
use std::sync::Arc;

// mod chat_gpt2;
mod chat_llama2;
mod chat_template;
mod components;
mod llm;
mod prompt;
mod sampler;

#[cfg(not(target_arch = "wasm32"))]
mod cli;
mod segment_anything;

#[cfg(feature = "cuda")]
pub type SelectedBackend = slang_hal::backend::Cuda;
#[cfg(not(feature = "cuda"))]
pub type SelectedBackend = WebGpu;

#[derive(Copy, Clone)]
pub struct UnsupportedBackend;

pub struct GpuInstanceCtx<B: Backend> {
    pub backend: Arc<B>,
    pub compiler: Rc<SlangCompiler>,
}

impl<B: Backend> Clone for GpuInstanceCtx<B> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend.clone(),
            compiler: self.compiler.clone(),
        }
    }
}

impl<B: Backend> GpuInstanceCtx<B> {
    pub fn new(backend: B) -> Self {
        let mut compiler = SlangCompiler::new(vec![]);
        slai::register_shaders(&mut compiler);

        Self {
            backend: Arc::new(backend),
            compiler: Rc::new(compiler),
        }
    }
}

pub type LoadedModelSignal = Signal<Option<LoadedModel<SelectedBackend>>>;

#[derive(Clone, Debug, Default)]
enum PromptResponse {
    #[default]
    Empty,
    Thinking,
    Responding(String),
}

#[derive(Default)]
struct PromptState {
    prompt: Prompt,
    response: PromptResponse,
}

#[derive(Clone)]
pub struct GgufMetadata {
    metadata: Vec<String>,
    tensors: Vec<String>,
}

pub struct LoadedModel<B: Backend> {
    pub llm: Arc<ChatLlm<B>>,
    pub sampler: SamplerParams,
    pub template: ChatTemplate,
    pub metadata: GgufMetadata,
}

impl<B: Backend> Clone for LoadedModel<B> {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone(),
            sampler: self.sampler,
            template: self.template.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

const FAVICON: Asset = asset!("/assets/slai-logo.png");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use clap::Parser;
        let cli = Cli::parse();
        if cli.headless {
            futures::executor::block_on(cli::run_headless(&cli)).unwrap();
            return;
        }
    }

    #[cfg(feature = "desktop")]
    {
        use dioxus::desktop::{tao, LogicalSize};
        let window =
            tao::window::WindowBuilder::default().with_inner_size(LogicalSize::new(1300.0, 900.0));
        dioxus::LaunchBuilder::new()
            .with_cfg(dioxus::desktop::Config::new().with_window(window))
            .launch(App);
    }

    #[cfg(not(feature = "desktop"))]
    {
        dioxus::launch(App);
    }
}

async fn init_wgpu() -> anyhow::Result<GpuInstanceCtx<WebGpu>> {
    let features = wgpu::Features::default();
    let limits = wgpu::Limits {
        max_buffer_size: 2_000_000_000,
        max_storage_buffer_binding_size: 2_000_000_000,
        ..Default::default()
    };

    let mut backend = WebGpu::new(features, limits).await?;
    backend.force_buffer_copy_src = true;
    Ok::<_, anyhow::Error>(GpuInstanceCtx::new(backend))
}

#[cfg(feature = "cuda")]
async fn init_cuda() -> anyhow::Result<GpuInstanceCtx<slang_hal::backend::Cuda>> {
    println!("loading cuda");
    let res = Ok::<_, anyhow::Error>(GpuInstanceCtx::new(slang_hal::backend::Cuda::new()?));
    println!("Loaded cuda");
    res
}

#[component]
fn App() -> Element {
    let gpu_wgpu = use_resource(init_wgpu);

    #[cfg(feature = "cuda")]
    let gpu_cuda = use_resource(init_cuda);
    #[cfg(not(feature = "cuda"))]
    let gpu_cuda = use_resource(|| async move { Ok::<_, anyhow::Error>(UnsupportedBackend) });

    match (&*gpu_wgpu.read_unchecked(), &*gpu_cuda.read_unchecked()) {
        (Some(Ok(gpu_wgpu)), Some(Ok(gpu_cuda))) => {
            use_context_provider(|| gpu_wgpu.clone());
            use_context_provider(|| *gpu_cuda);
            use_context_provider(|| LoadedModelSignal::new(None));
            use_context_provider(|| Signal::new(PromptState::default()));

            rsx! {
                // Global app resources
                document::Link { rel: "icon", href: FAVICON }
                document::Link { rel: "stylesheet", href: MAIN_CSS }
                document::Title { "slai chat" }

                if use_context::<LoadedModelSignal>().read().is_none() {
                    Home {}
                } else {
                    Chat {}
                }
            }
        }
        (Some(Err(e)), _) => {
            rsx! {
                p {
                    strong {
                        { format!("WebGPU is not supported on this browser: {e}.") }
                    }
                }
                p {
                    "See ",
                    a {
                        href: "https://caniuse.com/webgpu",
                        " caniuse.com"
                    },
                    " for a list of compatible browsers."
                }
            }
        }
        _ => rsx! {},
    }
}
