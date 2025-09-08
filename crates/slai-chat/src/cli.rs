use crate::chat_template::ChatTemplate;
use crate::init_wgpu;
use crate::llm::ChatLlm;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use async_std::sync::RwLock;
use clap::Parser;
use colored::Colorize;
use slai::gguf::Gguf;
use slai::models::segment_anything::SamGgmlFile;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Cli {
    /// Path to the GGUF model to load.
    pub path: Option<PathBuf>,

    /// Image input for models that need it (like segment-anything).
    #[arg(long)]
    pub image: Option<PathBuf>,

    /// If `true` the app will run without a GPU.
    #[arg(long, default_value_t = false)]
    pub headless: bool,

    /// If `true`, details of the GGUF file will be printed.
    #[arg(long, default_value_t = false)]
    pub inspect: bool,
}

pub async fn run_headless(cli: &Cli) -> anyhow::Result<()> {
    let gpu = init_wgpu().await?;
    let Some(path) = &cli.path else {
        println!("{}", "No model file provided, exiting.".red());
        return Ok(());
    };
    println!("{}", format!("Loading GGUF file: {:?}", cli.path).dimmed());
    let t_gguf = std::time::Instant::now();
    let file = File::open(path).expect("Unable to open the GGUF model file");
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = {
        if path.extension().map(|ext| ext.to_str().unwrap()) == Some("bin") {
            println!("Attempting to load file as legacy GGML format.");
            SamGgmlFile::from_bytes(&mmap[..])
                .map(|ggml| Ok(ggml.into_gguf()))
                .unwrap_or_else(|e| {
                    println!("Error loading legacy GGML format: {}", e);
                    println!("Trying GGUF loader.");
                    Gguf::from_bytes(&mmap[..])
                })?
        } else {
            Gguf::from_bytes(&mmap[..])?
        }
    };
    println!(
        "{}",
        format!(
            "GGUF model loaded in {:.2} seconds.",
            t_gguf.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    if cli.inspect {
        gguf.print_metadata();
        gguf.print_tensors();
    }

    let t_chat_llm = std::time::Instant::now();
    let llm = Arc::new(RwLock::new(
        ChatLlm::from_gguf(&*gpu.backend, &gpu.compiler, &gguf, Some(cli)).await?,
    ));
    let chat_template = ChatTemplate::from_gguf(&gguf);
    println!(
        "{}",
        format!(
            "Uploaded model to GPU in {:.2} seconds.",
            t_chat_llm.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    println!("{}", "Starting interactive chat:".dimmed());
    let mut prompt = Prompt::default();
    let mut next_pos = 0;
    let mut tok_per_second = 0.0;
    let sampler = SamplerParams::default();

    loop {
        // Read stdin.
        println!("{}", "[User]".purple().bold());
        let mut user_prompt = String::new();
        std::io::stdout().flush()?;
        std::io::stdin().read_line(&mut user_prompt)?;
        user_prompt.truncate(user_prompt.trim_end().len());
        prompt.append_user(user_prompt);

        // Forward the transformer.
        let (snd, rcv) = async_channel::unbounded();

        {
            let gpu = gpu.clone();
            let llm = llm.clone();
            let prompt = prompt.clone();
            let chat_template = chat_template.clone();
            async_std::task::spawn(async move {
                llm.write()
                    .await
                    .forward(
                        &*gpu.backend,
                        prompt,
                        sampler,
                        chat_template,
                        next_pos,
                        |msg| Ok(snd.send_blocking(msg)?),
                    )
                    .await
            });
        }

        let mut full_response = String::new();
        let mut last_tok = String::new();
        println!("{}", "[Assistant]".green().bold());

        while let Ok(event) = rcv.recv().await {
            if let ChatEvent::Token {
                string,
                next_pos: next,
                token_count,
                token_time,
            } = event
            {
                let tps = token_count as f64 / token_time;

                // Don’t print multiple newlines, takes too
                // much room on the console.
                if last_tok != "\n" || string != "\n" {
                    print!("{}", string);
                }
                next_pos = next;
                tok_per_second = tps;
                full_response.push_str(&string);
                last_tok = string;
                std::io::stdout().flush()?;
            }
        }
        println!();
        println!(
            "{}",
            format!(
                "({:.2} tok/s) − generated {} tokens",
                tok_per_second, next_pos
            )
            .italic()
            .dimmed()
        );

        prompt.append_assistant(full_response);
    }
}
