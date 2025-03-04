use std::collections::HashMap;
use std::fs::File;

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use executorch::evalue::{IntoEValue, Tag};
use executorch::module::{LoadMode, Module};
use executorch::ndarray::{ArrayView2, Ix1};
use executorch::tensor::TensorPtr;
use rand::distr::Distribution;
use rand::SeedableRng;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the model in flatbuffer format (.pte)
    #[arg(long)]
    model: PathBuf,

    /// Path to the tokenizer
    #[arg(long)]
    tokenizer: PathBuf,

    /// Prompt to generate text from
    #[arg(long)]
    prompt: String,

    /// Total number of tokens to generate (prompt + output).
    #[arg(long, default_value_t = 128)]
    length: usize,

    /// Sampling temperature.
    ///
    /// 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Random generator seed
    #[arg(long)]
    seed: Option<u64>,

    /// Verbose prints
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();
    executorch::platform::pal_init();

    let mut model = Module::new(&args.model, Some(LoadMode::File), None);
    let tokenizer = Tokenizer::from_json(&args.tokenizer);
    let mut rng = if let Some(seed) = args.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_os_rng()
    };

    let max_seq_len = model
        .execute("get_max_seq_len", &[])
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
        .as_i64() as usize;
    // let bos_token = model
    //     .execute("get_bos_id", &[])
    //     .unwrap()
    //     .into_iter()
    //     .next()
    //     .unwrap()
    //     .as_i64();

    if args.verbose {
        // println!("bos_token: {}", bos_token);
        println!("max_seq_len: {}", max_seq_len);

        for method_name in model.method_names().unwrap() {
            println!("method: {method_name}");
            let meta = model.method_meta(&method_name).unwrap();
            for input_idx in 0..meta.num_inputs() {
                println!("\tinput {input_idx}:");
                println!("\t\ttag {:?}", meta.input_tag(input_idx).unwrap());
                if meta.input_tag(input_idx).unwrap() == Tag::Tensor {
                    let tensor_info = meta.input_tensor_meta(input_idx).unwrap();
                    println!("\t\tscalar {:?}", tensor_info.scalar_type());
                    println!("\t\tsizes {:?}", tensor_info.sizes());
                    println!("\t\tdim_order {:?}", tensor_info.dim_order());
                }
            }
            for output_idx in 0..meta.num_outputs() {
                println!("\toutput {output_idx}");
                println!("\t\ttag {:?}", meta.output_tag(output_idx).unwrap());
                if meta.output_tag(output_idx).unwrap() == Tag::Tensor {
                    let tensor_info = meta.output_tensor_meta(output_idx).unwrap();
                    println!("\t\tscalar {:?}", tensor_info.scalar_type());
                    println!("\t\tsizes {:?}", tensor_info.sizes());
                    println!("\t\tdim_order {:?}", tensor_info.dim_order());
                }
            }
        }
    }

    let load_t0 = Instant::now();
    model.load_method("forward", None).unwrap();
    println!(
        "'forward' method loaded in {}s",
        load_t0.elapsed().as_millis()
    );

    let mut prompt_tokens = tokenizer.encode(&args.prompt);

    while prompt_tokens.len() < args.length {
        let input_tokens = if prompt_tokens.len() < max_seq_len {
            &prompt_tokens
        } else {
            &prompt_tokens[(prompt_tokens.len() - max_seq_len)..]
        };
        let input = TensorPtr::from_array_view(
            ArrayView2::from_shape((1, input_tokens.len()), input_tokens).unwrap(),
        );
        let outputs = model.forward(&[input.into_evalue()]).unwrap();
        assert_eq!(outputs.len(), 1);
        let logits = outputs[0]
            .as_tensor()
            .into_typed::<executorch::scalar::bf16>();
        let logits = logits
            .as_array()
            .squeeze()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let logits = logits.into_iter().map(|x| x.to_f32()).collect::<Vec<_>>();

        let probs = softmax(&logits, args.temperature);
        let next_token = rand::distr::weighted::WeightedIndex::new(probs)
            .unwrap()
            .sample(&mut rng) as i64;

        prompt_tokens.push(next_token);
        println!("{}", tokenizer.decode(&prompt_tokens));
    }
}

struct Tokenizer {
    str2idx: HashMap<String, i64>,
    idx2str: HashMap<i64, String>,
    special_tokens: HashMap<String, i64>,
}
impl Tokenizer {
    fn from_json(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let str2idx: HashMap<String, i64> = serde_json::from_reader(file).unwrap();

        let idx2str = str2idx
            .iter()
            .map(|(k, v)| (*v, k.replace('Ġ', " ").replace('Ċ', "\n")))
            .collect::<HashMap<_, _>>();
        let str2idx = idx2str
            .iter()
            .map(|(k, v)| (v.clone(), *k))
            .collect::<HashMap<_, _>>();

        let special_tokens = str2idx
            .iter()
            .filter(|(k, _)| k.starts_with("<|") && k.ends_with("|>"))
            .map(|(k, v)| (k.clone(), *v))
            .collect::<HashMap<_, _>>();

        Self {
            str2idx,
            idx2str,
            special_tokens,
        }
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut tokens = vec![];
        let orig_text = text;
        let mut text = text;
        while !text.is_empty() {
            let (token_str, token_id) = self
                .special_tokens
                .iter()
                .find(|(special_str, _)| text.starts_with(*special_str))
                .map(|(special_str, token)| (special_str.as_str(), *token))
                .unwrap_or_else(|| {
                    let first_char = &text[..text.chars().next().unwrap().len_utf8()];
                    let token = self.str2idx[first_char];
                    (first_char, token)
                });

            tokens.push(token_id);
            text = &text[token_str.len()..];
        }

        let mut tokens_next = vec![];
        let mut buf = String::new();
        while tokens.len() > 1 {
            tokens_next.clear();
            let mut changed = false;
            let mut i = 0;
            loop {
                let token1 = tokens[i];
                let token2 = tokens[i + 1];
                buf.clear();
                buf.push_str(&self.idx2str[&token1]);
                buf.push_str(&self.idx2str[&token2]);
                if let Some(token) = self.str2idx.get(&buf) {
                    tokens_next.push(*token);
                    i += 2;
                    changed = true;
                } else {
                    tokens_next.push(token1);
                    i += 1;
                }

                if i >= tokens.len() - 1 {
                    if i == tokens.len() - 1 {
                        tokens_next.push(tokens[i]);
                    }
                    break;
                }
            }
            std::mem::swap(&mut tokens, &mut tokens_next);
            if !changed {
                break;
            }
        }

        debug_assert_eq!(orig_text, self.decode(&tokens));

        tokens
    }

    fn decode(&self, tokens: &[i64]) -> String {
        tokens
            .iter()
            .map(|token| self.idx2str[token].as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}

fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = logits
        .iter()
        .map(|x| (x - max_logit).exp() / temperature)
        .collect::<Vec<_>>();
    let sum_exp: f32 = exps.iter().sum();
    exps.iter_mut().for_each(|x| *x /= sum_exp);
    exps
}
