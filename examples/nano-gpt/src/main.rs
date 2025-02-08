use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use executorch::module::Module;
use executorch::tensor::{Tensor, TensorImpl};

fn main() {
    // Load the exported nanoGPT program, which was generated via the previous
    // steps.
    let mut model = Gpt2::new(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("nanogpt.pte"),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("vocab.json"),
    );

    // Set up the prompt. This provides the seed text for the model to elaborate.
    println!("Enter model prompt: ");
    let mut prompt = String::new();
    std::io::stdin().read_line(&mut prompt).unwrap();

    let max_input_tokens = 1024;
    let max_output_tokens = 30;
    println!("{}", prompt);
    model.generate(&prompt, max_input_tokens, max_output_tokens);
}

struct Gpt2 {
    model: Module,

    // The tokenizer is used to convert between tokens (used by the model) and
    // human-readable strings.
    tokenizer: BasicTokenizer,
}
impl Gpt2 {
    pub fn new(model_path: &Path, vocab_file: &Path) -> Self {
        let model = Module::new(
            model_path,
            Some(executorch::module::LoadMode::MmapUseMlockIgnoreErrors),
        );
        let tokenizer = BasicTokenizer::from_file(vocab_file);
        Self { model, tokenizer }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_input_length: usize,
        max_output_length: usize,
    ) -> String {
        // Convert the input text into a list of integers (tokens) that represents it,
        // using the string-to-token mapping that the model was trained on. Each token
        // is an integer that represents a word or part of a word.
        let mut input_tokens = self.tokenizer.encode(prompt);
        let mut output_tokens = Vec::new();

        for _ in 0..max_output_length {
            // Convert the input_tokens from a vector of int64_t to EValue. EValue is a
            // unified data type in the ExecuTorch runtime.
            // auto inputs = from_blob(
            //     input_tokens.data(),
            //     {1, static_cast<int>(input_tokens.size())},
            //     ScalarType::Long);
            let inputs_sizes = [1, input_tokens.len() as i32];
            let inputs_data = input_tokens.as_ptr();
            let inputs_dim_order = [0, 1];
            let inputs_strides = [input_tokens.len() as i32, 1];
            let inputs_tensor_impl = unsafe {
                TensorImpl::from_ptr(
                    &inputs_sizes,
                    inputs_data,
                    &inputs_dim_order,
                    &inputs_strides,
                )
            };
            let inputs_tensor = Tensor::new(&inputs_tensor_impl);

            // Run the model. It will return a tensor of logits (log-probabilities).
            let model_outputs = self
                .model
                .forward(&[inputs_tensor.into()])
                .expect("Failed to run model");

            // Convert the output logits from EValue to std::vector, which is what the
            // sampler expects.
            let logits = model_outputs[0]
                .as_tensor()
                .as_typed::<f32>()
                .as_array_dyn()
                .squeeze()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap()
                .to_vec();

            // Sample the next token from the logits.
            let next_token = Self::sample(&logits);

            // Break if we reached the end of the text.
            const ENDOFTEXT_TOKEN: i64 = 50256; // The value of the gpt2 `<|endoftext|>` token.
            if next_token == ENDOFTEXT_TOKEN {
                break;
            }

            // Add the next token to the output.
            output_tokens.push(next_token);

            print!("{}", self.tokenizer.decode(&[next_token]));
            std::io::stdout().flush().unwrap();

            // Update next input.
            input_tokens.push(next_token);
            if input_tokens.len() > max_input_length {
                input_tokens.remove(0);
            }
        }
        println!();

        // Convert the output tokens into a human-readable string.
        self.tokenizer.decode(&output_tokens)
    }

    fn sample(logits: &[f32]) -> i64 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as i64
    }
}

pub struct BasicTokenizer {
    encode: HashMap<String, i64>,
    decode: HashMap<i64, String>,
}
impl BasicTokenizer {
    pub fn from_file(file_path: &Path) -> Self {
        let file = std::fs::File::open(file_path).expect("Failed to open tokenizer file");
        let encode: HashMap<String, i64> =
            serde_json::from_reader(file).expect("Failed to parse tokenizer JSON file");
        let encode = encode
            .into_iter()
            .map(|(k, v)| {
                let k = k.replace("Ġ", " ").replace("Ċ", "\n");
                (k, v)
            })
            .collect::<HashMap<_, _>>();
        let decode: HashMap<i64, String> = encode.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self { encode, decode }
    }

    pub fn encode(&self, prompt: &str) -> Vec<i64> {
        let words = self.parse_prompt(prompt);
        words.iter().map(|word| self.encode[word]).collect()
    }

    pub fn decode(&self, indices: &[i64]) -> String {
        indices
            .iter()
            .map(|index| &self.decode[index])
            .fold(String::new(), |a, b| a + b)
    }

    fn parse_prompt(&self, prompt: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut word = String::new();
        for c in prompt.chars() {
            if c == ' ' {
                if !word.is_empty() {
                    result.push(word.clone());
                    word.clear();
                }
                word.push(c);
            } else if c.is_ascii_punctuation() {
                if !word.is_empty() {
                    result.push(word.clone());
                    word.clear();
                }
                result.push(c.to_string());
            } else {
                word.push(c);
            }
        }
        if !word.is_empty() {
            result.push(word);
        }
        result
    }
}
