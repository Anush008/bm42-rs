use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::Path;
use std::{fs::File, path::PathBuf};

use anyhow::Result;
use hf_hub::api::sync::ApiRepo;
use murmur3::murmur3_32;
use rust_stemmers::Stemmer;
use std::io::Cursor;
use std::io::{self, BufRead};
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, TruncationParams};

pub const DEFAULT_CACHE_DIR: &str = ".bm42_cache";

// Tokenizer files for "bring your own" embedding models
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerFiles {
    pub tokenizer_file: Vec<u8>,
    pub config_file: Vec<u8>,
    pub special_tokens_map_file: Vec<u8>,
    pub tokenizer_config_file: Vec<u8>,
}

pub fn load_tokenizer_hf_hub(
    model_repo: &ApiRepo,
    max_length: usize,
) -> Result<(Tokenizer, HashMap<String, u32>)> {
    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file: read_file_to_bytes(&model_repo.get("tokenizer.json")?)?,
        config_file: read_file_to_bytes(&model_repo.get("config.json")?)?,
        special_tokens_map_file: read_file_to_bytes(&model_repo.get("special_tokens_map.json")?)?,

        tokenizer_config_file: read_file_to_bytes(&model_repo.get("tokenizer_config.json")?)?,
    };

    load_tokenizer(tokenizer_files, max_length)
}

pub fn load_tokenizer(
    tokenizer_files: TokenizerFiles,
    max_length: usize,
) -> Result<(Tokenizer, HashMap<String, u32>)> {
    let base_error_message = "Error building TokenizerFiles. Could not read {} file.";

    // Serialise each tokenizer file
    let config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "config.json"),
            )
        })?;
    let special_tokens_map: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.special_tokens_map_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "special_tokens_map.json"),
            )
        })?;
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&tokenizer_files.tokenizer_config_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer_config.json"),
            )
        })?;
    let mut tokenizer: tokenizers::Tokenizer =
        tokenizers::Tokenizer::from_bytes(tokenizer_files.tokenizer_file).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                base_error_message.replace("{}", "tokenizer.json"),
            )
        })?;

    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .expect("Error reading model_max_length from tokenizer_config.json")
        as f32;
    let max_length = max_length.min(model_max_length as usize);
    let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
    let pad_token = tokenizer_config["pad_token"]
        .as_str()
        .expect("Error reading pad_token from tokenier_config.json")
        .into();

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)?
        .clone();

    let mut special_token_to_id: HashMap<String, u32> = HashMap::new();

    if let serde_json::Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if value.is_string() {
                let token = value.as_str().unwrap();
                tokenizer.add_special_tokens(&[AddedToken {
                    content: token.into(),
                    special: true,
                    ..Default::default()
                }]);
                special_token_to_id.insert(token.into(), tokenizer.token_to_id(token).unwrap());
            } else if value.is_object() {
                let token = value["content"].as_str().unwrap();
                tokenizer.add_special_tokens(&[AddedToken {
                    content: token.into(),
                    special: value["special"].as_bool().unwrap_or(true),
                    single_word: value["single_word"].as_bool().unwrap(),
                    lstrip: value["lstrip"].as_bool().unwrap(),
                    rstrip: value["rstrip"].as_bool().unwrap(),
                    normalized: value["normalized"].as_bool().unwrap(),
                }]);
                special_token_to_id.insert(token.into(), tokenizer.token_to_id(token).unwrap());
            }
        }
    }
    Ok((tokenizer, special_token_to_id))
}

fn read_file_to_bytes(file: &PathBuf) -> Result<Vec<u8>> {
    let mut file = File::open(file)?;
    let file_size = file.metadata()?.len() as usize;
    let mut buffer = Vec::with_capacity(file_size);
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

// This type was inferred using IDE hints
// Turned into a type alias for type hinting in struct BM42
pub type Tokenizer = tokenizers::TokenizerImpl<
    tokenizers::ModelWrapper,
    tokenizers::NormalizerWrapper,
    tokenizers::PreTokenizerWrapper,
    tokenizers::PostProcessorWrapper,
    tokenizers::DecoderWrapper,
>;

pub fn lines_from_file(filename: impl AsRef<Path>) -> io::Result<Vec<String>> {
    BufReader::new(File::open(filename)?).lines().collect()
}

pub fn rescore_vector(vector: &HashMap<String, f32>, alpha: f32) -> HashMap<u32, f32> {
    let mut new_vector: HashMap<u32, f32> = HashMap::new();

    for (token, &value) in vector.iter() {
        let token_id = murmur3_32(&mut Cursor::new(token), 0).unwrap();

        let new_score = (1.0 + value).ln().powf(alpha);

        new_vector.insert(token_id, new_score);
    }

    new_vector
}

pub fn query_rehash(tokes: Vec<String>) -> HashMap<u32, f32> {
    tokes
        .into_iter()
        .map(|tok| {
            let token_id = murmur3_32(&mut Cursor::new(tok), 0).unwrap();
            (token_id, 1.0)
        })
        .collect()
}

pub fn reconstruct_bpe(
    bpe_tokens: impl IntoIterator<Item = (usize, String)>,
    special_tokens: &[String],
) -> Vec<(String, Vec<usize>)> {
    let mut result = Vec::new();
    let mut acc = String::new();
    let mut acc_idx = Vec::new();

    let continuing_subword_prefix = "##";
    let continuing_subword_prefix_len = continuing_subword_prefix.len();

    for (idx, token) in bpe_tokens {
        if special_tokens.contains(&token) {
            continue;
        }

        if token.starts_with(continuing_subword_prefix) {
            acc.push_str(&token[continuing_subword_prefix_len..]);
            acc_idx.push(idx);
        } else {
            if !acc.is_empty() {
                result.push((acc.clone(), acc_idx.clone()));
                acc_idx = vec![];
            }
            acc = token;
            acc_idx.push(idx);
        }
    }

    if !acc.is_empty() {
        result.push((acc, acc_idx));
    }

    result
}

pub fn aggregate_weights(tokens: &[(String, Vec<usize>)], weights: &[f32]) -> Vec<(String, f32)> {
    let mut result: Vec<(String, f32)> = Vec::new();

    for (token, idxs) in tokens.iter() {
        let sum_weight: f32 = idxs.iter().map(|&idx| weights[idx]).sum();
        result.push((token.clone(), sum_weight));
    }

    result
}

pub fn filter_pair_tokens(
    tokens: Vec<(String, Vec<usize>)>,
    stopwords: &[String],
    punctuation: &[String],
) -> Vec<(String, Vec<usize>)> {
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();

    for (token, value) in tokens.into_iter() {
        if stopwords.contains(&token) || punctuation.contains(&token) {
            continue;
        }
        result.push((token.clone(), value));
    }

    result
}

pub fn stem_pair_tokens(
    stemmer: &Stemmer,
    tokens: Vec<(String, Vec<usize>)>,
) -> Vec<(String, Vec<usize>)> {
    let mut result: Vec<(String, Vec<usize>)> = Vec::new();

    for (token, value) in tokens.into_iter() {
        let processed_token = stemmer.stem(&token).to_string();
        result.push((processed_token, value));
    }

    result
}

pub fn pooled_attention(
    model_output: &[Vec<Vec<f32>>],
    attention_mask: &[Vec<u32>],
) -> Vec<Vec<f32>> {
    model_output
        .iter()
        .zip(attention_mask)
        .map(|(output, mask)| {
            let mean = output
                .iter()
                .fold(vec![0.0; output[0].len()], |acc, inner_vec| {
                    acc.iter().zip(inner_vec).map(|(&a, &b)| a + b).collect()
                });
            let mean = mean
                .iter()
                .map(|&sum| sum / output.len() as f32)
                .collect::<Vec<f32>>();

            mean.iter().zip(mask).map(|(&m, &a)| m * a as f32).collect()
        })
        .collect()
}
