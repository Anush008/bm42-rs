mod utils;

use crate::utils::{load_tokenizer_hf_hub, Tokenizer, DEFAULT_CACHE_DIR};

use anyhow::Result;
use core::str;
use hf_hub::{api::sync::ApiBuilder, Cache};
use ndarray::{s, Array};
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

use rust_stemmers::{Algorithm, Stemmer};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    thread::available_parallelism,
};
use utils::{
    aggregate_weights, filter_pair_tokens, lines_from_file, pooled_attention, query_rehash,
    reconstruct_bpe, rescore_vector, stem_pair_tokens,
};

const MAX_LENGTH: usize = 512;
const MODEL_REPO: &str = "Qdrant/all_miniLM_L6_v2_with_attentions";
const MODEL_FILE: &str = "model.onnx";
const STOPWORDS_FILE: &str = "stopwords.txt";

impl BM42 {
    /// Try to create a new BM42 instance
    pub fn try_new(options: BM42Options) -> Result<BM42> {
        let BM42Options {
            execution_providers,
            cache_dir,
            show_download_progress,
            alpha,
        } = options;

        let threads = available_parallelism()?.get();

        let cache = Cache::new(cache_dir);
        let api = ApiBuilder::from_cache(cache)
            .with_progress(show_download_progress)
            .build()
            .expect("Failed to build API from cache");
        let model_repo = api.model(MODEL_REPO.to_string());

        let model_file_reference = model_repo
            .get(MODEL_FILE)
            .unwrap_or_else(|_| panic!("Failed to retrieve model file: {}", MODEL_FILE));

        let session = Session::builder()?
            .with_execution_providers(execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(model_file_reference)?;

        let (tokenizer, special_token_to_id) = load_tokenizer_hf_hub(&model_repo, MAX_LENGTH)?;

        let stop_words_file = model_repo
            .get(STOPWORDS_FILE)
            .unwrap_or_else(|_| panic!("Failed to retrieve model file: {}", STOPWORDS_FILE));

        let stopwords: Vec<String> = lines_from_file(stop_words_file).unwrap();

        Ok(Self::new(
            tokenizer,
            session,
            special_token_to_id,
            alpha,
            stopwords,
        ))
    }

    // Private constructor
    fn new(
        tokenizer: Tokenizer,
        session: Session,
        special_token_to_id: HashMap<String, u32>,
        alpha: f32,
        stopwords: Vec<String>,
    ) -> Self {
        let (special_tokens, special_tokens_ids): (Vec<String>, Vec<u32>) = special_token_to_id
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .unzip();
        let stemmer = Stemmer::create(Algorithm::English);
        let punctuation = &[
            "!", "\"", "#", "$", "%", "&", "\'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";",
            "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
        ];
        let punctuation: Vec<String> = punctuation.iter().map(|x| x.to_string()).collect();
        let punctuation = punctuation.to_vec();
        let invert_vocab: HashMap<u32, String> = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(key, value)| (value, key))
            .collect();

        Self {
            tokenizer,
            session,
            special_tokens,
            special_tokens_ids,
            alpha,
            punctuation,
            stemmer,
            invert_vocab,
            stopwords,
        }
    }

    /// Embed a list of texts for ingesting
    pub fn embed<S: AsRef<str>>(&self, texts: Vec<S>) -> Result<Vec<SparseEmbedding>> {
        let mut results: Vec<SparseEmbedding> = Vec::with_capacity(texts.len());

        let onnx_output = self.onnx_embed(texts)?;

        let OnnxOutput {
            model_output,
            attention_masks,
            input_ids,
        } = onnx_output;

        let pooled = pooled_attention(&model_output, &attention_masks);

        for i in 0..input_ids.len() {
            let document_token_ids = &input_ids[i];
            let attention_value = &pooled[i];

            let doc_tokens_with_ids: Vec<(usize, String)> = document_token_ids
                .iter()
                .enumerate()
                .map(|(idx, &id)| (idx, self.invert_vocab[&id].clone()))
                .collect();

            let reconstructed = reconstruct_bpe(doc_tokens_with_ids, &self.special_tokens.clone());

            let filtered = filter_pair_tokens(reconstructed, &self.stopwords, &self.punctuation);

            let stemmed = stem_pair_tokens(&self.stemmer, filtered);

            let weighted = aggregate_weights(&stemmed, attention_value);

            let mut max_token_weight: HashMap<String, f32> = HashMap::new();

            weighted.into_iter().for_each(|(token, weight)| {
                let weight = max_token_weight.get(&token).unwrap_or(&0.0).max(weight);
                max_token_weight.insert(token, weight);
            });

            let rescored = rescore_vector(&max_token_weight, self.alpha);

            let (indices, values): (Vec<i32>, Vec<f32>) = rescored.into_iter().unzip();

            results.push(SparseEmbedding { indices, values });
        }

        Ok(results)
    }

    // Embed a list of texts for querying
    pub fn query_embed<S: AsRef<str>>(&self, texts: Vec<S>) -> Result<Vec<SparseEmbedding>> {
        let mut results: Vec<SparseEmbedding> = Vec::with_capacity(texts.len());

        for text in texts {
            let encoded = self.tokenizer.encode(text.as_ref(), true).unwrap();
            let doc_tokens_with_ids: Vec<(usize, String)> = encoded
                .get_tokens()
                .iter()
                .enumerate()
                .map(|(idx, token)| (idx, token.clone()))
                .collect();
            let reconstructed = reconstruct_bpe(doc_tokens_with_ids, &self.special_tokens.clone());

            let filtered = filter_pair_tokens(reconstructed, &self.stopwords, &self.punctuation);

            let stemmed = stem_pair_tokens(&self.stemmer, filtered);

            let rehashed = query_rehash(stemmed.into_iter().map(|(token, _)| token).collect());

            let (indices, values): (Vec<i32>, Vec<f32>) = rehashed.into_iter().unzip();

            results.push(SparseEmbedding { indices, values });
        }

        Ok(results)
    }

    // Private method to embed texts using the ONNX model
    fn onnx_embed<S: AsRef<str>>(&self, texts: Vec<S>) -> Result<OnnxOutput> {
        let inputs = texts.iter().map(|d| d.as_ref()).collect();

        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .expect("Failed to encode batch");

        let encoding_length = encodings[0].len();
        let batch_size = texts.len();

        let max_size = encoding_length * batch_size;

        let mut ids_flattened = Vec::with_capacity(max_size);

        let mut input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut attention_masks: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        encodings.iter().for_each(|encoding| {
            let ids = encoding.get_ids();
            ids_flattened.extend(ids.iter().map(|x| *x as i64));

            input_ids.push(ids.to_vec());
            attention_masks.push(encoding.get_attention_mask().to_vec());
        });

        let inputs_ids_array = Array::from_shape_vec((batch_size, encoding_length), ids_flattened)?;

        let session_inputs = (ort::inputs![
            "input_ids" => Value::from_array(inputs_ids_array.clone()).unwrap()
        ])
        .unwrap();

        let outputs = self.session.run(session_inputs)?;

        let output_data = outputs["attention_6"].try_extract_tensor::<f32>()?;

        let mut outputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(batch_size);

        // TODO: Is there a built-in to do this?
        for i in 0..batch_size {
            let output: Vec<Vec<f32>> = output_data
                .view()
                .slice(s![i, .., 0, ..])
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();

            outputs.push(output);
        }

        Ok(OnnxOutput {
            model_output: outputs,
            attention_masks,
            input_ids,
        })
    }
}

#[allow(dead_code)]
pub struct BM42 {
    tokenizer: Tokenizer,
    session: Session,
    special_tokens: Vec<String>,
    special_tokens_ids: Vec<u32>,
    alpha: f32,
    punctuation: Vec<String>,
    invert_vocab: HashMap<u32, String>,
    stemmer: Stemmer,
    stopwords: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BM42Options {
    pub alpha: f32,
    pub execution_providers: Vec<ExecutionProviderDispatch>,
    pub cache_dir: PathBuf,
    pub show_download_progress: bool,
}

impl Default for BM42Options {
    fn default() -> Self {
        Self {
            execution_providers: Default::default(),
            cache_dir: Path::new(DEFAULT_CACHE_DIR).to_path_buf(),
            show_download_progress: true,
            alpha: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    pub indices: Vec<i32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct OnnxOutput {
    model_output: Vec<Vec<Vec<f32>>>,
    attention_masks: Vec<Vec<u32>>,
    input_ids: Vec<Vec<u32>>,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_embed() {
        let options = BM42Options::default();
        let bm42 = BM42::try_new(options).unwrap();

        let texts = vec![
            "You can't always get what you want.",
            "People are strange when you're a stranger.",
        ];

        let embeddings = bm42.embed(texts).unwrap();
        assert_eq!(embeddings.len(), 2);
    }

    #[test]
    fn test_query_embed() {
        let options = BM42Options::default();
        let bm42 = BM42::try_new(options).unwrap();

        let query_texts = vec!["The quick brown fox jumps over the lazy dog."];

        let query_embeddings = bm42.query_embed(query_texts).unwrap();
        assert_eq!(query_embeddings.len(), 1);
    }
}
