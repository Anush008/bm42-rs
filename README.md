<div align="center">
  <h1><a href="https://crates.io/crates/bm42">BM42-rs 🦀</a></h1>
 <h3>Rust implementation of <a href="https://qdrant.tech/articles/bm42/" target="_blank">Qdrant BM42</a></h3>
  <a href="https://crates.io/crates/bm42"><img src="https://img.shields.io/crates/v/bm42.svg" alt="Crates.io"></a>
  <a href="https://github.com/Anush008/bm42-rs/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-apache-blue.svg" alt="Apache Licensed Licensed"></a>
</div>

## 🔍 Not looking for Rust?

- Python 🐍: [BM42 with FastEmbed](https://github.com/qdrant/fastembed)
- JavaScript 🌐: [BM42-js](https://github.com/Anush008/bm42-js)

📥 Installation

Run the following command in your project directory:

```bash
cargo add bm42
```

Or add the following line to your Cargo.toml:

```toml
[dependencies]
bm42 = "0"
```

## 📖 Usage

### Generating Sparse Embeddings

```rust
use bm42::{BM42Options, BM42};

// With default InitOptions
let bm42 = BM42::try_new(Default::default()).unwrap();

// With custom BM42Options
let bm42_options = BM42Options {
    alpha: 0.5,
    show_download_progress: true,
    ..Default::default()
};

let texts = vec![
    "It's a truth universally acknowledged that a zombie in possession of brains must be in want of more brains.",
    "We're not in Infinity; we're in the suburbs.",
    "I was a thousand times more evil than thou!",
    "History is merely a list of surprises... It can only prepare us to be surprised yet again.",
];

// Generate embeddings for indexing
let doc_embeddings = bm42.embed(texts).unwrap();

// Generate embeddings for querying
let query_embeddings = bm42.query_embed(texts).unwrap();
```

## 📄 LICENSE

Apache 2.0 © [2024](https://github.com/Anush008/bm42-rs/blob/master/LICENSE)
