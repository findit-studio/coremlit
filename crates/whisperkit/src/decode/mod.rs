//! Autoregressive decoding.
//!
//! Currently home to [`filter`], the logits-filter chain the decode loop
//! runs after every step's raw logits are produced; the loop itself
//! (Swift's `TextDecoder.decodeText`) lands in a later task.

pub mod filter;
