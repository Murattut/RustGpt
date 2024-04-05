#[allow(dead_code)]
pub(crate) struct Config {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

#[allow(dead_code)]
impl Config {
    pub(crate) fn config_test() -> Self{
        Self { block_size: 64, vocab_size: 1000, n_layer: 4, n_head: 4, n_embd: 64 }
    }
    fn config_7b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 32, n_head: 32, n_embd: 4096 }
    }

    fn config_13b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 40, n_head: 40, n_embd: 5120 }
    }

    fn config_30b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 60, n_head: 52, n_embd: 6656 }
    }

    fn config_65b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 80, n_head: 64, n_embd: 8192 }
    }
}