use tch::{Device, Tensor};
use tch::nn::{self, Module};

#[derive(Debug)]
struct CausalSelfAttention {
    device: Device,
    key_linear: nn::Linear,
    query_linear: nn::Linear,
    value_linear: nn::Linear,
}

impl CausalSelfAttention {
    fn new(vs: nn::Path, head_size: i64, n_embd: i64) -> Self {
        let linear_config = nn::LinearConfig { bias: false, ..Default::default() };
        let key_linear = nn::linear(&vs / "key_linear", n_embd, head_size, linear_config);
        let query_linear = nn::linear(&vs / "query_linear", n_embd, head_size, linear_config);
        let value_linear = nn::linear(&vs / "value_linear", n_embd, head_size, linear_config);
        Self { device: vs.device(), key_linear, query_linear, value_linear }
    }
}

impl Module for CausalSelfAttention  {

    fn forward(&self, x: &tch::Tensor) -> Tensor {
        let kind = x.kind();
        let (_b, t, _c) = x.size3().unwrap();

        let k = self.key_linear.forward(&x);
        let q = self.query_linear.forward(&x);
        let v = self.value_linear.forward(&x);

        //let (_k_b, _k_t, _k_hs) = x.size3().unwrap();
        //let k_shape = k.size();

        //let att: Tensor = q.matmul(&k.transpose(-2, -1)) / (*k_shape.last().unwrap() as f64).sqrt();
        //println!("att {:?}", att.size());
        let mask = Tensor::ones([t, t], (kind, self.device)).tril(0).reshape([ 1, t, t]);
        //println!("mask {:?}", mask.size());
        //let att = att.masked_fill(&mask.eq(0.), f64::NEG_INFINITY);
        //println!("att {:?}", att.size());
        //let y = att.softmax(-1, kind).matmul(&v);
        //println!("y {:?}", y.size());
        //let out = y.matmul(&v);
        //return out
        let out = Tensor::scaled_dot_product_attention(&q, &k, &v, Option::Some(mask), 0.1, false, Option::None);
        return out
    }
}

#[derive(Debug)]
struct Mlp {
    linear: nn::Linear,
    attn: CausalSelfAttention,
}

impl Mlp {
    fn new(vs: nn::Path, n_embd: i64, n_head: i64) -> Self {
        let head_size: i64 = n_embd / n_head;
        let attn =
            CausalSelfAttention::new(&vs / "attn", head_size, n_embd);
        let linear_config = nn::LinearConfig { bias: false, ..Default::default() };
        let linear = nn::linear(&vs / "linear", head_size, n_embd, linear_config);
        Self { linear, attn }
    }
}

impl Module for Mlp{
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = self.attn.forward(xs);
        let x = self.linear.forward(&x);
        let x = Tensor::dropout(&x, 0.1, false);
        return x
    }
}
#[derive(Debug)]
struct FeedForward {
    linear_1: nn::Linear,
    linear_2: nn::Linear,
}

impl FeedForward {
    fn new(vs: nn::Path, n_embd: i64) -> Self {
        let linear_1 = nn::linear(&vs / "forward_linear_1", n_embd, n_embd * 4, Default::default());
        let linear_2 = nn::linear(&vs / "forward_linear_2", n_embd * 4, n_embd, Default::default());

        Self { linear_1, linear_2 }
    }
}
impl Module for FeedForward{
    fn forward(&self, xs: &tch::Tensor) -> Tensor {
        let x =  self.linear_1.forward(&xs);
        let x = Tensor::gelu(&x, "none");
        let x = self.linear_2.forward(&x);
        let x = Tensor::dropout(&x, 0.1, false);
        return x
    }
}
#[derive(Debug)]
struct Block {
    layer_norm_1: nn::LayerNorm,
    mlp: Mlp,
    layer_norm_2: nn::LayerNorm,
    feedforward: FeedForward,
}

impl Block {
    fn new(vs: nn::Path, n_embd: i64, n_head: i64) -> Self {
        let mlp = Mlp::new(&vs / "multi_head", n_embd, n_head);
        let layer_norm_1 = nn::layer_norm(&vs / "layer_norm_1", vec![n_embd], Default::default());
        let layer_norm_2 = nn::layer_norm(&vs / "layer_norm_2", vec![n_embd], Default::default());
        let feedforward = FeedForward::new(vs, n_embd);

        Self {
            layer_norm_1,
            mlp,
            layer_norm_2,
            feedforward,
        }
    }
}
impl Module for Block{
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs + self.mlp.forward(&self.layer_norm_1.forward(xs));
        let y = self.feedforward.forward(&self.layer_norm_2.forward(&x));
        return x + y;

    }
}

#[derive(Debug)]
pub(crate) struct Gpt {
    vocab_embed: nn::Embedding,
    pos_embed: nn::Embedding,
    blocks: Vec<Block>,
    layer_norm: nn::LayerNorm,
    linear_head: nn::Linear,
}

impl Gpt {
    pub(crate) fn new(vs: nn::Path, vocab_size: i64, n_embd: i64, block_size: i64, n_layer: i64, n_head: i64) -> Self {
        let linear_config = nn::LinearConfig { bias: false, ..Default::default() };


        let vocab_embed = nn::embedding(&vs / "vocab_embed",
                                        vocab_size,
                                        n_embd, Default::default());

        let pos_embed = nn::embedding(&vs / "pos_embed",
                                      block_size,
                                      n_embd, Default::default());

        let layer_norm = nn::layer_norm(&vs / "layer_norm",
                                        vec![n_embd],
                                        Default::default());

        let linear_head = nn::linear(&vs / "linear_head",
                                     n_embd,
                                     vocab_size,
                                     linear_config);
        let blocks = (0..n_layer)
            .map(|i| Block::new(&vs / "transformer" / "h" / i, n_embd, n_head))
            .collect::<Vec<_>>();

        Self {
            vocab_embed,
            pos_embed,
            blocks,
            layer_norm,
            linear_head,
        }
    }
}
impl Module for Gpt{
    fn forward(&self, x: &Tensor) -> Tensor {
        let device = x.device();
        let kind= x.kind();
        let (_b, t) = x.size2().unwrap();

        assert_eq!(t, 128, "error in input dimension");

        let posx = Tensor::arange(t,(kind, device));

        let tok_embed = self.vocab_embed.forward(x);
        let pos_embed = self.pos_embed.forward(&posx);

        let mut x = tok_embed + pos_embed;

        for block in self.blocks.iter() {

            x = block.forward(&x);
        }
        let x = self.layer_norm.forward(&x);
        let logits: Tensor = self.linear_head.forward(&x);
        return logits;
    }
}