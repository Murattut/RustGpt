use anyhow::Result;
use tch::{Device, Kind, nn};
use tch::nn::{Module, OptimizerConfig};

mod dataset;
mod model;
use crate::model::Gpt;

const LEARNING_RATE:f64 = 0.0001;

const EPOCHS:i64 = 100;

fn main()  -> Result<()> {
    let block_size=  128;
    let vocab_size=  968;
    let n_layer=  4;
    let n_head=  4;
    let n_embd=  128;
    let device = find_device();
    let vs = nn::VarStore::new(device);
    //vs.set_kind(Kind::Half);

    let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE)?;
    let model = Gpt::new(vs.root(), vocab_size, n_embd, block_size, n_layer, n_head);

    //let mut losses = 0.0;
    let mut temp_train_pointer = 0;
    let mut temp_val_pointer = 0;
    for epoch in 0..EPOCHS {


        let (xs, ys, train_file_pointer) = dataset::get_batch_train(temp_train_pointer);
        let logits = model.forward(&xs.to_kind(Kind::Int64).to_device(device));
        let (b, t, c) = logits.size3().unwrap();
        let logits = logits.view([b*t, c]);
        let target = &ys.to_kind(Kind::Int64).to_device(device);
        let targets = target.view([b*t]).to_device(device);
        let train_loss = logits.cross_entropy_for_logits(&targets);
        temp_train_pointer = train_file_pointer;

        //opt.zero_grad();
        opt.backward_step(&train_loss);
        //opt.step();
        if epoch % 10 == 0{
            let (xs, ys, train_file_pointer) = dataset::get_batch_val(temp_val_pointer);
            let logits = model.forward(&xs.to_kind(Kind::Int64).to_device(device));
            let (b, t, c) = logits.size3().unwrap();
            let logits = logits.view([b*t, c]);
            let target = &ys.to_kind(Kind::Int64).to_device(device);
            let targets = target.view([b*t]).to_device(device);
            let val_loss = logits.cross_entropy_for_logits(&targets);
            temp_val_pointer = train_file_pointer;
            println!("this is epoch, {:?} ,this is train loss {:?}, this is val loss {:?}",epoch, train_loss, val_loss);
        }



        //losses += f64::try_from(loss)?;
        //println!("{}", losses)
    }
    Ok(())
}

fn find_device() -> Device {
    let device = if tch::utils::has_mps(){
        Device::Mps
    }else if tch::utils::has_cuda() {
        Device::Cuda(0)

    }else {
        Device::Cpu
    };
    println!("device is selected: {:?}", device);
    return device;
}

