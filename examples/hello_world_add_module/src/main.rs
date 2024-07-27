#![deny(warnings)]

use executorch::{EValue, Module, Tag, Tensor, TensorImpl};
use ndarray::array;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    executorch::pal_init();

    let mut module = Module::new("model.pte");

    let data1 = array![1.0_f32];
    let input_tensor1 = TensorImpl::from_array(data1.view());
    let input_evalue1 = EValue::from_tensor(Tensor::new(input_tensor1.as_ref()));

    let data2 = array![1.0_f32];
    let input_tensor2 = TensorImpl::from_array(data2.view());
    let input_evalue2 = EValue::from_tensor(Tensor::new(input_tensor2.as_ref()));

    let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
    assert_eq!(outputs.len(), 1);
    let output = outputs.into_iter().next().unwrap();
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor().as_array::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(output, array![2.0].into_dyn());
}
