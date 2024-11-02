#![deny(warnings)]

use std::path::PathBuf;

use executorch::evalue::EValue;
use executorch::module::Module;
use executorch::tensor::{Array, Tensor};
use ndarray::array;

fn main() {
    executorch::platform::pal_init();

    let mut module = Module::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model.pte"),
        None,
    );

    let input_array1 = Array::new(array![1.0_f32]);
    let input_tensor1 = input_array1.as_tensor_impl();
    let input_evalue1 = EValue::new(Tensor::new(&input_tensor1));

    let input_array2 = Array::new(array![1.0_f32]);
    let input_tensor2 = input_array2.as_tensor_impl();
    let input_evalue2 = EValue::new(Tensor::new(&input_tensor2));

    let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
    assert_eq!(outputs.len(), 1);
    let output = outputs.into_iter().next().unwrap();
    let output = output.as_tensor().into_typed::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![2.0], output.as_array());
}
