#![deny(warnings)]

use std::path::Path;

use executorch::evalue::{EValue, IntoEValue};
use executorch::module::Module;
use executorch::tensor::{RawTensor, RawTensorImpl};
use ndarray::array;

fn main() {
    executorch::platform::pal_init();

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join("add.pte");
    let mut module = Module::from_file_path(model_path);

    let data1 = [1.0_f32];
    let sizes1 = [1];
    let dim_order1 = [0];
    let strides1 = [1];
    let input_tensor1 = unsafe {
        RawTensorImpl::from_ptr(&sizes1, data1.as_ptr().cast_mut(), &dim_order1, &strides1).unwrap()
    };
    let input_evalue1 = unsafe { RawTensor::new(&input_tensor1) }.into_evalue();

    let sizes2 = [1];
    let data2 = [1.0_f32];
    let dim_order2 = [0];
    let strides2 = [1];
    let input_tensor2 = unsafe {
        RawTensorImpl::from_ptr(&sizes2, data2.as_ptr().cast_mut(), &dim_order2, &strides2).unwrap()
    };
    let input_evalue2 = unsafe { RawTensor::new(&input_tensor2) }.into_evalue();

    let inputs = [input_evalue1, input_evalue2];

    let outputs = module.forward(&inputs).unwrap();
    let [output]: [EValue; 1] = outputs.try_into().expect("not a single output");
    let output = output.as_tensor().into_typed::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![2.0], output.as_array());
}
