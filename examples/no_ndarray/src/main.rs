#![deny(warnings)]

use std::path::Path;

use executorch::evalue::IntoEValue;
use executorch::module::Module;
use executorch::tensor::{Tensor, TensorImpl};

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
    let input_tensor1 = TensorImpl::from_slice(&sizes1, &data1, &dim_order1, &strides1).unwrap();
    let input_evalue1 = Tensor::new(&input_tensor1).into_evalue();

    let sizes2 = [1];
    let data2 = [1.0_f32];
    let dim_order2 = [0];
    let strides2 = [1];
    let input_tensor2 = TensorImpl::from_slice(&sizes2, &data2, &dim_order2, &strides2).unwrap();
    let input_evalue2 = Tensor::new(&input_tensor2).into_evalue();

    let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
    let [output] = outputs
        .try_into()
        .unwrap_or_else(|_| panic!("not a single output"));
    let output = output.as_tensor().into_typed::<f32>();

    assert_eq!(output.dim(), 1);
    assert_eq!(output.sizes()[0], 1);
    let output = unsafe { *output.as_data_ptr() };

    println!("Output tensor computed: {:?}", output);
    assert_eq!(2.0, output);
}
