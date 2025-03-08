#![deny(warnings)]

use std::path::Path;

use executorch::evalue::IntoEValue;
use executorch::module::Module;
use executorch::tensor::TensorPtr;
use ndarray::array;

fn main() {
    executorch::platform::pal_init();

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join("add.pte");
    let mut module = Module::new(model_path, None, None);

    let tensor1 = TensorPtr::from_array(array![1.0_f32]).unwrap();
    let tensor2 = TensorPtr::from_array(array![1.0_f32]).unwrap();
    let inputs = [tensor1.into_evalue(), tensor2.into_evalue()];

    let outputs = module.forward(&inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    let output = outputs.into_iter().next().unwrap();
    let output = output.as_tensor().into_typed::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![2.0], output.as_array());
}
