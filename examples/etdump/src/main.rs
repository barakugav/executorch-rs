#![deny(warnings)]

use std::path::Path;

use executorch::evalue::IntoEValue;
use executorch::event_tracer::ETDumpGen;
use executorch::module::Module;
use executorch::tensor_ptr;
use ndarray::array;

fn main() {
    executorch::platform::pal_init();

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join("add.pte");
    let mut module = Module::from_file_path(model_path);

    let mut etdump_gen = ETDumpGen::new();
    module
        .load_method("forward", None, Some(etdump_gen.as_mut()))
        .unwrap();

    let (tensor1, tensor2) = (tensor_ptr![1.0_f32], tensor_ptr![1.0_f32]);
    let inputs = [tensor1.into_evalue(), tensor2.into_evalue()];

    {
        let outputs = module.forward(&inputs).unwrap();
        let [output] = outputs.try_into().expect("not a single output");
        let output = output.as_tensor().into_typed::<f32>();

        println!("Output tensor computed: {:?}", output);
        assert_eq!(array![2.0], output.as_array());
    }

    println!(
        "Got an ETDump data buffer of {} bytes!",
        etdump_gen.get_etdump_data().unwrap().len()
    );
}
