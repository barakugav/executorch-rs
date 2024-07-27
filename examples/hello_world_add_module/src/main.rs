use executorch::{EValue, Module, Tag, Tensor, TensorImpl};
use ndarray::array;
use std::vec;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    executorch::pal_init();

    let mut module = Module::new("model.pte");

    let mut data1 = vec![1.0_f32; 1];
    let sizes1 = [1];
    let data_order1 = [0];
    let strides1 = [1];
    let mut input_tensor1 = TensorImpl::new(&sizes1, &mut data1, &data_order1, &strides1);
    let input_evalue1 = EValue::from_tensor(Tensor::new(&mut input_tensor1));

    let mut data2 = vec![1.0_f32; 1];
    let sizes2 = [1];
    let data_order2 = [0];
    let strides2 = [1];
    let mut input_tensor2 = TensorImpl::new(&sizes2, &mut data2, &data_order2, &strides2);
    let input_evalue2 = EValue::from_tensor(Tensor::new(&mut input_tensor2));

    let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
    assert_eq!(outputs.len(), 1);
    let output = outputs.into_iter().next().unwrap();
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor().as_array::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(output, array![2.0].into_dyn());
}
