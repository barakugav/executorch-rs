#![deny(warnings)]

use executorch::{
    EValue, FileDataLoader, HierarchicalAllocator, MallocMemoryAllocator, MemoryManager, Program,
    ProgramVerification, Span, Tag, Tensor, TensorImpl,
};
use ndarray::array;
use std::vec;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    executorch::pal_init();

    let mut file_data_loader = FileDataLoader::new("model.pte").unwrap();

    let program = Program::load(
        &mut file_data_loader,
        ProgramVerification::InternalConsistency,
    )
    .unwrap();

    let method_meta = program.method_meta("forward").unwrap();

    let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    let mut planned_buffers = (0..num_memory_planned_buffers)
        .map(|idx| vec![0; method_meta.memory_planned_buffer_size(idx).unwrap()])
        .collect::<Vec<_>>();
    let mut planned_arenas = planned_buffers
        .iter_mut()
        .map(|buffer| Span::new(buffer.as_mut_slice()))
        .collect::<Vec<_>>();

    let mut planned_memory = HierarchicalAllocator::new(Span::new(planned_arenas.as_mut_slice()));

    let mut method_allocator = MallocMemoryAllocator::new();
    let mut memory_manager = MemoryManager::new(&mut method_allocator, &mut planned_memory);

    let mut method = program.load_method("forward", &mut memory_manager).unwrap();

    let data1 = array![1.0_f32];
    let input_tensor1 = TensorImpl::from_array(data1.view());
    let input_evalue1 = EValue::from_tensor(Tensor::new(input_tensor1.as_ref()));

    let data2 = array![1.0_f32];
    let input_tensor2 = TensorImpl::from_array(data2.view());
    let input_evalue2 = EValue::from_tensor(Tensor::new(input_tensor2.as_ref()));

    let mut method_exe = method.start_execution();

    method_exe.set_input(&input_evalue1, 0).unwrap();
    method_exe.set_input(&input_evalue2, 1).unwrap();

    let outputs = method_exe.execute().unwrap();
    let output = outputs.get_output(0);
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor().as_array::<f32>();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(output, array![2.0].into_dyn());
}
