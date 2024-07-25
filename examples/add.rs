use executorch::{
    evalue::{EValue, Tag},
    tensor::{Tensor, TensorImpl},
    FileDataLoader, HierarchicalAllocator, MallocMemoryAllocator, MemoryManager, Program,
    ProgramVerification, Span,
};
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

    let mut method_exe = method.start_execution();

    method_exe.set_input(&input_evalue1, 0).unwrap();
    method_exe.set_input(&input_evalue2, 1).unwrap();

    let outputs = method_exe.execute().unwrap();
    let output = outputs.get_output(0);
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor();

    println!("Output tensor computed: {:?}", output.as_array::<f32>());
}
