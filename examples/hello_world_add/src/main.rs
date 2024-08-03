#![deny(warnings)]

use executorch::data_loader::FileDataLoader;
use executorch::evalue::{EValue, Tag};
use executorch::memory::{HierarchicalAllocator, MallocMemoryAllocator, MemoryManager};
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{Tensor, TensorImpl};
use executorch::util::Span;

use ndarray::array;
use std::vec;

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    executorch::platform::pal_init();

    let mut file_data_loader = FileDataLoader::from_cstr(cstr::cstr!(b"model.pte"), None).unwrap();

    let program = Program::load(
        &mut file_data_loader,
        Some(ProgramVerification::InternalConsistency),
    )
    .unwrap();

    let method_meta = program.method_meta(cstr::cstr!(b"forward")).unwrap();

    let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    let mut planned_buffers = (0..num_memory_planned_buffers)
        .map(|idx| vec![0; method_meta.memory_planned_buffer_size(idx).unwrap()])
        .collect::<Vec<_>>();
    let mut planned_arenas = planned_buffers
        .iter_mut()
        .map(|buffer| Span::from_slice(buffer.as_mut_slice()))
        .collect::<Vec<_>>();

    let mut planned_memory =
        HierarchicalAllocator::new(Span::from_slice(planned_arenas.as_mut_slice()));

    let mut method_allocator = MallocMemoryAllocator::new();
    let memory_manager = MemoryManager::new(&mut method_allocator, Some(&mut planned_memory), None);

    let mut method = program
        .load_method(cstr::cstr!(b"forward"), &memory_manager)
        .unwrap();

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
    let output = &outputs[0];
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor();

    println!("Output tensor computed: {:?}", output);
    assert_eq!(array![2.0_f32], output.as_array());
}
