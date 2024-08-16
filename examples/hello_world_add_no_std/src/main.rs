#![deny(warnings)]
#![no_std]

use executorch::data_loader::FileDataLoader;
use executorch::evalue::{EValue, Tag};
use executorch::memory::{HierarchicalAllocator, MemoryAllocator, MemoryManager};
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{Array, Tensor};
use executorch::util::Span;

static mut MEMORY_ALLOCATOR_BUF: [u8; 4096] = [0; 4096];

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Debug)
        .init();

    executorch::platform::pal_init();

    // Safety: We are the main function, no other function access the buffer
    let buffer = unsafe { &mut *core::ptr::addr_of_mut!(MEMORY_ALLOCATOR_BUF) };
    let memory_allocator = MemoryAllocator::new(buffer);

    let file_data_loader = FileDataLoader::from_path_cstr(cstr::cstr!(b"model.pte"), None).unwrap();

    let program = Program::load(
        &file_data_loader,
        Some(ProgramVerification::InternalConsistency),
    )
    .unwrap();

    let method_meta = program.method_meta(cstr::cstr!(b"forward")).unwrap();

    let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    let planned_arenas = memory_allocator
        .allocate_arr_fn(num_memory_planned_buffers, |idx| {
            let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
            let buf = memory_allocator
                .allocate_arr::<u8>(buf_size)
                .expect("Failed to allocate buffer");
            Span::from_slice(buf)
        })
        .unwrap();

    let mut planned_memory = HierarchicalAllocator::new(Span::from_slice(planned_arenas));

    let memory_manager = MemoryManager::new(&memory_allocator, Some(&mut planned_memory), None);

    let mut method = program
        .load_method(cstr::cstr!(b"forward"), &memory_manager)
        .unwrap();

    let input_array1 = Array::new(ndarray::arr1(&[1.0_f32]));
    let input_tensor1 = input_array1.to_tensor_impl();
    let input_evalue1 = EValue::from_tensor(Tensor::new(&input_tensor1));

    let input_array2 = Array::new(ndarray::arr1(&[1.0_f32]));
    let input_tensor2 = input_array2.to_tensor_impl();
    let input_evalue2 = EValue::from_tensor(Tensor::new(&input_tensor2));

    let mut method_exe = method.start_execution();

    method_exe.set_input(&input_evalue1, 0).unwrap();
    method_exe.set_input(&input_evalue2, 1).unwrap();

    let outputs = method_exe.execute().unwrap();
    let output = &outputs[0];
    assert_eq!(output.tag(), Some(Tag::Tensor));
    let output = output.as_tensor();

    log::info!("Output tensor computed: {:?}", output);
    assert_eq!(ndarray::arr1(&[2.0_f32]), output.as_array());
}
