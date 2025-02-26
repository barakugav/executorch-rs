#![deny(warnings)]
// #![no_std]
// #![no_main]

use executorch::data_loader::FileDataLoader;
use executorch::evalue::EValue;
use executorch::memory::{HierarchicalAllocator, MemoryAllocator, MemoryManager};
use executorch::ndarray::array;
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{ArrayStorage, Tensor};
use executorch::util::Span;

use libc_print::libc_println;

static mut MEMORY_ALLOCATOR_BUF: [u8; 4096] = [0; 4096];

fn real_main() {
    executorch::platform::pal_init();

    // Safety: We are the main function, no other function access the buffer
    let buffer = unsafe { &mut *core::ptr::addr_of_mut!(MEMORY_ALLOCATOR_BUF) };
    let allocator = MemoryAllocator::new(buffer);

    let file_data_loader = FileDataLoader::from_path_cstr(cstr::cstr!(b"model.pte"), None).unwrap();

    let program = Program::load(
        &file_data_loader,
        Some(ProgramVerification::InternalConsistency),
    )
    .unwrap();

    let method_meta = program.method_meta(cstr::cstr!(b"forward")).unwrap();

    let num_memory_planned_buffers = method_meta.num_memory_planned_buffers();
    let planned_arenas = allocator
        .allocate_arr_fn(num_memory_planned_buffers, |idx| {
            let buf_size = method_meta.memory_planned_buffer_size(idx).unwrap();
            let buf = allocator
                .allocate_arr::<u8>(buf_size)
                .expect("Failed to allocate buffer");
            Span::from_slice(buf)
        })
        .unwrap();

    let mut planned_memory = HierarchicalAllocator::new(planned_arenas);

    let memory_manager = MemoryManager::new(&allocator, Some(&mut planned_memory), None);

    let mut method = program
        .load_method(cstr::cstr!(b"forward"), &memory_manager)
        .unwrap();

    let input_array1 = ArrayStorage::new(array!(1.0_f32));
    let input_tensor_impl1 = input_array1.as_tensor_impl();
    let input_tensor1 =
        Tensor::new_in_storage(&input_tensor_impl1, allocator.allocate_pinned().unwrap());
    // allocate storage for EValue in the allocator
    let storage = allocator.allocate_pinned().unwrap();
    let input_evalue1 = EValue::new_in_storage(input_tensor1, storage);

    let input_array2 = ArrayStorage::new(array!(1.0_f32));
    let input_tensor_impl2 = input_array2.as_tensor_impl();
    let input_tensor2 =
        Tensor::new_in_storage(&input_tensor_impl2, allocator.allocate_pinned().unwrap());
    // allocate storage for EValue on the local stack
    let storage = executorch::storage!(EValue);
    let input_evalue2 = EValue::new_in_storage(input_tensor2, storage);

    let mut method_exe = method.start_execution();

    method_exe.set_input(&input_evalue1, 0).unwrap();
    method_exe.set_input(&input_evalue2, 1).unwrap();

    let outputs = method_exe.execute().unwrap();
    let output = outputs.get(0);
    let output = output.as_tensor().into_typed::<f32>();

    libc_println!("Output tensor computed: {:?}", output);
    assert_eq!(array!(2.0), output.as_array());
}

// FIXME: Unfortunately, no_std is WIP

// #[no_mangle]
// pub fn main(_argc: i32, _argv: *const *const u8) -> u32 {
//     real_main();
//     0
// }

// #[panic_handler]
// fn panic(_info: &core::panic::PanicInfo) -> ! {
//     loop {}
// }

fn main() {
    real_main();
}
