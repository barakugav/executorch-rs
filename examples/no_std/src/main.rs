#![deny(warnings)]
// #![no_std]
// #![no_main]

// Unfortunately, no_std is still WIP, see https://github.com/pytorch/executorch/issues/4561
extern crate link_cplusplus;

use executorch::data_loader::BufferDataLoader;
use executorch::evalue::EValue;
use executorch::memory::{BufferMemoryAllocator, HierarchicalAllocator, MemoryManager};
use executorch::ndarray::array;
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{ArrayStorage, Tensor};
use executorch::util::Span;

use libc_print::libc_println;

#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);
const ADD_MODEL_BYTES_ALIGNED: AlignedBytes<{ include_bytes!("../../models/add.pte").len() }> =
    AlignedBytes(*include_bytes!("../../models/add.pte"));
const ADD_MODEL_BYTES: &[u8] = &ADD_MODEL_BYTES_ALIGNED.0;

fn real_main() {
    executorch::platform::pal_init();

    let mut buffer = [0_u8; 4096];
    let allocator = BufferMemoryAllocator::new(&mut buffer);

    let data_loader = BufferDataLoader::new(ADD_MODEL_BYTES);
    let program =
        Program::load(&data_loader, Some(ProgramVerification::InternalConsistency)).unwrap();

    let method_meta = program.method_meta(c"forward").unwrap();

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
        .load_method(c"forward", &memory_manager, None)
        .unwrap();

    let input_array1 = ArrayStorage::new(array!(1.0_f32)).unwrap();
    let input_tensor_impl1 = input_array1.as_tensor_impl();
    let input_tensor1 = Tensor::new_in_allocator(&input_tensor_impl1, &allocator);
    let input_evalue1 = EValue::new_in_allocator(input_tensor1, &allocator);

    let input_array2 = ArrayStorage::new(array!(1.0_f32)).unwrap();
    let input_tensor_impl2 = input_array2.as_tensor_impl();
    let input_tensor2 = Tensor::new_in_allocator(&input_tensor_impl2, &allocator);
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
