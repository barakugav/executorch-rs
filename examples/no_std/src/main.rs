#![deny(warnings)]
#![no_std]
#![no_main]

use core::ffi::CStr;

use executorch::data_loader::BufferDataLoader;
use executorch::evalue::EValue;
use executorch::memory::{
    BufferMemoryAllocator, HierarchicalAllocator, MemoryAllocatorExt, MemoryManager,
};
use executorch::platform::{LogLevel, PlatformImpl};
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{Tensor, TensorImpl};
use executorch::util::Span;

use libc_print::{libc_eprintln, libc_println};

#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);
const ADD_MODEL_BYTES_ALIGNED: AlignedBytes<{ include_bytes!("../../models/add.pte").len() }> =
    AlignedBytes(*include_bytes!("../../models/add.pte"));
const ADD_MODEL_BYTES: &[u8] = &ADD_MODEL_BYTES_ALIGNED.0;

fn real_main() {
    init_platform();

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
        .load_method(c"forward", &memory_manager, None, None, None)
        .unwrap();

    let (data1, sizes1, dim_order1, strides1) = ([1.0_f32], [1], [0], [1]);
    let input_tensor_impl1 =
        TensorImpl::from_slice(&sizes1, &data1, &dim_order1, &strides1).unwrap();
    let input_tensor1 = Tensor::new_in_allocator(&input_tensor_impl1, &allocator);
    let input_evalue1 = EValue::new_in_allocator(input_tensor1, &allocator);

    let (data2, sizes2, dim_order2, strides2) = ([1.0_f32], [1], [0], [1]);
    let input_tensor_impl2 =
        TensorImpl::from_slice(&sizes2, &data2, &dim_order2, &strides2).unwrap();
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

    assert_eq!(output.sizes(), [1]);
    let output = *output.get(&[0]).unwrap();
    libc_println!("Output tensor computed: {:?}", output);
    assert_eq!(2.0, output);
}

fn init_platform() {
    let mut platform = PlatformImpl::new(Some(c"examples/no_std/src/main.rs"));
    platform.set_emit_log_message(|entry| {
        let level = match entry.level {
            LogLevel::Debug => "D",
            LogLevel::Info => "I",
            LogLevel::Error => "E",
            LogLevel::Fatal => "F",
            LogLevel::Unknown => "?",
        };
        libc_eprintln!(
            "{level} executorch:{}:{} {}",
            entry.filename.unwrap_or("?"),
            entry.line,
            entry.message
        );
    });
    platform.set_abort(|| unsafe { libc::abort() });
    // Safety: we register the platform and init it once, before any other executorch operations
    unsafe {
        assert!(executorch::platform::register_platform_impl(platform));
        executorch::platform::pal_init();
    }
}

#[unsafe(no_mangle)]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const core::ffi::c_char) -> core::ffi::c_int {
    real_main();
    0
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    libc_eprintln!("{info}");
    unsafe { libc::abort() }
}

/// Called by the C++ runtime functions defined in `cpp_runtime_stubs.cpp` on a fatal error.
#[unsafe(no_mangle)]
extern "C" fn cpp_runtime_abort(msg: *const core::ffi::c_char) -> ! {
    let msg = unsafe { CStr::from_ptr(msg) }.to_str().unwrap_or("?");
    panic!("C++ runtime error: {msg}");
}

/// The precompiled `core` library of hosted targets (such as `x86_64-unknown-linux-gnu`) is compiled with
/// `panic=unwind`, and references the unwinding personality function, even though this binary is compiled with
/// `panic=abort` and never unwinds. Bare-metal targets (such as `thumbv7em-none-eabihf`) don't need this.
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
