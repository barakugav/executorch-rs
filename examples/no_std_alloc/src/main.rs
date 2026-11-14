#![deny(warnings)]
#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::ffi::CStr;
use core::mem::MaybeUninit;

use executorch::data_loader::BufferDataLoader;
use executorch::evalue::IntoEValue;
use executorch::memory::{BufferMemoryAllocator, HierarchicalAllocator, MemoryManager};
use executorch::ndarray::array;
use executorch::platform::{LogLevel, PlatformImpl};
use executorch::program::{Program, ProgramVerification};
use executorch::tensor::{ArrayStorage, Tensor};
use executorch::util::Span;

use libc_print::{libc_eprintln, libc_println};
use linked_list_allocator::LockedHeap;

/// The global allocator, used by the `alloc` crate (`Box`, `Vec`, ...) and by the `executorch` crate when its
/// `alloc` feature is enabled. It manages the static `HEAP` buffer, and does not depend on `malloc` or an OS.
#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

const HEAP_SIZE: usize = 64 * 1024;
static mut HEAP: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

#[repr(align(16))]
struct AlignedBytes<const N: usize>([u8; N]);
const ADD_MODEL_BYTES_ALIGNED: AlignedBytes<{ include_bytes!("../../models/add.pte").len() }> =
    AlignedBytes(*include_bytes!("../../models/add.pte"));
const ADD_MODEL_BYTES: &[u8] = &ADD_MODEL_BYTES_ALIGNED.0;

fn real_main() {
    init_heap();
    init_platform();

    let data_loader = BufferDataLoader::new(ADD_MODEL_BYTES);
    let program =
        Program::load(&data_loader, Some(ProgramVerification::InternalConsistency)).unwrap();
    let method_meta = program.method_meta(c"forward").unwrap();

    // All the memory used by the method is allocated on the heap
    let mut method_buffer = vec![0_u8; 4096];
    let method_allocator = BufferMemoryAllocator::new(&mut method_buffer);
    let mut planned_buffers: Vec<Vec<u8>> = (0..method_meta.num_memory_planned_buffers())
        .map(|idx| vec![0_u8; method_meta.memory_planned_buffer_size(idx).unwrap()])
        .collect();
    let mut planned_spans: Vec<Span<u8>> = planned_buffers
        .iter_mut()
        .map(|buf| Span::from_slice(buf))
        .collect();
    let mut planned_memory = HierarchicalAllocator::new(&mut planned_spans);
    let memory_manager = MemoryManager::new(&method_allocator, Some(&mut planned_memory), None);

    let mut method = program
        .load_method(c"forward", &memory_manager, None, None, None)
        .unwrap();

    // With the `alloc` feature, the C++ tensor and EValue objects are allocated on the heap
    let input1 = ArrayStorage::new(array![1.0_f32]).unwrap();
    let input2 = ArrayStorage::new(array![1.0_f32]).unwrap();
    let (input1, input2) = (input1.as_tensor_impl(), input2.as_tensor_impl());
    let inputs = [
        Tensor::new(&input1).into_evalue(),
        Tensor::new(&input2).into_evalue(),
    ];

    let mut method_exe = method.start_execution();
    for (idx, input) in inputs.iter().enumerate() {
        method_exe.set_input(input, idx).unwrap();
    }
    let outputs = method_exe.execute().unwrap();
    let output = outputs.get(0);
    let output = output.as_tensor().into_typed::<f32>();

    libc_println!("Output tensor computed: {:?}", output);
    assert_eq!(array![2.0], output.as_array());

    let heap = ALLOCATOR.lock();
    libc_println!("Heap usage: {} / {} bytes", heap.used(), heap.size());
}

fn init_heap() {
    // Safety: called once, before any allocation, and `HEAP` is not accessed anywhere else
    unsafe {
        ALLOCATOR
            .lock()
            .init((&raw mut HEAP).cast::<u8>(), HEAP_SIZE);
    }
}

fn init_platform() {
    let mut platform = PlatformImpl::new(Some(c"examples/no_std_alloc/src/main.rs"));
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
extern "C" fn main(
    _argc: core::ffi::c_int,
    _argv: *const *const core::ffi::c_char,
) -> core::ffi::c_int {
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
