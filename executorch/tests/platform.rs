use core::ffi::CStr;
use std::ptr::NonNull;

static mut INIT_CALLS_COUNT: u32 = 0;
static mut ABORT_CALLS_COUNT: u32 = 0;
static mut CURRENT_TICKS_CALLS_COUNT: u32 = 0;
static mut TICKS_TO_NS_MULTIPLIER_CALLS_COUNT: u32 = 0;
static mut EMIT_LOG_MESSAGE_CALLS_COUNT: u32 = 0;
static mut ALLOCATE_CALLS_COUNT: u32 = 0;
static mut FREE_CALLS_COUNT: u32 = 0;

#[test]
fn custom_pal() {
    let source_filename =
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!(file!(), "\0").as_bytes()) };
    let mut plat_impl = executorch::platform::PlatformImpl::new(Some(source_filename));
    assert_eq!(unsafe { INIT_CALLS_COUNT }, 0);
    plat_impl.set_init(|| {
        unsafe { INIT_CALLS_COUNT += 1 };
    });
    plat_impl.set_abort(|| {
        unsafe { ABORT_CALLS_COUNT += 1 };
    });
    plat_impl.set_current_ticks(|| {
        unsafe { CURRENT_TICKS_CALLS_COUNT += 1 };
        executorch::platform::Timestamp::new(0)
    });
    plat_impl.set_ticks_to_ns_multiplier(|| {
        unsafe { TICKS_TO_NS_MULTIPLIER_CALLS_COUNT += 1 };
        executorch::platform::TickRatio::new(1, 1)
    });
    plat_impl.set_emit_log_message(|log_entry| {
        unsafe { EMIT_LOG_MESSAGE_CALLS_COUNT += 1 };
        println!(
            "[custom PAL][{:?}] {}::{}::{} ({:?}) {}",
            log_entry.timestamp,
            log_entry.filename.unwrap_or("?"),
            log_entry.function.unwrap_or("?"),
            log_entry.line,
            log_entry.level,
            log_entry.message
        )
    });
    plat_impl.set_allocate(|size| {
        unsafe { ALLOCATE_CALLS_COUNT += 1 };
        let ptr = unsafe { libc::malloc(size) };
        Some(NonNull::new(ptr).unwrap())
    });
    plat_impl.set_free(|ptr| {
        unsafe { FREE_CALLS_COUNT += 1 };
        unsafe { libc::free(ptr) };
    });

    assert_eq!(unsafe { INIT_CALLS_COUNT }, 0);
    unsafe { executorch::platform::register_platform_impl(plat_impl) };
    assert_eq!(unsafe { INIT_CALLS_COUNT }, 1); // registration should can init
    unsafe { executorch::platform::pal_init() };
    assert_eq!(unsafe { INIT_CALLS_COUNT }, 2);

    assert_eq!(unsafe { EMIT_LOG_MESSAGE_CALLS_COUNT }, 0);
    // Cause a error log message to be emitted by loading a program from an empty data loader
    let data_loader = executorch::data_loader::BufferDataLoader::new(&[]);
    let _ = executorch::program::Program::load(
        &data_loader,
        Some(executorch::program::ProgramVerification::InternalConsistency),
    );
    assert_eq!(unsafe { EMIT_LOG_MESSAGE_CALLS_COUNT }, 1);
}
