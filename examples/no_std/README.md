# `no_std` Example

This example executes a model in a `no_std` environment:
- No Rust standard library (`#![no_std]`, `#![no_main]`) and no global allocator (the `alloc` crate is not used).
- No C++ standard library (libstdc++, libc++), only the C standard library is linked, used for printing and as the
  program entry point.

The example runs on a regular host (Linux or macOS), but the same setup applies to bare-metal targets.

## C++ Library Configuration
The C++ ExecuTorch library should be compiled without dependencies on the C++ standard library:
```bash
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_PAL_DEFAULT=minimal \
    -DEXECUTORCH_OPTIMIZE_SIZE=ON \
    -DEXECUTORCH_SELECT_OPS_LIST=aten::add.out \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    ..
```
- `EXECUTORCH_PAL_DEFAULT=minimal`: The default platform abstraction layer (PAL) uses `std::chrono` and
  `stderr`. The `minimal` PAL doesn't assume anything about the platform, and its functions can be overridden at
  runtime using `executorch::platform::register_platform_impl`, as done in this example.
- `EXECUTORCH_OPTIMIZE_SIZE=ON`: In a `Release` build, compiles with `-fno-exceptions -fno-rtti`, which require
  runtime support from the C++ standard library.
- `EXECUTORCH_SELECT_OPS_LIST=aten::add.out`: Generates the `executorch_selected_kernels` library that registers
  only the `add` operator. The `build.rs` of this example links it as a whole archive, and links `portable_kernels`
  as a regular archive, so only the `add` kernel is linked. Some kernels (for example `rand` and `randn`) use the C++
  standard library and can't be linked without it.

When the `std` feature of the `executorch` crate is disabled, its C++ bridge is also compiled with
`-fno-exceptions -fno-rtti` and no C++ standard library is linked. The few C++ runtime functions that are still
referenced (`operator delete`, `__cxa_pure_virtual`, ...) are defined in `src/cpp_runtime_stubs.cpp`, which is
compiled by `build.rs`. They are written in C++ rather than Rust so that the compiler chooses their symbols, which
differ between platforms. Alternatively, a C++ runtime library can be linked, for example using the `link-cplusplus`
crate.

## Run
The `.cargo/config.toml` of this example sets `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` to the
`etc/.dev-env/executorch/cmake-out-no-std` directory, which is created by `etc/setup_dev_env.py --build-no-std`.
To use a different build of the C++ library, set the `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` environment variable.
```bash
cargo run
```
