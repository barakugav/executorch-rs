#![cfg_attr(deny_warnings, deny(warnings))]
// some new clippy::lint annotations are supported in latest Rust but not recognized by older versions
#![cfg_attr(deny_warnings, allow(unknown_lints))]
#![cfg_attr(deny_warnings, deny(missing_docs))]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Unsafe bindings for ExecuTorch - On-device AI across mobile, embedded and edge for PyTorch.
//!
//! Provides a low level Rust bindings for the ExecuTorch library.
//! For the common use case, it is recommended to use the high-level API provided by the `executorch` crate, where
//! a more detailed documentation can be found.
//!
//!
//! To build the library, you need to build the C++ library first.
//! The C++ library allow for great flexibility with many flags, customizing which modules, kernels, and extensions are
//! built.
//! Multiple static libraries are built, and the Rust library links to them.
//! In the following example we build the C++ library with the necessary flags to run example `hello_world`:
//! ```bash
//! # Clone the C++ library
//! cd ${EXECUTORCH_CPP_DIR}
//! git clone --depth 1 --branch v1.0.0 https://github.com/pytorch/executorch.git .
//! git submodule sync --recursive
//! git submodule update --init --recursive
//!
//! # Install requirements
//! ./install_requirements.sh
//!
//! # Build C++ library
//! mkdir cmake-out && cd cmake-out
//! cmake \
//!     -DDEXECUTORCH_SELECT_OPS_LIST=aten::add.out \
//!     -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
//!     -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
//!     -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
//!     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
//!     -DEXECUTORCH_ENABLE_LOGGING=ON \
//!     ..
//! make -j
//!
//! # Run example
//! # We set EXECUTORCH_RS_EXECUTORCH_LIB_DIR to the path of the C++ build output
//! cd ${EXECUTORCH_RS_DIR}/examples/hello_world
//! python export_model.py
//! EXECUTORCH_RS_EXECUTORCH_LIB_DIR=${EXECUTORCH_CPP_DIR}/cmake-out cargo run
//! ```
//!
//! The `executorch` crate will always look for the following static libraries:
//! - `libexecutorch.a`
//! - `libexecutorch_core.a`
//!
//! Additional libs are required if feature flags are enabled.
//! For example the `libextension_data_loader.a` is required if the `data-loader` feature is enabled,
//! and `libextension_tensor.a` is required if the `tensor-ptr` feature is enabled.
//! See the feature flags section for more info.
//!
//! The static libraries of the kernels implementations are required only if your model uses them, and they should be
//! **linked manually** by the binary that uses the `executorch` crate.
//! For example, the `hello_world` example uses a model with a single addition operation, so it compile the C++
//! library with `DEXECUTORCH_SELECT_OPS_LIST=aten::add.out` and contain the following lines in its `build.rs`:
//! ```rust
//! println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
//! println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");
//!
//! let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").unwrap();
//! println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
//! ```
//! Note that the ops and kernels libs are linked with `+whole-archive` to ensure that all symbols are included in the
//! binary.
//!
//! The `EXECUTORCH_RS_EXECUTORCH_LIB_DIR` environment variable should be set to the path of the C++ build output.
//! If its not provided, its the responsibility of the binary to add the libs directories to the linker search path, and
//! the crate will just link to the static libraries using `cargo::rustc-link-lib=...`.
//!
//! If you want to link to executorch libs yourself, set the environment variable `EXECUTORCH_RS_LINK` to `0`, and
//! the crate will not link to any library and not modify the linker search path.
//!
//! The crate contains a small C/C++ bridge that uses the headers of the C++ library,
//! and it is compiled using the `cc` crate (and the `cxx` crate, that uses `cc` under the hood).
//! If custom compiler flags (for example `-DET_MIN_LOG_LEVEL=Debug`) are used when compiling the C++ library,
//! you should set the matching environment variables that `cc` reads during `cargo build`
//! (for example `CFLAGS=-DET_MIN_LOG_LEVEL=Debug CXXFLAGS=-DET_MIN_LOG_LEVEL=Debug`),
//! see the [cc docs](https://docs.rs/cc/latest/cc/).
//!
//!
//! ## Cargo Features
//! By default the `std` feature is enabled.
//! - `data-loader`:
//!   Includes the [`FileDataLoader`] and [`MmapDataLoader`] structs. Without this feature the only available
//!   data loader is [`BufferDataLoader`]. The `libextension_data_loader.a` static library is required, compile C++
//!   `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
//! - `module`:
//!   Includes the `Module` struct, a high-level API for loading and executing PyTorch models. It is an alternative to
//!   the lower-level `Program` API, which is more suitable for embedded systems.
//!   The `libextension_module_static.a` static library is required, compile C++ `executorch` with
//!   `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
//!   Also includes the `std`, `data-loader` and `flat-tensor` features.
//! - `tensor-ptr`:
//!   Includes a few functions creating `cxx::SharedPtr<Tensor>` pointers, that manage the lifetime of the tensor
//!   object alongside the lifetimes of the data buffer and additional metadata. The `libextension_tensor.a`
//!   static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`.
//!   Also includes the `std` feature.
//! - `flat-tensor`:
//!   Includes the `FlatTensorDataMap` struct that can read `.ptd` files with external tensors for models.
//!   The `libextension_flat_tensor.a` static library is required,
//!   compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON`.
//! - `etdump`:
//!   Includes the `ETDumpGen` struct, an implementation of an `EventTracer`, used for debugging and profiling.
//!   The `libetdump.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_DEVTOOLS=ON` and
//!   `EXECUTORCH_ENABLE_EVENT_TRACER=ON`.
//!   In addition, the `flatcc` (or `flatcc_d`) library is required, available at `{CMAKE_DIR}/third-party/flatcc_ep/lib/`,
//!   and should be linked by the user.
//! - `std`:
//!   Enable the standard library. This feature is enabled by default, but can be disabled to build `executorch` in a `no_std` environment.
//!   NOTE: no_std is still WIP, see <https://github.com/pytorch/executorch/issues/4561>
//!
//! [`FileDataLoader`]: crate::FileDataLoader
//! [`MmapDataLoader`]: crate::MmapDataLoader
//! [`BufferDataLoader`]: crate::BufferDataLoader
//! [`Module`]: crate::cpp::Module

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(all(feature = "std", link_cxx))]
extern crate link_cplusplus;

/// The version of the ExecuTorch C++ library that this crate is compatible and linked with.
pub const EXECUTORCH_CPP_VERSION: &str = "1.0.0";

mod c_bridge;
pub use c_bridge::*;

#[cfg(feature = "std")]
mod cxx_bridge;

/// Bindings generated by the `cxx` crate.
#[cfg(feature = "std")]
pub mod cpp {
    pub use crate::cxx_bridge::core::ffi::*;

    #[cfg(feature = "module")]
    pub use crate::cxx_bridge::module::ffi::*;

    #[cfg(feature = "tensor-ptr")]
    pub use crate::cxx_bridge::tensor_ptr::ffi::*;

    #[cfg(feature = "tensor-ptr")]
    pub use super::cxx_bridge::tensor_ptr::cxx_util as util;
}

// Re-export cxx
#[cfg(feature = "std")]
pub use cxx;
