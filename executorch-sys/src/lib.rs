//! Unsafe bindings for ExecuTorch - On-device AI across mobile, embedded and edge for PyTorch.
//!
//! Provides a low level Rust bindings for the ExecuTorch library.
//! For the common use case, it is recommended to use the high-level API provided by the `executorch` crate, where
//! a more detailed documentation can be found.
//!
//!
//! To build the library, you need to build the C++ library first.
//! The C++ library allow for great flexibility with many flags, customizing which modules, kernels, and extensions are built.
//! Multiple static libraries are built, and the Rust library links to them.
//! In the following example we build the C++ library with the necessary flags to run example `hello_world_add`:
//! ```bash
//! # Clone the C++ library
//! cd ${TEMP_DIR}
//! git clone --depth 1 --branch v0.2.1 https://github.com/pytorch/executorch.git
//! cd executorch
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
//!     -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
//!     -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
//!     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
//!     -DEXECUTORCH_ENABLE_LOGGING=ON \
//!     ..
//! make -j
//!
//! # Static libraries are in cmake-out/
//! # core:
//! #   cmake-out/libexecutorch.a
//! #   cmake-out/libexecutorch_no_prim_ops.a
//! # kernels implementations:
//! #   cmake-out/kernels/portable/libportable_ops_lib.a
//! #   cmake-out/kernels/portable/libportable_kernels.a
//! # extension data loader, enabled with EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON:
//! #   cmake-out/extension/data_loader/libextension_data_loader.a
//! # extension module, enabled with EXECUTORCH_BUILD_EXTENSION_MODULE=ON:
//! #   cmake-out/extension/module/libextension_module_static.a
//!
//! # Run example
//! # We set EXECUTORCH_RS_EXECUTORCH_LIB_DIR to the path of the C++ build output
//! cd ${EXECUTORCH_RS_DIR}/examples/hello_world_add
//! python export_model.py
//! EXECUTORCH_RS_EXECUTORCH_LIB_DIR=${TEMP_DIR}/executorch/cmake-out cargo run
//! ```
//!
//! The `executorch` crate will always look for the following static libraries:
//! - `libexecutorch.a`
//! - `libexecutorch_no_prim_ops.a`
//!
//! Additional libs are required if feature flags are enabled (see next section):
//! - `libextension_data_loader.a`
//! - `libextension_module_static.a`
//!
//! The static libraries of the kernels implementations are required only if your model uses them, and they should be **linked manually** by the binary that uses the `executorch` crate.
//! For example, the `hello_world_add` example uses a model with a single addition operation, so it compile the C++ library with `DEXECUTORCH_SELECT_OPS_LIST=aten::add.out` and contain the following lines in its `build.rs`:
//! ```rust
//! println!("cargo::rustc-link-lib=static=portable_kernels");
//! println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");
//!
//! let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").unwrap();
//! println!("cargo::rustc-link-search={}/kernels/portable/", libs_dir);
//! ```
//! Note that the `portable_ops_lib` is linked with `+whole-archive` to ensure that all symbols are included in the binary.
//!
//! ## Cargo Features
//! By default all features are disabled.
//! - `data-loader`: include the `FileDataLoader` struct. The `libextension_data_loader.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
//! - `module`: include the `Module` struct. The `libextension_module_static.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

mod c_link {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
pub use c_link::root::*;

use crate::executorch_rs as et_rs_c;
use crate::torch::executor as et_c;

impl Drop for et_c::Tensor {
    fn drop(&mut self) {
        unsafe { et_rs_c::Tensor_destructor(self) }
    }
}
