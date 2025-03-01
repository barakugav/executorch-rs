#![deny(warnings)]
#![deny(missing_docs)]

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
//! git clone --depth 1 --branch v0.5.0 https://github.com/pytorch/executorch.git .
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
//!     -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
//!     -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
//!     -DEXECUTORCH_ENABLE_LOGGING=ON \
//!     ..
//! make -j
//!
//! # Static libraries are in cmake-out/
//! # core:
//! #   cmake-out/libexecutorch.a
//! #   cmake-out/libexecutorch_core.a
//! # kernels implementations:
//! #   cmake-out/kernels/portable/libportable_ops_lib.a
//! #   cmake-out/kernels/portable/libportable_kernels.a
//! # extension data loader, enabled with EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON:
//! #   cmake-out/extension/data_loader/libextension_data_loader.a
//! # extension module, enabled with EXECUTORCH_BUILD_EXTENSION_MODULE=ON:
//! #   cmake-out/extension/module/libextension_module_static.a
//! # extension tensor, enabled with EXECUTORCH_BUILD_EXTENSION_TENSOR=ON:
//! #   cmake-out/extension/tensor/libextension_tensor.a
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
//! Additional libs are required if feature flags are enabled (see next section):
//! - `libextension_data_loader.a`
//! - `libextension_module_static.a`
//! - `libextension_tensor.a`
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
//! If its not provided, its the resposibility of the binary to add the libs directories to the linker search path, and
//! the crate will just link to the static libraries using `cargo::rustc-link-lib=...`.
//!
//! If you want to link to executorch libs yourself, set the environment variable `EXECUTORCH_RS_LINK` to `0`, and
//! the crate will not link to any library and not modify the linker search path.
//!
//! ## Cargo Features
//! By default all features are disabled.
//! - `data-loader`: Includes the [`FileDataLoader`] and [`MmapDataLoader`] structs. Without this feature the only available
//!     data loader is [`BufferDataLoader`]. The `libextension_data_loader.a` static library is required, compile C++
//!     `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
//! - `module`: Includes the [`Module`] struct. The `libextension_module_static.a` static library is required, compile C++
//!     `executorch` with `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
//! - `tensor-ptr`:
//!     Includes a few functions creating `cxx::SharedPtr<Tensor>` pointers, that manage the lifetime of the tensor
//!     object alongside the lifetimes of the data buffer and additional metadata. The `extension_tensor.a`
//!     static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`.
//!     Also includes the `std` feature.
//! - `std`:
//!     Enable the standard library. This feature is enabled by default, but can be disabled to build `executorch` in a `no_std` environment.
//!     NOTE: no_std is still WIP, see <https://github.com/pytorch/executorch/issues/4561>
//!
//! [`FileDataLoader`]: crate::executorch::extension::FileDataLoader
//! [`MmapDataLoader`]: crate::executorch::extension::MmapDataLoader
//! [`BufferDataLoader`]: crate::executorch::extension::BufferDataLoader
//! [`Module`]: crate::executorch::extension::Module

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

#[cfg(feature = "std")]
extern crate link_cplusplus;

mod c_link {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(rustdoc::invalid_html_tags)]
    #![allow(rustdoc::broken_intra_doc_links)]
    #![allow(missing_docs)]
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
pub use c_link::*;

#[cfg(any(feature = "tensor-ptr", feature = "module"))]
mod cxx_bridge;

/// Bindings to C/C++ wrapper functions and structs written by this crate around the Cpp `executorch` library.
pub mod cpp {

    /// Cpp bindings to the `Module` extension.
    #[cfg(feature = "module")]
    pub mod module {
        pub use crate::cxx_bridge::module::ffi::*;
    }

    /// Cpp bindings to the `TensorPtr` extension.
    #[cfg(feature = "tensor-ptr")]
    pub mod tensor_ptr {
        pub use crate::cxx_bridge::tensor_ptr::ffi::*;
    }

    #[cfg(feature = "tensor-ptr")]
    pub use super::cxx_bridge::tensor_ptr::cxx_util as util;
}

impl Copy for crate::Tag {}

#[cfg(any(feature = "tensor-ptr", feature = "module"))]
pub use cxx;
