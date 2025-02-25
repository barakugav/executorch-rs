#![deny(warnings)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

//! Bindings for ExecuTorch - On-device AI across mobile, embedded and edge for PyTorch.
//!
//! Provides a high-level Rust API for executing PyTorch models on mobile, embedded and edge devices using the
//! [ExecuTorch library](https://pytorch.org/executorch-overview), specifically the
//! [C++ API](https://github.com/pytorch/executorch).
//! PyTorch models are created and exported in Python, and then loaded and executed on-device using the
//! ExecuTorch library.
//!
//! The following example create a simple model in Python, exports it, and then executes it in Rust:
//!
//! Create a model in `Python` and export it:
//! ```ignore
//! import torch
//! from executorch.exir import to_edge
//! from torch.export import export
//!
//! class Add(torch.nn.Module):
//!     def __init__(self):
//!         super(Add, self).__init__()
//!
//!     def forward(self, x: torch.Tensor, y: torch.Tensor):
//!         return x + y
//!
//!
//! aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))
//! edge_program = to_edge(aten_dialect)
//! executorch_program = edge_program.to_executorch()
//! with open("model.pte", "wb") as file:
//!     file.write(executorch_program.buffer)
//! ```
//!
//! Execute the model in Rust:
//! ```rust,ignore
//! use executorch::evalue::IntoEValue;
//! use executorch::module::Module;
//! use executorch::tensor::TensorPtr;
//! use ndarray::array;
//!
//! let mut module = Module::new("model.pte", None);
//!
//! let tensor1 = TensorPtr::from_array(array![1.0_f32]);
//! let tensor2 = TensorPtr::from_array(array![1.0_f32]);
//! let inputs = [tensor1.into_evalue(), tensor2.into_evalue()];
//!
//! let outputs = module.forward(&[inputs]).unwrap();
//! assert_eq!(outputs.len(), 1);
//! let output = outputs.into_iter().next().unwrap();
//! let output = output.as_tensor().into_typed::<f32>();
//!
//! println!("Output tensor computed: {:?}", output);
//! assert_eq!(array![2.0], output.as_array());
//! ```
//!
//! ## Cargo Features
//! - `data-loader`:
//!     Includes additional structs in the [`data_loader`] module for loading data. Without this feature the only
//!     available data loader is `BufferDataLoader. `The `libextension_data_loader.a` static library is
//!     required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
//! - `module`:
//!     Includes the [`module`] API, a high-level API for loading and executing PyTorch models. It is an alternative to
//!     the lower-level [`Program`](crate::program::Program) API, which is more suitable for embedded systems.
//!     The `libextension_module_static.a` static library is required, compile C++ `executorch` with
//!     `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`. Also includes the `std` feature.
//! - `tensor-ptr`:
//!     Includes the [`tensor::TensorPtr`] struct, a smart pointer for tensors that manage the lifetime of the tensor
//!     object alongside the lifetimes of the data buffer and additional metadata. The `extension_tensor.a`
//!     static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`.
//!     Also includes the `std` feature.
//! - `ndarray`:
//!     Conversions between `executorch` tensors and `ndarray` arrays.
//!     Adds a dependency to the `ndarray` crate.
//!     This feature is enabled by default.
//! - `f16`:
//!     Support for half precision floating point numbers using the [`half`](https://docs.rs/half/latest/half/) crate.
//!     Models that require input or output tensors with `f16` data type can be operated on with this features.
//! - `complex`:
//!     Support for complex numbers using the [`num-complex`](https://docs.rs/num/latest/num/complex/struct.Complex.html)
//!     crate. Models that require input or output tensors with complex `32` or `64` bit floating point numbers can be
//!     operated on with this feature. If the `f16` feature is enabled as well, complex numbers with half
//!     precision can be used.
//! - `std`:
//!     Enable the standard library. This feature is enabled by default, but can be disabled to build [`executorch`](crate)
//!     in a `no_std` environment.
//!     See the `examples/no_std` example.
//!     Also includes the `alloc` feature.
//!     NOTE: no_std is still WIP, see <https://github.com/pytorch/executorch/issues/4561>
//! - `alloc`:
//!     Enable allocations.
//!     When this feature is disabled, all methods that require allocations will not be compiled.
//!     This feature is enabled by the `std` feature, which is enabled by default.
//!     Its possible to enable this feature without the `std` feature, and the allocations will be done using the
//!     [`alloc`](https://doc.rust-lang.org/alloc/) crate, that requires a global allocator to be set.
//!
//! By default the `std` and `ndarray` features are enabled.
//!
//! ## Build
//! To use the library you must compile the C++ executorch library yourself, as there are many configurations that
//! determines which modules, backends, and operations are supported. See the `executorch-sys` crate for more info.
//!
//! ## Embedded Systems
//! The library is designed to be used both in `std` and `no_std` environments. The `no_std` environment is useful for
//! embedded systems, where the standard library is not available. The `alloc` feature can be used to provide an
//! alternative to the standard library's allocator, but it is possible to use the library without allocations at all.
//! Due to some difference between Cpp and Rust, it is not trivial to provide such API, and the interface may feel
//! more verbose. See the `memory::Storage` struct for stack allocations of Cpp objects, and the `examples/no_std`
//! example.
//!
//! ## API Stability
//! The C++ API is still in Beta, and this Rust lib will continue to change with it. Currently the supported
//! executorch version is `0.5.0`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

#[doc(hidden)]
pub mod __private {
    #[cfg(feature = "std")]
    pub mod alloc {
        pub use std::boxed::Box;
        pub use std::vec::Vec;
    }
    #[cfg(not(feature = "std"))]
    pub mod alloc {
        extern crate alloc;
        pub use alloc::boxed::Box;
        pub use alloc::vec::Vec;
    }
}

#[allow(unused_imports)]
use crate::__private::alloc;

use executorch_sys::executorch as et_c;
use executorch_sys::executorch_rs as et_rs_c;

#[macro_use]
mod private;
pub mod data_loader;
mod error;
pub mod evalue;
pub mod memory;
#[cfg(feature = "module")]
pub mod module;
pub mod platform;
pub mod program;
pub mod tensor;
pub mod util;

pub(crate) use error::Result;
pub use error::{CError, Error};

#[cfg(feature = "ndarray")]
pub use ndarray;

#[cfg(feature = "f16")]
pub use half;

#[cfg(feature = "complex")]
pub use num_complex;
