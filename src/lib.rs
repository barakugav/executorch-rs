#![deny(warnings)]
#![deny(missing_docs)]

//! Bindings for ExecuTorch - On-device AI across mobile, embedded and edge for PyTorch.
//!
//! Provides a high-level Rust API for executing PyTorch models on mobile, embedded and edge devices using the
//! [ExecuTorch library](https://pytorch.org/executorch-overview), specifically the [C++ API](https://github.com/pytorch/executorch).
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
//! ```no_run
//! use executorch::evalue::{EValue, Tag};
//! use executorch::module::Module;
//! use executorch::tensor::{Array, Tensor};
//! use ndarray::array;
//!
//! let mut module = Module::new("model.pte", None);
//!
//! let input_array1 = Array::new(array![1.0_f32]);
//! let input_tensor1 = input_array1.to_tensor_impl();
//! let input_evalue1 = EValue::from_tensor(Tensor::new(&input_tensor1));
//!
//! let input_array2 = Array::new(array![1.0_f32]);
//! let input_tensor2 = input_array2.to_tensor_impl();
//! let input_evalue2 = EValue::from_tensor(Tensor::new(&input_tensor2));
//!
//! let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
//! assert_eq!(outputs.len(), 1);
//! let output = outputs.into_iter().next().unwrap();
//! assert_eq!(output.tag(), Some(Tag::Tensor));
//! let output = output.as_tensor();
//!
//! println!("Output tensor computed: {:?}", output);
//! assert_eq!(array![2.0_f32], output.as_array());
//! ```
//!
//! The library have a few features that can be enabled or disabled:
//! to the lower-level `Program` API, which is mort suitable for embedded systems.
//! - `data-loader`:
//!     include the [`data_loader`] module for loading data. The `libextension_data_loader.a` static library is
//!     required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
//! - `module`:
//!     include the [`module`] API, a high-level API for loading and executing PyTorch models. It is an alternative to
//!     the lower-level `Program` API, which is mort suitable for embedded systems. The `libextension_module_static.a`
//!     static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
//!     Also includes the `std` feature.
//! - `f16`:
//!     Support for half precision floating point numbers using the `half` crate. Models that require input or output
//!     tensors with `f16` data type can be operated on with this features.
//! - `complex`:
//!     Support for complex numbers using the `num-complex` crate. Models that require input or output tensors with
//! complex `32` or `64` bit floating point numbers can be operated on with this feature. If in addition the `f16`
//! feature is enabled, complex numbers with half precision can be used.
//! - `std`:
//!     Enable the standard library. This feature is enabled by default, but can be disabled to build `executorch`
//!     in a `no_std` environment.
//!     See the `hello_world_add_no_std` example.
//!     Also includes the `alloc` feature.
//! - `alloc`:
//!     Enable allocations.
//!     When this feature is disabled, all methods that require allocations will not be compiled.
//!     This feature is enabled by the `std` feature, which is enabled by default.
//!     Its possible to enable this feature without the `std` feature, and the allocations will be done using the
//!     `alloc` crate, that requires a global allocator to be set.
//!
//! By default the `std` feature is enabled.
//!
//!
//! The C++ API is still in Alpha, and this Rust lib will continue to change with it. Currently the supported
//! executorch version is `0.3.0`.
//!
//! To use the library you must compile the C++ executorch library yourself, as there are many configurations that
//! determines which modules, backends, and operations are supported. See the `executorch-sys` crate for more info.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate core as std;

cfg_if::cfg_if! { if #[cfg(feature = "std")] {
    use std as et_alloc;
} else if #[cfg(feature = "alloc")] {
    extern crate alloc;
    use alloc as et_alloc;
} }

use executorch_sys::executorch_rs as et_rs_c;
use executorch_sys::torch::executor as et_c;

#[macro_use]
mod private;
pub mod data_loader;
pub mod error;
pub mod evalue;
pub mod memory;
#[cfg(feature = "module")]
pub mod module;
pub mod platform;
pub mod program;
pub mod tensor;
pub mod util;
