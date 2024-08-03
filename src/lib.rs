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
//! ```
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
//! ```rust
//! rustuse executorch::evalue::{EValue, Tag};
//! use executorch::module::Module;
//! use executorch::tensor::{Tensor, TensorImpl};
//! use ndarray::array;
//!
//! let mut module = Module::new("model.pte", None);
//!
//! let data1 = array![1.0_f32];
//! let input_tensor1 = TensorImpl::from_array(data1.view());
//! let input_evalue1 = EValue::from_tensor(Tensor::new(input_tensor1.as_ref()));
//!
//! let data2 = array![1.0_f32];
//! let input_tensor2 = TensorImpl::from_array(data2.view());
//! let input_evalue2 = EValue::from_tensor(Tensor::new(input_tensor2.as_ref()));
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
//! The library have a few features that can be enabled or disabled, by default all are disabled:
//! - `module`: Enable the [`module`] API, a high-level API for loading and executing PyTorch models. It is an alternative
//! to the lower-level `Program` API, which is mort suitable for embedded systems.
//! - `data_loader`: Enable the [`data_loader`] module for loading data.
//!
//! The C++ API is still in Alpha, and this Rust lib will continue to change with it. Currently the supported
//! executorch version is `0.2.1`.
//!
//! To use the library you must compile the C++ executorch library yourself, as there are many configurations that
//! determines which modules, backends, and operations are supported. See the `executorch-sys` crate for more info.

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
