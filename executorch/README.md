# ExecuTorch-rs

[![Crates.io](https://img.shields.io/crates/v/executorch.svg)](https://crates.io/crates/executorch/)
[![Documentation](https://docs.rs/executorch/badge.svg)](https://docs.rs/executorch/)
![License](https://img.shields.io/crates/l/executorch)



Bindings for ExecuTorch - On-device AI across mobile, embedded and edge for PyTorch.

Provides a high-level Rust API for executing PyTorch models on mobile, embedded and edge devices using the
[ExecuTorch library](https://pytorch.org/executorch-overview), specifically the
[C++ API](https://github.com/pytorch/executorch).
PyTorch models are created and exported in Python, and then loaded and executed on-device using the
ExecuTorch library.

The following example create a simple model in Python, exports it, and then executes it in Rust:

Create a model in Python and export it:
```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


model = Add()
exported_program = export(model, (torch.ones(1), torch.ones(1)))
executorch_program = to_edge_transform_and_lower(exported_program).to_executorch()
with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)
```

Execute the model in Rust:
```rust,ignore
use executorch::evalue::{EValue, IntoEValue};
use executorch::module::Module;
use executorch::tensor_ptr;
use ndarray::array;

let mut module = Module::from_file_path("model.pte");

let (tensor1, tensor2) = (tensor_ptr![1.0_f32], tensor_ptr![1.0_f32]);
let inputs = [tensor1.into_evalue(), tensor2.into_evalue()];

let outputs = module.forward(&inputs).unwrap();
let [output]: [EValue; 1] = outputs.try_into().expect("not a single output");
let output = output.as_tensor().into_typed::<f32>();

println!("Output tensor computed: {:?}", output);
assert_eq!(array![2.0], output.as_array());
```

See `example/hello_world` for a complete example.

## Build
To use the library you must compile the C++ executorch library yourself, as there are many configurations that
determines which modules, backends, and operations are supported. See the `executorch-sys` crate for more info.
Currently the supported Cpp executorch version is `1.0.0`.


## Cargo Features
- `data-loader`:
  Includes additional structs in the `data_loader` module for loading data. Without this feature the only
  available data loader is `BufferDataLoader`. The `libextension_data_loader.a` static library is
  required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.
- `module`:
  Includes the `module` API, a high-level API for loading and executing PyTorch models. It is an alternative to
  the lower-level `Program` API, which is more suitable for embedded systems.
  The `libextension_module_static.a` static library is required, compile C++ `executorch` with
  `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
  Also includes the `std`, `data-loader` and `flat-tensor` features.
- `tensor-ptr`:
  Includes the `tensor::TensorPtr` struct, a smart pointer for tensors that manage the lifetime of the tensor
  object alongside the lifetimes of the data buffer and additional metadata. The `extension_tensor.a`
  static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`.
  Also includes the `std` feature.
- `flat-tensor`:
  Includes the `FlatTensorDataMap` struct that can read `.ptd` files with external tensors for models.
  The `libextension_flat_tensor.a` static library is required,
  compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON`.
- `etdump`:
  Includes the `ETDumpGen` struct, an implementation of an `EventTracer`, used for debugging and profiling.
  The `libetdump.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_DEVTOOLS=ON` and
  `EXECUTORCH_ENABLE_EVENT_TRACER=ON`.
  In addition, the `flatcc` (or `flatcc_d`) library is required, available at `{CMAKE_DIR}/third-party/flatcc_ep/lib/`,
  and should be linked by the user.
- `ndarray`:
  Conversions between `executorch` tensors and `ndarray` arrays.
  Adds a dependency to the `ndarray` crate.
  This feature is enabled by default.
- `f16`:
  Adds a dependency to the `half` crate, which provides a fully capable `f16` and `bf16` types.
  Without this feature enabled, both of these types are available with a simple conversions to/from `u16` only.
  Note that this only affect input/output tensors, the internal computations always have the capability to operate on such scalars.
- `num-complex`:
  Adds a dependency to the `num-complex` crate, which provides a fully capable complex number type.
  Without this feature enabled, complex numbers are available as a simple struct with two public fields without any operations.
  Note that this only affect input/output tensors, the internal computations always have the capability to operate on such scalars.
- `std`:
  Enable the standard library. This feature is enabled by default, but can be disabled to build [`executorch`](crate)
  in a `no_std` environment.
  See the `examples/no_std` example.
  Also includes the `alloc` feature.
  NOTE: no_std is still WIP, see <https://github.com/pytorch/executorch/issues/4561>
- `alloc`:
  Enable allocations.
  When this feature is disabled, all methods that require allocations will not be compiled.
  This feature is enabled by the `std` feature, which is enabled by default.
  Its possible to enable this feature without the `std` feature, and the allocations will be done using the
  [`alloc`](https://doc.rust-lang.org/alloc/) crate, that requires a global allocator to be set.

By default the `std` and `ndarray` features are enabled.
