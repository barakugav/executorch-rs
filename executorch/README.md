# ExecuTorch-rs

[![Crates.io](https://img.shields.io/crates/v/executorch.svg)](https://crates.io/crates/executorch/)
[![Documentation](https://docs.rs/executorch/badge.svg)](https://docs.rs/executorch/)
![License](https://img.shields.io/crates/l/executorch)


`executorch` is a Rust library for executing PyTorch models in Rust.
It is a Rust wrapper around the [ExecuTorch C++ API](https://pytorch.org/executorch).
It depends on version `0.5.0` of the Cpp API, but will advance as the API does.
The underlying C++ library is still in Beta, and its API is subject to change together with the Rust API.

## Usage
Create a model in Python and export it:
```python
import torch
from executorch.exir import to_edge
from torch.export import export

class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + y


aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))
edge_program = to_edge(aten_dialect)
executorch_program = edge_program.to_executorch()
with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)
```
Execute the model in Rust:
```rust
use executorch::evalue::IntoEValue;
use executorch::module::Module;
use executorch::tensor_ptr;
use ndarray::array;

let mut module = Module::new("model.pte", None);

let (tensor1, tensor2) = (tensor_ptr![1.0_f32], tensor_ptr![1.0_f32]);
let inputs = [tensor1.into_evalue(), tensor2.into_evalue()];

let outputs = module.forward(&inputs).unwrap();
assert_eq!(outputs.len(), 1);
let output = outputs.into_iter().next().unwrap();
let output = output.as_tensor().into_typed::<f32>();

println!("Output tensor computed: {:?}", output);
assert_eq!(array![2.0], output.as_array());
```
See `example/hello_world` for a complete example.

## Build
To build the library, you need to build the C++ library first.
The C++ library allow for great flexibility with many flags, customizing which modules, kernels, and extensions are built.
Multiple static libraries are built, and the Rust library links to them.
In the following example we build the C++ library with the necessary flags to run example `hello_world`:
```bash
# Clone the C++ library
cd ${EXECUTORCH_CPP_DIR}
git clone --depth 1 --branch v0.5.0 https://github.com/pytorch/executorch.git .
git submodule sync --recursive
git submodule update --init --recursive

# Install requirements
./install_requirements.sh

# Build C++ library
mkdir cmake-out && cd cmake-out
cmake \
    -DDEXECUTORCH_SELECT_OPS_LIST=aten::add.out \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
    -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    ..
make -j

# Static libraries are in cmake-out/
# core:
#   cmake-out/libexecutorch.a
#   cmake-out/libexecutorch_core.a
# kernels implementations:
#   cmake-out/kernels/portable/libportable_ops_lib.a
#   cmake-out/kernels/portable/libportable_kernels.a
# extension data loader, enabled with EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON:
#   cmake-out/extension/data_loader/libextension_data_loader.a
# extension module, enabled with EXECUTORCH_BUILD_EXTENSION_MODULE=ON:
#   cmake-out/extension/module/libextension_module_static.a
# extension tensor, enabled with EXECUTORCH_BUILD_EXTENSION_TENSOR=ON:
#   cmake-out/extension/tensor/libextension_tensor.a
# extension tensor, enabled with EXECUTORCH_BUILD_DEVTOOLS=ON:
#   cmake-out/devtools/libetdump.a

# Run example
# We set EXECUTORCH_RS_EXECUTORCH_LIB_DIR to the path of the C++ build output
cd ${EXECUTORCH_RS_DIR}/examples/hello_world
python export_model.py
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=${EXECUTORCH_CPP_DIR}/cmake-out cargo run
```

The `executorch` crate will always look for the following static libraries:
- `libexecutorch.a`
- `libexecutorch_core.a`

Additional libs are required if feature flags are enabled (see next section):
- `libextension_data_loader.a`
- `libextension_module_static.a`
- `libextension_tensor.a`
- `libetdump.a`

The static libraries of the kernels implementations are required only if your model uses them, and they should be **linked manually** by the binary that uses the `executorch` crate.
For example, the `hello_world` example uses a model with a single addition operation, so it compile the C++ library with `DEXECUTORCH_SELECT_OPS_LIST=aten::add.out` and contain the following lines in its `build.rs`:
```rust
println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").unwrap();
println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
```
Note that the ops and kernels libs are linked with `+whole-archive` to ensure that all symbols are included in the binary.

The build (and library) is tested on Ubuntu and MacOS, not on Windows.

## Cargo Features
- `data-loader`

    Includes the `FileDataLoader` and `MmapDataLoader` structs. Without this feature the only available data loader is `BufferDataLoader`. The `libextension_data_loader.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.

- `module`

    Includes the `Module` struct. The `libextension_module_static.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
    Also includes the `std` feature.

- `tensor-ptr`

    Includes the `TensorPtr` struct, a smart pointer for tensors that manage the lifetime of the tensor
    object alongside the lifetimes of the data buffer and additional metadata. The `libextension_tensor.a`
    static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`.
    Also includes the `std` feature.

- `etdump`

    Includes the `ETDumpGen` struct, an implementation of an `EventTracer`, used for debugging and profiling.
    The `libetdump.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_DEVTOOLS=ON` and
    `EXECUTORCH_ENABLE_EVENT_TRACER=ON`.
    In addition, the `flatcc` (or `flatcc_d`) library is required, available at `{CPP_EXECUTORCH_DIR}/third-party/flatcc/lib/`,
    and should be linked by the user.

- `ndarray`

    Conversions between `executorch` tensors and `ndarray` arrays.
    Adds a dependency to the `ndarray` crate.
    This feature is enabled by default.

- `half`

    Adds a dependency to the `half` crate, which provides a fully capable `f16` and `bf16` types.
    Without this feature enabled, both of these types are available with a simple conversions to/from `u16` only.
    Note that this only affect input/output tensors, the internal computations always have the capability to operate on such scalars.

- `num-complex`

    Adds a dependency to the `num-complex` crate, which provides a fully capable complex number type.
    Without this feature enabled, complex numbers are available as a simple struct with two public fields without any operations.
    Note that this only affect input/output tensors, the internal computations always have the capability to operate on such scalars.

- `std`

    Enable the standard library. This feature is enabled by default, but can be disabled to build `executorch` in a `no_std` environment.
    See the `examples/no_std` example.
    Also includes the `alloc` feature.
    NOTE: no_std is still WIP, see https://github.com/pytorch/executorch/issues/4561

- `alloc`

    Enable allocations.
    When this feature is disabled, all methods that require allocations will not be compiled.
    This feature is enabled by the `std` feature, which is enabled by default.
    Its possible to enable this feature without the `std` feature, and the allocations will be done using the `alloc` crate, that requires a global allocator to be set.

By default the `std` and `ndarray` features are enabled.
