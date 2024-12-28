# ExecuTorch-rs

`executorch` is a Rust library for executing PyTorch models in Rust.
It is a Rust wrapper around the [ExecuTorch C++ API](https://pytorch.org/executorch).
It depends on version `0.4.0` of the Cpp API, but will advance as the API does.
The underlying C++ library is still in alpha, and its API is subject to change together with the Rust API.

## Usage
Create a model in `Python` and export it:
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
use executorch::tensor::{ArrayStorage, Tensor};
use ndarray::array;

let mut module = Module::new("model.pte", None);

let input_array1 = ArrayStorage::new(array![1.0_f32]);
let input_tensor1 = input_array1.as_tensor_impl();
let input_evalue1 = Tensor::new(&input_tensor1).into_evalue();

let input_array2 = ArrayStorage::new(array![1.0_f32]);
let input_tensor2 = input_array2.as_tensor_impl();
let input_evalue2 = Tensor::new(&input_tensor2).into_evalue();

let outputs = module.forward(&[input_evalue1, input_evalue2]).unwrap();
assert_eq!(outputs.len(), 1);
let output = outputs.into_iter().next().unwrap();
let output = output.as_tensor().into_typed::<f32>();

println!("Output tensor computed: {:?}", output);
assert_eq!(array![2.0], output.as_array());
```
See `example/hello_world_add` and `example/hello_world_add_no_std` for the complete examples.

## Build
To build the library, you need to build the C++ library first.
The C++ library allow for great flexibility with many flags, customizing which modules, kernels, and extensions are built.
Multiple static libraries are built, and the Rust library links to them.
In the following example we build the C++ library with the necessary flags to run example `hello_world_add`:
```bash
# Clone the C++ library
cd ${TEMP_DIR}
git clone --depth 1 --branch v0.4.0 https://github.com/pytorch/executorch.git
cd executorch
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
    -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    ..
make -j

# Static libraries are in cmake-out/
# core:
#   cmake-out/libexecutorch.a
#   cmake-out/libexecutorch_no_prim_ops.a
# kernels implementations:
#   cmake-out/kernels/portable/libportable_ops_lib.a
#   cmake-out/kernels/portable/libportable_kernels.a
# extension data loader, enabled with EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON:
#   cmake-out/extension/data_loader/libextension_data_loader.a
# extension module, enabled with EXECUTORCH_BUILD_EXTENSION_MODULE=ON:
#   cmake-out/extension/module/libextension_module_static.a

# Run example
# We set EXECUTORCH_RS_EXECUTORCH_LIB_DIR to the path of the C++ build output
cd ${EXECUTORCH_RS_DIR}/examples/hello_world_add
python export_model.py
EXECUTORCH_RS_EXECUTORCH_LIB_DIR=${TEMP_DIR}/executorch/cmake-out cargo run
```

The `executorch` crate will always look for the following static libraries:
- `libexecutorch.a`
- `libexecutorch_no_prim_ops.a`

Additional libs are required if feature flags are enabled (see next section):
- `libextension_data_loader.a`
- `libextension_module_static.a`

The static libraries of the kernels implementations are required only if your model uses them, and they should be **linked manually** by the binary that uses the `executorch` crate.
For example, the `hello_world_add` example uses a model with a single addition operation, so it compile the C++ library with `DEXECUTORCH_SELECT_OPS_LIST=aten::add.out` and contain the following lines in its `build.rs`:
```rust
println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").unwrap();
println!("cargo::rustc-link-search={}/kernels/portable/", libs_dir);
```
Note that the ops and kernels libs are linked with `+whole-archive` to ensure that all symbols are included in the binary.

The build (and library) is tested on Ubuntu and MacOS, not on Windows.

## Cargo Features
- `data-loader`

    Includes the `FileDataLoader` and `MmapDataLoader` structs. Without this feature the only available data loader is `BufferDataLoader`. The `libextension_data_loader.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`.

- `module`

    Includes the `Module` struct. The `libextension_module_static.a` static library is required, compile C++ `executorch` with `EXECUTORCH_BUILD_EXTENSION_MODULE=ON`.
    Also includes the `std` feature.

- `f16`

    Support for half precision floating point numbers using the `half` crate. Models that require input or output tensors with `f16` data type can be operated on with this features.

- `complex`

    Support for complex numbers using the `num-complex` crate. Models that require input or output tensors with complex `32` or `64` bit floating point numbers can be operated on with this feature. If in addition the `f16` feature is enabled, complex numbers with half precision can be used.

- `std`

    Enable the standard library. This feature is enabled by default, but can be disabled to build `executorch` in a `no_std` environment.
    See the `hello_world_add_no_std` example.
    Also includes the `alloc` feature.
    NOTE: no_std is still WIP, see https://github.com/pytorch/executorch/issues/4561

- `alloc`

    Enable allocations.
    When this feature is disabled, all methods that require allocations will not be compiled.
    This feature is enabled by the `std` feature, which is enabled by default.
    Its possible to enable this feature without the `std` feature, and the allocations will be done using the `alloc` crate, that requires a global allocator to be set.

By default the `std` feature is enabled.
