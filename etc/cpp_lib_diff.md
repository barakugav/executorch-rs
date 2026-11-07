```bash
gid v1.4.0 v1.5.1 --name-only -- \
    ':!.*' \
    ':!backends/*' \
    ':!extension/android/*' \
    ':!extension/apple/*' \
    ':!extension/benchmark/*' \
    ':!extension/pybindings/*' \
    ':!extension/llm/*' \
    ':!kernels/*' \
    ':!examples/*' \
    ':!scripts/*' \
    ':!codegen/*' \
    ':!test/*' \
    ':!*/test/*' \
    ':!*_test.*' \
    ':!*/testing_util/*' \
    ':!docs/*' \
    ':!website/*' \
    ':!zephyr/*' \
    ':!*.py' \
    ':!*.pyi' \
    ':!*.bzl' \
    ':!*BUCK' \
    ':!*TARGETS' \
    ':!*.json' \
    ':!*.swift' \
    ':!*.swift' \
    ':!CLAUDE.md' \
    ':!CODEOWNERS' \
    ':!CONTRIBUTING.md' \
    ':!README-wheel.md' \
    ':!README.md' \
    > diff_files.txt


gid v1.4.0 v1.5.1 -- \
    ':!.*' \
    ':!backends/*' \
    ':!extension/android/*' \
    ':!extension/apple/*' \
    ':!extension/benchmark/*' \
    ':!extension/pybindings/*' \
    ':!extension/llm/*' \
    ':!kernels/*' \
    ':!examples/*' \
    ':!scripts/*' \
    ':!codegen/*' \
    ':!test/*' \
    ':!*/test/*' \
    ':!*_test.*' \
    ':!*/testing_util/*' \
    ':!docs/*' \
    ':!website/*' \
    ':!zephyr/*' \
    ':!*.py' \
    ':!*.pyi' \
    ':!*.bzl' \
    ':!*BUCK' \
    ':!*TARGETS' \
    ':!*.json' \
    ':!*.swift' \
    ':!*.swift' \
    ':!CLAUDE.md' \
    ':!CODEOWNERS' \
    ':!CONTRIBUTING.md' \
    ':!README-wheel.md' \
    ':!README.md'

gid v1.3.1 v1.4.0 -- \
    "*CMakeLists.txt" \
    "runtime/*.h" \
    "extension/*.h" \
    ':!extension/apple/*' \
    ':!extension/wasm/*' \
    ':!extension/android/*' \
    ':!extension/kernel_util/*' \
    ':!extension/llm/*' \
    "devtools/*.h" \
    ':!backends/*' \
    ':!examples/*' \
    ':!*/test/*' \
    ':!*/testing_util/*' \
```


EXECUTORCH_BUILD_CUDA
EXECUTORCH_BUILD_EXTENSION_IMAGE
device arg in make_tensor_ptr
clone_tensor_ptr_to
HierarchicalAllocator constructor with `Span<const etensor::Device> planned_buffer_devices`
HierarchicalAllocator::planned_buffer_devices
HierarchicalAllocator sized increased, `Span<const etensor::Device> planned_buffer_devices_;`
MemoryAllocator::used_size
MemoryAllocator::free_size
optional deprecated, use std::optional instead
Tensor::device instead of Tensor::unsafe_storage().device()
MemoryManager::planned_buffer_devices
MemoryManager::has_device_memory
