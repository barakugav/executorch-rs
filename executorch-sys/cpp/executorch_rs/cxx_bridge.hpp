#pragma once

#include "executorch_rs/c_bridge.h"

#include <cstddef>
#include <cstdint>
#include "executorch/runtime/core/exec_aten/exec_aten.h"

#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"

#include "executorch/extension/module/module.h"
#if defined(EXECUTORCH_RS_MODULE)
#include "executorch-sys/src/cxx_bridge/module.rs.h"
#endif

#if defined(EXECUTORCH_RS_TENSOR_PTR)
#include "executorch-sys/src/cxx_bridge/tensor_ptr.rs.h"
#include <vector>
#endif

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_STD)
    std::unique_ptr<executorch::extension::MallocMemoryAllocator> MallocMemoryAllocator_new();
    struct MemoryAllocator *MallocMemoryAllocator_as_memory_allocator(executorch::extension::MallocMemoryAllocator &self);
#endif

#if defined(EXECUTORCH_RS_TENSOR_PTR)
    std::shared_ptr<executorch::aten::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        ScalarType scalar_type,
        TensorShapeDynamism dynamism,
        rust::Box<executorch_rs::cxx_util::RustAny> allocation);
#endif

#if defined(EXECUTORCH_RS_MODULE)
    std::unique_ptr<executorch::extension::Module> Module_new(
        const std::string &file_path,
        rust::Slice<const rust::Str> data_files,
        const ModuleLoadMode load_mode,
        std::unique_ptr<executorch::runtime::EventTracer> event_tracer);

    Error Module_load(executorch::extension::Module &self, ProgramVerification verification);
    bool Module_is_loaded(const executorch::extension::Module &self);
    Error Module_num_methods(executorch::extension::Module &self, size_t *method_num_out);
    Error Module_method_names(executorch::extension::Module &self, rust::Vec<rust::String> *method_names_out);
    Error Module_load_method(executorch::extension::Module &self, const std::string &method_name, HierarchicalAllocator *planned_memory, executorch::runtime::EventTracer *event_tracer);
    bool Module_unload_method(executorch::extension::Module &self, const std::string &method_name);
    bool Module_is_method_loaded(const executorch::extension::Module &self, const std::string &method_name);
    Error Module_method_meta(executorch::extension::Module &self, const std::string &method_name, MethodMeta *method_meta_out);
    Error Module_execute(executorch::extension::Module &self, const std::string &method_name, ArrayRefEValue inputs, VecEValue *outputs);
#endif
}
