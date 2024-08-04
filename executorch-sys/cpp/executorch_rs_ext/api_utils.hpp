#pragma once

#include "executorch/runtime/core/error.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/core/span.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"

#if defined(EXECUTORCH_RS_EXTENSION_DATA_LOADER)
#include "executorch/extension/data_loader/buffer_data_loader.h"
#endif

#if defined(EXECUTORCH_RS_EXTENSION_MODULE)
#include "executorch/extension/module/module.h"
#endif

#include <cstdint>

namespace executorch_rs
{
    template <typename T>
    struct RawVec
    {
        T *data;
        size_t len;
        size_t cap;
    };

    struct Result_i64
    {

        union
        {
            int64_t value_;                // Used if hasValue_ is true.
            torch::executor::Error error_; // Used if hasValue_ is false.
        };

        /// True if the Result contains a value.
        const bool hasValue_;
    };
    struct Result_MethodMeta
    {

        union
        {
            torch::executor::MethodMeta value_; // Used if hasValue_ is true.
            torch::executor::Error error_;      // Used if hasValue_ is false.
        };

        /// True if the Result contains a value.
        const bool hasValue_;
    };

    Result_MethodMeta Program_method_meta(const torch::executor::Program *program, const char *method_name);
    void Program_destructor(torch::executor::Program *program);
    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta *method_meta, size_t index);
    torch::executor::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *MemoryAllocator_allocate(torch::executor::MemoryAllocator *allocator, size_t size, size_t alignment);
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new();
    void MallocMemoryAllocator_destructor(torch::executor::util::MallocMemoryAllocator *allocator);
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers);
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator *allocator);

    // Tensor
    size_t Tensor_nbytes(const exec_aten::Tensor *tensor);
    ssize_t Tensor_size(const exec_aten::Tensor *tensor, ssize_t dim);
    ssize_t Tensor_dim(const exec_aten::Tensor *tensor);
    ssize_t Tensor_numel(const exec_aten::Tensor *tensor);
    exec_aten::ScalarType Tensor_scalar_type(const exec_aten::Tensor *tensor);
    ssize_t Tensor_element_size(const exec_aten::Tensor *tensor);
    const exec_aten::ArrayRef<exec_aten::SizesType> Tensor_sizes(const exec_aten::Tensor *tensor);
    const exec_aten::ArrayRef<exec_aten::DimOrderType> Tensor_dim_order(const exec_aten::Tensor *tensor);
    const exec_aten::ArrayRef<exec_aten::StridesType> Tensor_strides(const exec_aten::Tensor *tensor);
    const void *Tensor_const_data_ptr(const exec_aten::Tensor *tensor);
    void *Tensor_mutable_data_ptr(const exec_aten::Tensor *tensor);
    void Tensor_destructor(exec_aten::Tensor *tensor);

    void EValue_destructor(torch::executor::EValue *evalue);
    const exec_aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const torch::executor::BoxedEvalueList<int64_t> *list);
    const exec_aten::ArrayRef<exec_aten::Tensor> BoxedEvalueList_Tensor_get(const torch::executor::BoxedEvalueList<exec_aten::Tensor> *list);

#if defined(EXECUTORCH_RS_EXTENSION_DATA_LOADER)
    torch::executor::util::BufferDataLoader BufferDataLoader_new(const void *data, size_t size);
#endif

#if defined(EXECUTORCH_RS_EXTENSION_MODULE)
    torch::executor::Module Module_new(torch::executor::ArrayRef<char> file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer);
    void Module_destructor(torch::executor::Module *module);
    torch::executor::Result<RawVec<RawVec<char>>> Module_method_names(torch::executor::Module *module);
    torch::executor::Error Module_load_method(torch::executor::Module *module, torch::executor::ArrayRef<char> method_name);
    bool Module_is_method_loaded(const torch::executor::Module *module, torch::executor::ArrayRef<char> method_name);
    Result_MethodMeta Module_method_meta(const torch::executor::Module *module, torch::executor::ArrayRef<char> method_name);
    torch::executor::Result<RawVec<torch::executor::EValue>> Module_execute(torch::executor::Module *module, torch::executor::ArrayRef<char> method_name, torch::executor::ArrayRef<torch::executor::EValue> inputs);
#endif

}
