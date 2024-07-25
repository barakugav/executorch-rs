#pragma once

#include "executorch/runtime/core/error.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/core/span.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"

#include <cstdint>

namespace executorch_rs
{

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
    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta *method_meta, size_t index);
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new();
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers);

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
}
