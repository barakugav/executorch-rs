#pragma once

#include "executorch/runtime/core/error.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"

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

}