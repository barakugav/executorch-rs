#pragma once

#include "executorch_rs_defines.h"

#if defined(EXECUTORCH_RS_MODULE) && !defined(EXECUTORCH_RS_STD)
#error "EXECUTORCH_RS_MODULE requires EXECUTORCH_RS_STD"
#endif

#include <cstddef>
#include <cstdint>
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/core/span.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"

#include "executorch/extension/data_loader/buffer_data_loader.h"

#if defined(EXECUTORCH_RS_MODULE)
#include "executorch/extension/module/module.h"
#endif

#if defined(EXECUTORCH_RS_STD)
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"
#endif

#include <cstdint>

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_STD)
    template <typename T>
    struct Vec
    {
        T *data;
        size_t len;
        size_t cap;
    };

#define VEC_DESTRUCTOR_DEC(T, name) \
    void Vec_##name##_destructor(Vec<T> *vec);

    VEC_DESTRUCTOR_DEC(char, char)
    VEC_DESTRUCTOR_DEC(Vec<char>, Vec_char)
    VEC_DESTRUCTOR_DEC(torch::executor::EValue, EValue)

#endif

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

    Result_MethodMeta Program_method_meta(const torch::executor::Program *self, const char *method_name);
    void Program_destructor(torch::executor::Program *self);
    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta *self, size_t index);
    torch::executor::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *MemoryAllocator_allocate(torch::executor::MemoryAllocator *self, size_t size, size_t alignment);
#if defined(EXECUTORCH_RS_STD)
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new();
    void MallocMemoryAllocator_destructor(torch::executor::util::MallocMemoryAllocator *self);
#endif
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers);
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator *self);

    // Tensor
    void Tensor_new(exec_aten::Tensor *self, exec_aten::TensorImpl *tensor_impl);
    size_t Tensor_nbytes(const exec_aten::Tensor *self);
    ssize_t Tensor_size(const exec_aten::Tensor *self, ssize_t dim);
    ssize_t Tensor_dim(const exec_aten::Tensor *self);
    ssize_t Tensor_numel(const exec_aten::Tensor *self);
    exec_aten::ScalarType Tensor_scalar_type(const exec_aten::Tensor *self);
    ssize_t Tensor_element_size(const exec_aten::Tensor *self);
    exec_aten::ArrayRef<exec_aten::SizesType> Tensor_sizes(const exec_aten::Tensor *self);
    exec_aten::ArrayRef<exec_aten::DimOrderType> Tensor_dim_order(const exec_aten::Tensor *self);
    exec_aten::ArrayRef<exec_aten::StridesType> Tensor_strides(const exec_aten::Tensor *self);
    const void *Tensor_const_data_ptr(const exec_aten::Tensor *self);
    void *Tensor_mutable_data_ptr(const exec_aten::Tensor *self);
    void Tensor_destructor(exec_aten::Tensor *self);

    // torch::executor::EValue EValue_shallow_clone(torch::executor::EValue *evalue);
    void EValue_new_from_i64(torch::executor::EValue *self, int64_t value);
    void EValue_new_from_f64(torch::executor::EValue *self, double value);
    void EValue_new_from_f64_arr(torch::executor::EValue *self, exec_aten::ArrayRef<double> value);
    void EValue_new_from_bool(torch::executor::EValue *self, bool value);
    void EValue_new_from_bool_arr(torch::executor::EValue *self, exec_aten::ArrayRef<bool> value);
    void EValue_new_from_chars(torch::executor::EValue *self, exec_aten::ArrayRef<char> value);
    void EValue_new_from_tensor(torch::executor::EValue *self, const exec_aten::Tensor *value);
    void EValue_copy(const torch::executor::EValue *src, torch::executor::EValue *dst);
    void EValue_destructor(torch::executor::EValue *self);
    void EValue_move(torch::executor::EValue *src, torch::executor::EValue *dst);
    // exec_aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const torch::executor::BoxedEvalueList<int64_t> *self);
    // exec_aten::ArrayRef<exec_aten::Tensor> BoxedEvalueList_Tensor_get(const torch::executor::BoxedEvalueList<exec_aten::Tensor> *self);

    torch::executor::util::BufferDataLoader BufferDataLoader_new(const void *data, size_t size);

#if defined(EXECUTORCH_RS_MODULE)
    void Module_new(torch::executor::Module *self, torch::executor::ArrayRef<char> file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer);
    void Module_destructor(torch::executor::Module *self);
    torch::executor::Result<Vec<Vec<char>>> Module_method_names(torch::executor::Module *self);
    torch::executor::Error Module_load_method(torch::executor::Module *self, torch::executor::ArrayRef<char> method_name);
    bool Module_is_method_loaded(const torch::executor::Module *self, torch::executor::ArrayRef<char> method_name);
    Result_MethodMeta Module_method_meta(torch::executor::Module *self, torch::executor::ArrayRef<char> method_name);
    torch::executor::Result<Vec<torch::executor::EValue>> Module_execute(torch::executor::Module *self, torch::executor::ArrayRef<char> method_name, torch::executor::ArrayRef<torch::executor::EValue> inputs);
#endif

}
