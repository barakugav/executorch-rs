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

#include "executorch/runtime/core/data_loader.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"
#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/data_loader/mmap_data_loader.h"

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
    struct VecChar
    {
        char *data;
        size_t len;
        size_t cap;
    };
    void VecChar_destructor(VecChar *vec);

    struct VecVecChar
    {
        VecChar *data;
        size_t len;
        size_t cap;
    };
    void VecVecChar_destructor(VecVecChar *vec);

    struct VecEValue
    {
        torch::executor::EValue *data;
        size_t len;
        size_t cap;
    };
    void VecEValue_destructor(VecEValue *vec);
#endif

    struct ArrayRefChar
    {
        const char *data;
        size_t len;
    };
    struct ArrayRefBool
    {
        const bool *data;
        size_t len;
    };
    struct ArrayRefU8
    {
        const uint8_t *data;
        size_t len;
    };
    struct ArrayRefI32
    {
        const int32_t *data;
        size_t len;
    };
    struct ArrayRefF64
    {
        const double *data;
        size_t len;
    };
    struct ArrayRefSizesType
    {
        const exec_aten::SizesType *data;
        size_t len;
    };
    struct ArrayRefDimOrderType
    {
        const exec_aten::DimOrderType *data;
        size_t len;
    };
    struct ArrayRefStridesType
    {
        const exec_aten::StridesType *data;
        size_t len;
    };
    struct ArrayRefEValue
    {
        const torch::executor::EValue *data;
        size_t len;
    };
    struct SpanU8
    {
        uint8_t *data;
        size_t len;
    };
    struct SpanSpanU8
    {
        SpanU8 *data;
        size_t len;
    };

    torch::executor::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *MemoryAllocator_allocate(torch::executor::MemoryAllocator &self, size_t size, size_t alignment);
#if defined(EXECUTORCH_RS_STD)
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new();
    void MallocMemoryAllocator_destructor(torch::executor::util::MallocMemoryAllocator &self);
#endif
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(SpanSpanU8 buffers);
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator &self);
    torch::executor::Error FileDataLoader_new(const char *file_path, size_t alignment, torch::executor::util::FileDataLoader *out);
    torch::executor::util::FileDataLoader FileDataLoader_new(const char *file_path, size_t alignment);
    torch::executor::Error MmapDataLoader_new(const char *file_path, torch::executor::util::MmapDataLoader::MlockConfig mlock_config, torch::executor::util::MmapDataLoader *out);

    // Program
    torch::executor::Error Program_load(torch::executor::DataLoader *loader, torch::executor::Program::Verification verification, torch::executor::Program *out);
    torch::executor::Error Program_load_method(const torch::executor::Program &self, const char *method_name, torch::executor::MemoryManager *memory_manager, torch::executor::EventTracer *event_tracer, torch::executor::Method *out);
    torch::executor::Error Program_get_method_name(const torch::executor::Program &self, size_t method_index, const char **out);
    torch::executor::Error Program_method_meta(const torch::executor::Program &self, const char *method_name, torch::executor::MethodMeta *method_meta_out);
    void Program_destructor(torch::executor::Program &self);

    // MethodMeta
    torch::executor::Error MethodMeta_input_tag(const torch::executor::MethodMeta &self, size_t index, torch::executor::Tag *tag_out);
    torch::executor::Error MethodMeta_output_tag(const torch::executor::MethodMeta &self, size_t index, torch::executor::Tag *tag_out);
    torch::executor::Error MethodMeta_input_tensor_meta(const torch::executor::MethodMeta &self, size_t index, torch::executor::TensorInfo *tensor_info_out);
    torch::executor::Error MethodMeta_output_tensor_meta(const torch::executor::MethodMeta &self, size_t index, torch::executor::TensorInfo *tensor_info_out);
    torch::executor::Error MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta &self, size_t index, int64_t *size_out);

    // TensorInfo
    ArrayRefI32 TensorInfo_sizes(const torch::executor::TensorInfo &self);
    ArrayRefU8 TensorInfo_dim_order(const torch::executor::TensorInfo &self);

    // Tensor
    void Tensor_new(exec_aten::Tensor *self, exec_aten::TensorImpl *tensor_impl);
    size_t Tensor_nbytes(const exec_aten::Tensor &self);
    ssize_t Tensor_size(const exec_aten::Tensor &self, ssize_t dim);
    ssize_t Tensor_dim(const exec_aten::Tensor &self);
    ssize_t Tensor_numel(const exec_aten::Tensor &self);
    exec_aten::ScalarType Tensor_scalar_type(const exec_aten::Tensor &self);
    ssize_t Tensor_element_size(const exec_aten::Tensor &self);
    ArrayRefSizesType Tensor_sizes(const exec_aten::Tensor &self);
    ArrayRefDimOrderType Tensor_dim_order(const exec_aten::Tensor &self);
    ArrayRefStridesType Tensor_strides(const exec_aten::Tensor &self);
    const void *Tensor_const_data_ptr(const exec_aten::Tensor &self);
    void *Tensor_mutable_data_ptr(const exec_aten::Tensor &self);
    size_t Tensor_coordinate_to_index(const exec_aten::Tensor &self, const size_t *coordinate);
    void Tensor_destructor(exec_aten::Tensor &self);

    // torch::executor::EValue EValue_shallow_clone(torch::executor::EValue *evalue);
    void EValue_new_from_i64(torch::executor::EValue *self, int64_t value);
    void EValue_new_from_f64(torch::executor::EValue *self, double value);
    void EValue_new_from_f64_arr(torch::executor::EValue *self, ArrayRefF64 value);
    void EValue_new_from_bool(torch::executor::EValue *self, bool value);
    void EValue_new_from_bool_arr(torch::executor::EValue *self, ArrayRefBool value);
    void EValue_new_from_chars(torch::executor::EValue *self, ArrayRefChar value);
    void EValue_new_from_tensor(torch::executor::EValue *self, const exec_aten::Tensor *value);
    int64_t EValue_as_i64(const torch::executor::EValue &self);
    double EValue_as_f64(const torch::executor::EValue &self);
    bool EValue_as_bool(const torch::executor::EValue &self);
    ArrayRefChar EValue_as_string(const torch::executor::EValue &self);
    ArrayRefBool EValue_as_bool_list(const torch::executor::EValue &self);
    ArrayRefF64 EValue_as_f64_list(const torch::executor::EValue &self);
    void EValue_copy(const torch::executor::EValue *src, torch::executor::EValue *dst);
    void EValue_destructor(torch::executor::EValue &self);
    void EValue_move(torch::executor::EValue *src, torch::executor::EValue *dst);
    // exec_aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const torch::executor::BoxedEvalueList<int64_t> &self);
    // exec_aten::ArrayRef<exec_aten::Tensor> BoxedEvalueList_Tensor_get(const torch::executor::BoxedEvalueList<exec_aten::Tensor> &self);

    torch::executor::util::BufferDataLoader BufferDataLoader_new(const void *data, size_t size);

#if defined(EXECUTORCH_RS_MODULE)
    void Module_new(torch::executor::Module *self, ArrayRefChar file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer);
    void Module_destructor(torch::executor::Module &self);
    torch::executor::Error Module_method_names(torch::executor::Module &self, VecVecChar *method_names_out);
    torch::executor::Error Module_load_method(torch::executor::Module &self, ArrayRefChar method_name);
    bool Module_is_method_loaded(const torch::executor::Module &self, ArrayRefChar method_name);
    torch::executor::Error Module_method_meta(torch::executor::Module &self, ArrayRefChar method_name, torch::executor::MethodMeta *method_meta_out);
    torch::executor::Error Module_execute(torch::executor::Module &self, ArrayRefChar method_name, ArrayRefEValue inputs, VecEValue *outputs);
#endif

}
