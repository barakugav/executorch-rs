#pragma once

#include "executorch_rs/defines.h"

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
        executorch::runtime::EValue *data;
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
    struct ArrayRefUsizeType
    {
        const size_t *data;
        size_t len;
    };
    struct ArrayRefSizesType
    {
        const executorch::aten::SizesType *data;
        size_t len;
    };
    struct ArrayRefDimOrderType
    {
        const executorch::aten::DimOrderType *data;
        size_t len;
    };
    struct ArrayRefStridesType
    {
        const executorch::aten::StridesType *data;
        size_t len;
    };
    struct ArrayRefEValue
    {
        const executorch::runtime::EValue *data;
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

    executorch::runtime::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *MemoryAllocator_allocate(executorch::runtime::MemoryAllocator &self, size_t size, size_t alignment);
#if defined(EXECUTORCH_RS_STD)
    executorch::extension::MallocMemoryAllocator MallocMemoryAllocator_new();
    void MallocMemoryAllocator_destructor(executorch::extension::MallocMemoryAllocator &self);
#endif
    executorch::runtime::HierarchicalAllocator HierarchicalAllocator_new(SpanSpanU8 buffers);
    void HierarchicalAllocator_destructor(executorch::runtime::HierarchicalAllocator &self);
    executorch::runtime::Error FileDataLoader_new(const char *file_path, size_t alignment, executorch::extension::FileDataLoader *out);
    executorch::extension::FileDataLoader FileDataLoader_new(const char *file_path, size_t alignment);
    executorch::runtime::Error MmapDataLoader_new(const char *file_path, executorch::extension::MmapDataLoader::MlockConfig mlock_config, executorch::extension::MmapDataLoader *out);

    // Program
    executorch::runtime::Error Program_load(executorch::runtime::DataLoader *loader, executorch::runtime::Program::Verification verification, executorch::runtime::Program *out);
    executorch::runtime::Error Program_load_method(const executorch::runtime::Program &self, const char *method_name, executorch::runtime::MemoryManager *memory_manager, executorch::runtime::EventTracer *event_tracer, executorch::runtime::Method *out);
    executorch::runtime::Error Program_get_method_name(const executorch::runtime::Program &self, size_t method_index, const char **out);
    executorch::runtime::Error Program_method_meta(const executorch::runtime::Program &self, const char *method_name, executorch::runtime::MethodMeta *method_meta_out);
    void Program_destructor(executorch::runtime::Program &self);

    // MethodMeta
    executorch::runtime::Error MethodMeta_input_tag(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::Tag *tag_out);
    executorch::runtime::Error MethodMeta_output_tag(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::Tag *tag_out);
    executorch::runtime::Error MethodMeta_input_tensor_meta(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::TensorInfo *tensor_info_out);
    executorch::runtime::Error MethodMeta_output_tensor_meta(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::TensorInfo *tensor_info_out);
    executorch::runtime::Error MethodMeta_memory_planned_buffer_size(const executorch::runtime::MethodMeta &self, size_t index, int64_t *size_out);

    // TensorInfo
    ArrayRefI32 TensorInfo_sizes(const executorch::runtime::TensorInfo &self);
    ArrayRefU8 TensorInfo_dim_order(const executorch::runtime::TensorInfo &self);

    // Tensor
    void Tensor_new(executorch::aten::Tensor *self, executorch::aten::TensorImpl *tensor_impl);
    size_t Tensor_nbytes(const executorch::aten::Tensor &self);
    ssize_t Tensor_size(const executorch::aten::Tensor &self, ssize_t dim);
    ssize_t Tensor_dim(const executorch::aten::Tensor &self);
    ssize_t Tensor_numel(const executorch::aten::Tensor &self);
    executorch::aten::ScalarType Tensor_scalar_type(const executorch::aten::Tensor &self);
    ssize_t Tensor_element_size(const executorch::aten::Tensor &self);
    ArrayRefSizesType Tensor_sizes(const executorch::aten::Tensor &self);
    ArrayRefDimOrderType Tensor_dim_order(const executorch::aten::Tensor &self);
    ArrayRefStridesType Tensor_strides(const executorch::aten::Tensor &self);
    const void *Tensor_const_data_ptr(const executorch::aten::Tensor &self);
    void *Tensor_mutable_data_ptr(const executorch::aten::Tensor &self);
    ssize_t Tensor_coordinate_to_index(const executorch::aten::Tensor &self, ArrayRefUsizeType coordinate);
    void Tensor_destructor(executorch::aten::Tensor &self);

    // executorch::runtime::EValue EValue_shallow_clone(executorch::runtime::EValue *evalue);
    void EValue_new_from_i64(executorch::runtime::EValue *self, int64_t value);
    void EValue_new_from_f64(executorch::runtime::EValue *self, double value);
    void EValue_new_from_f64_arr(executorch::runtime::EValue *self, ArrayRefF64 value);
    void EValue_new_from_bool(executorch::runtime::EValue *self, bool value);
    void EValue_new_from_bool_arr(executorch::runtime::EValue *self, ArrayRefBool value);
    void EValue_new_from_chars(executorch::runtime::EValue *self, ArrayRefChar value);
    void EValue_new_from_tensor(executorch::runtime::EValue *self, const executorch::aten::Tensor *value);
    int64_t EValue_as_i64(const executorch::runtime::EValue &self);
    double EValue_as_f64(const executorch::runtime::EValue &self);
    bool EValue_as_bool(const executorch::runtime::EValue &self);
    ArrayRefChar EValue_as_string(const executorch::runtime::EValue &self);
    ArrayRefBool EValue_as_bool_list(const executorch::runtime::EValue &self);
    ArrayRefF64 EValue_as_f64_list(const executorch::runtime::EValue &self);
    void EValue_copy(const executorch::runtime::EValue *src, executorch::runtime::EValue *dst);
    void EValue_destructor(executorch::runtime::EValue &self);
    void EValue_move(executorch::runtime::EValue *src, executorch::runtime::EValue *dst);
    // executorch::aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const executorch::runtime::EValue<int64_t> &self);
    // executorch::aten::ArrayRef<executorch::aten::Tensor> BoxedEvalueList_Tensor_get(const executorch::runtime::EValue<executorch::aten::Tensor> &self);

    executorch::extension::BufferDataLoader BufferDataLoader_new(const void *data, size_t size);

#if defined(EXECUTORCH_RS_MODULE)
    void Module_new(executorch::extension::Module *self, ArrayRefChar file_path, const executorch::extension::Module::LoadMode load_mode, executorch::runtime::EventTracer *event_tracer);
    void Module_destructor(executorch::extension::Module &self);
    executorch::runtime::Error Module_method_names(executorch::extension::Module &self, VecVecChar *method_names_out);
    executorch::runtime::Error Module_load_method(executorch::extension::Module &self, ArrayRefChar method_name);
    bool Module_is_method_loaded(const executorch::extension::Module &self, ArrayRefChar method_name);
    executorch::runtime::Error Module_method_meta(executorch::extension::Module &self, ArrayRefChar method_name, executorch::runtime::MethodMeta *method_meta_out);
    executorch::runtime::Error Module_execute(executorch::extension::Module &self, ArrayRefChar method_name, ArrayRefEValue inputs, VecEValue *outputs);
#endif

}
