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

namespace executorch_rs
{
    struct Layout
    {
        size_t size;
        size_t alignment;
    };
    template <typename T>
    constexpr Layout layout_of()
    {
        return Layout{
            .size = sizeof(T),
            .alignment = alignof(T),
        };
    }
    constexpr bool operator==(const Layout &lhs, const Layout &rhs)
    {
        return lhs.size == rhs.size && lhs.alignment == rhs.alignment;
    }

    struct EValue
    {
        size_t _blob[4];
    };
    static_assert(layout_of<EValue>() == layout_of<executorch::runtime::EValue>());
    struct Tensor
    {
        size_t _blob[1];
    };
    struct TensorImpl
    {
        size_t _blob[8];
    };
    static_assert(layout_of<TensorImpl>() == layout_of<executorch::aten::TensorImpl>());
    struct Program
    {
        size_t _blob[11];
    };
    static_assert(layout_of<Program>() == layout_of<executorch::runtime::Program>());
    struct TensorInfo
    {
        size_t _blob[6];
    };
    static_assert(layout_of<TensorInfo>() == layout_of<executorch::runtime::TensorInfo>());
    struct MethodMeta
    {
        size_t _blob[1];
    };
    static_assert(layout_of<MethodMeta>() == layout_of<executorch::runtime::MethodMeta>());
    struct Method
    {
        size_t _blob[14];
    };
    static_assert(layout_of<Method>() == layout_of<executorch::runtime::Method>());

    struct DataLoader
    {
        size_t _blob[1];
    };
    static_assert(layout_of<DataLoader>() == layout_of<executorch::runtime::DataLoader>());
    struct BufferDataLoader
    {
        size_t _blob[3];
    };
    static_assert(layout_of<BufferDataLoader>() == layout_of<executorch::extension::BufferDataLoader>());
#if defined(EXECUTORCH_RS_DATA_LOADER)
    struct FileDataLoader
    {
        size_t _blob[5];
    };
    static_assert(layout_of<FileDataLoader>() == layout_of<executorch::extension::FileDataLoader>());
    struct MmapDataLoader
    {
        size_t _blob_1[4];
        int _blob_2[2];
    };
    static_assert(layout_of<MmapDataLoader>() == layout_of<executorch::extension::MmapDataLoader>());
#endif

    struct OptionalTensor
    {
        // Why is this not static?
        struct trivial_init_t
        {
        } trivial_init{};

        union storage_t
        {
            /// A small, trivially-constructable alternative to T.
            unsigned char dummy_;
            /// The constructed value itself, if optional::has_value_ is true.
            Tensor value_;
        };
        storage_t storage_;
        bool init_;
    };
    static_assert(sizeof(executorch::aten::optional<executorch::aten::Tensor>) == sizeof(OptionalTensor));
    static_assert(std::alignment_of<executorch::aten::optional<executorch::aten::Tensor>>() == std::alignment_of<OptionalTensor>());

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
        EValue *data;
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
    struct ArrayRefI64
    {
        const int64_t *data;
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
    struct ArrayRefTensor
    {
        const Tensor *data;
        size_t len;
    };
    struct ArrayRefOptionalTensor
    {
        const OptionalTensor *data;
        size_t len;
    };
    struct ArrayRefEValue
    {
        const EValue *data;
        size_t len;
    };
    struct ArrayRefEValuePtr
    {
        const EValue *const *data;
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
    // struct SpanEValue
    // {
    //     EValue *data;
    //     size_t len;
    // };
    struct SpanI64
    {
        int64_t *data;
        size_t len;
    };
    struct SpanTensor
    {
        Tensor *data;
        size_t len;
    };
    struct SpanOptionalTensor
    {
        OptionalTensor *data;
        size_t len;
    };
    struct BoxedEvalueListI64
    {
        ArrayRefEValuePtr wrapped_vals;
        SpanI64 unwrapped_vals;
    };
    struct BoxedEvalueListTensor
    {
        ArrayRefEValuePtr wrapped_vals;
        SpanTensor unwrapped_vals;
    };
    struct BoxedEvalueListOptionalTensor
    {
        ArrayRefEValuePtr wrapped_vals;
        SpanOptionalTensor unwrapped_vals;
    };

    executorch::runtime::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *MemoryAllocator_allocate(executorch::runtime::MemoryAllocator &self, size_t size, size_t alignment);
#if defined(EXECUTORCH_RS_STD)
    executorch::extension::MallocMemoryAllocator MallocMemoryAllocator_new();
    void MallocMemoryAllocator_destructor(executorch::extension::MallocMemoryAllocator &self);
#endif
    executorch::runtime::HierarchicalAllocator HierarchicalAllocator_new(SpanSpanU8 buffers);
    void HierarchicalAllocator_destructor(executorch::runtime::HierarchicalAllocator &self);

    // Loaders
    executorch_rs::BufferDataLoader BufferDataLoader_new(const void *data, size_t size);
    const executorch_rs::DataLoader *executorch_BufferDataLoader_as_data_loader(const executorch_rs::BufferDataLoader *self);
#if defined(EXECUTORCH_RS_DATA_LOADER)
    executorch::runtime::Error FileDataLoader_new(const char *file_path, size_t alignment, executorch_rs::FileDataLoader *out);
    void executorch_FileDataLoader_destructor(executorch_rs::FileDataLoader *self);
    const executorch_rs::DataLoader *executorch_FileDataLoader_as_data_loader(const executorch_rs::FileDataLoader *self);
    executorch::runtime::Error MmapDataLoader_new(const char *file_path, executorch::extension::MmapDataLoader::MlockConfig mlock_config, executorch_rs::MmapDataLoader *out);
    void executorch_MmapDataLoader_destructor(executorch_rs::MmapDataLoader *self);
    const executorch_rs::DataLoader *executorch_MmapDataLoader_as_data_loader(const executorch_rs::MmapDataLoader *self);

#endif

    // Program
    executorch::runtime::Program::HeaderStatus executorch_Program_check_header(const void *data, size_t size);
    executorch::runtime::Error Program_load(executorch_rs::DataLoader *loader, executorch::runtime::Program::Verification verification, Program *out);
    executorch::runtime::Error Program_load_method(const Program *self, const char *method_name, executorch::runtime::MemoryManager *memory_manager, executorch::runtime::EventTracer *event_tracer, Method *out);
    executorch::runtime::Error Program_get_method_name(const Program *self, size_t method_index, const char **out);
    executorch::runtime::Error Program_method_meta(const Program *self, const char *method_name, MethodMeta *method_meta_out);
    size_t executorch_Program_num_methods(const Program *self);
    void Program_destructor(Program *self);

    // MethodMeta
    size_t executorch_Method_inputs_size(const Method *self);
    size_t executorch_Method_outputs_size(const Method *self);
    executorch::runtime::Error executorch_Method_set_input(Method *self, const EValue *input_evalue, size_t input_idx);
    const EValue *executorch_Method_get_output(const Method *self, size_t i);
    executorch::runtime::Error executorch_Method_execute(Method *self);
    void executorch_Method_destructor(Method *self);
    const char *executorch_MethodMeta_name(const MethodMeta *self);
    size_t executorch_MethodMeta_num_inputs(const MethodMeta *self);
    size_t executorch_MethodMeta_num_outputs(const MethodMeta *self);
    size_t executorch_MethodMeta_num_memory_planned_buffers(const MethodMeta *self);
    executorch::runtime::Error MethodMeta_input_tag(const MethodMeta *self, size_t index, executorch::runtime::Tag *tag_out);
    executorch::runtime::Error MethodMeta_output_tag(const MethodMeta *self, size_t index, executorch::runtime::Tag *tag_out);
    executorch::runtime::Error MethodMeta_input_tensor_meta(const MethodMeta *self, size_t index, TensorInfo *tensor_info_out);
    executorch::runtime::Error MethodMeta_output_tensor_meta(const MethodMeta *self, size_t index, TensorInfo *tensor_info_out);
    executorch::runtime::Error MethodMeta_memory_planned_buffer_size(const MethodMeta *self, size_t index, int64_t *size_out);

    // TensorInfo
    ArrayRefI32 TensorInfo_sizes(const TensorInfo *self);
    ArrayRefU8 TensorInfo_dim_order(const TensorInfo *self);
    executorch::aten::ScalarType executorch_TensorInfo_scalar_type(const TensorInfo *self);
    size_t executorch_TensorInfo_nbytes(const TensorInfo *self);

    // Tensor
    void executorch_TensorImpl_new(
        TensorImpl *self,
        executorch::aten::ScalarType type,
        ssize_t dim,
        executorch::aten::SizesType *sizes,
        void *data,
        executorch::aten::DimOrderType *dim_order,
        executorch::aten::StridesType *strides,
        executorch::aten::TensorShapeDynamism dynamism);
    void Tensor_new(Tensor *self, TensorImpl *tensor_impl);
    size_t Tensor_nbytes(const Tensor *self);
    ssize_t Tensor_size(const Tensor *self, ssize_t dim);
    ssize_t Tensor_dim(const Tensor *self);
    ssize_t Tensor_numel(const Tensor *self);
    executorch::aten::ScalarType Tensor_scalar_type(const Tensor *self);
    ssize_t Tensor_element_size(const Tensor *self);
    ArrayRefSizesType Tensor_sizes(const Tensor *self);
    ArrayRefDimOrderType Tensor_dim_order(const Tensor *self);
    ArrayRefStridesType Tensor_strides(const Tensor *self);
    const void *Tensor_const_data_ptr(const Tensor *self);
    void *Tensor_mutable_data_ptr(const Tensor *self);
    ssize_t Tensor_coordinate_to_index(const Tensor *self, ArrayRefUsizeType coordinate);
    void Tensor_destructor(Tensor *self);

    // EValue EValue_shallow_clone(EValue *evalue);
    void executorch_EValue_new_none(EValue *self);
    void EValue_new_from_i64(EValue *self, int64_t value);
    void EValue_new_from_i64_list(EValue *self, BoxedEvalueListI64 value);
    void EValue_new_from_f64(EValue *self, double value);
    void EValue_new_from_f64_list(EValue *self, ArrayRefF64 value);
    void EValue_new_from_bool(EValue *self, bool value);
    void EValue_new_from_bool_list(EValue *self, ArrayRefBool value);
    void EValue_new_from_string(EValue *self, ArrayRefChar value);
    void EValue_new_from_tensor(EValue *self, const Tensor *value);
    void EValue_new_from_tensor_list(EValue *self, BoxedEvalueListTensor value);
    void EValue_new_from_optional_tensor_list(EValue *self, BoxedEvalueListOptionalTensor value);
    executorch::runtime::Tag executorch_EValue_tag(const EValue *self);
    int64_t EValue_as_i64(const EValue *self);
    ArrayRefI64 EValue_as_i64_list(const EValue *self);
    double EValue_as_f64(const EValue *self);
    ArrayRefF64 EValue_as_f64_list(const EValue *self);
    bool EValue_as_bool(const EValue *self);
    ArrayRefBool EValue_as_bool_list(const EValue *self);
    ArrayRefChar EValue_as_string(const EValue *self);
    const Tensor *EValue_as_tensor(const EValue *self);
    ArrayRefTensor EValue_as_tensor_list(const EValue *self);
    ArrayRefOptionalTensor EValue_as_optional_tensor_list(const EValue *self);
    void EValue_copy(const EValue *src, EValue *dst);
    void EValue_destructor(EValue *self);
    void EValue_move(EValue *src, EValue *dst);
}
