
#include <cstddef>
#include <vector>

#include "executorch_rs/c_bridge.h"
#include "executorch_rs/layout.hpp"

#include "executorch/runtime/platform/platform.h"
#include "executorch/runtime/core/error.h"
#include "executorch/runtime/core/span.h"
#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/memory_manager.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/exec_aten/util/tensor_util.h"
#include "executorch/runtime/core/exec_aten/util/dim_order_util.h"
#include "executorch/runtime/platform/assert.h"

#include "executorch/runtime/core/data_loader.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"
#if defined(EXECUTORCH_RS_DATA_LOADER)
#include "executorch/extension/data_loader/file_data_loader.h"
#include "executorch/extension/data_loader/mmap_data_loader.h"
#endif

#if defined(EXECUTORCH_RS_ETDUMP)
#include "executorch/devtools/etdump/etdump_flatcc.h"
#endif

#if defined(EXECUTORCH_RS_STD)
#include "executorch/extension/memory_allocator/malloc_memory_allocator.h"
#endif

// Layout asserts
namespace
{
    using executorch_rs::is_equal_layout;

    static_assert(is_equal_layout<EValueStorage, executorch::runtime::EValue>());
    static_assert(is_equal_layout<TensorStorage, executorch::aten::Tensor>());
    static_assert(is_equal_layout<OptionalTensorStorage, executorch::aten::optional<executorch::aten::Tensor>>());

    static_assert(is_equal_layout<TensorImpl, executorch::aten::TensorImpl>());
    static_assert(std::is_trivially_move_constructible_v<executorch::aten::TensorImpl>);

    static_assert(is_equal_layout<Program, executorch::runtime::Program>());
    // Program is not trivially move constructible because it has a FreeableBuffer field that
    // has a custom move constructor.
    // FreeableBuffer has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::Program>);

    static_assert(is_equal_layout<TensorInfo, executorch::runtime::TensorInfo>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::TensorInfo>);

    static_assert(is_equal_layout<MethodMeta, executorch::runtime::MethodMeta>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MethodMeta>);

    static_assert(is_equal_layout<Method, executorch::runtime::Method>());
    // Method has a move constructor that just clean the old object to avoid double free.
    // Its OK to move it in Rust because the old object is forgotten.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::Method>);

    static_assert(is_equal_layout<BufferDataLoader, executorch::extension::BufferDataLoader>());
    // BufferDataLoader is not trivially move constructible because it has a vtable with virtual
    // destructor inherited from DataLoader, but it has an empty implementation for it therefore
    // it is safe.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::BufferDataLoader>);

#if defined(EXECUTORCH_RS_DATA_LOADER)
    static_assert(is_equal_layout<FileDataLoader, executorch::extension::FileDataLoader>());
    // FileDataLoader has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::FileDataLoader>);

    static_assert(is_equal_layout<MmapDataLoader, executorch::extension::MmapDataLoader>());
    // MmapDataLoader has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::MmapDataLoader>);
#endif

    static_assert(is_equal_layout<MemoryAllocator, executorch::runtime::MemoryAllocator>());
    // MemoryAllocator is not trivially move constructible because it has a vtable with virtual
    // destructor, but when we have a concrete instance of it there is nothing virtual and no move
    // constructor, so it is safe to move it in Rust.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MemoryAllocator>);

    static_assert(is_equal_layout<HierarchicalAllocator, executorch::runtime::HierarchicalAllocator>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::HierarchicalAllocator>);

    static_assert(is_equal_layout<MemoryManager, executorch::runtime::MemoryManager>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MemoryManager>);

#if defined(EXECUTORCH_RS_ETDUMP)
    static_assert(is_equal_layout<ETDumpGen, executorch::etdump::ETDumpGen>());
// MemoryAllocator is not trivially move constructible because it has a vtable with virtual
// destructor, but when we have a concrete instance of it there is nothing virtual and no move
// constructor, so it is safe to move it in Rust.
//
// static_assert(std::is_trivially_move_constructible_v<executorch::etdump::ETDumpGen>);
#endif

}

constexpr size_t MAX_DIM = 16;

using executorch_rs::checked_reinterpret_cast;

#if defined(EXECUTORCH_RS_STD)
void executorch_VecChar_destructor(struct VecChar *vec)
{
    delete[] vec->data;
}
void executorch_VecVecChar_destructor(struct VecVecChar *vec)
{
    for (size_t i = 0; i < vec->len; i++)
    {
        executorch_VecChar_destructor(&vec->data[i]);
    }
    delete[] vec->data;
}
void executorch_VecEValue_destructor(struct VecEValue *vec)
{
    // Its safe to call the destructor of elements in `vec->data[len..cap]` because we created them with `new T[len]`
    // aka default constructor
    auto data = reinterpret_cast<executorch::runtime::EValue *>(vec->data.ptr);
    delete[] data;
}
#endif

template <typename T>
static enum Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
{
    if (result.ok())
        *output = std::move(result.get());
    return static_cast<Error>(result.error());
}

void executorch_pal_init()
{
    et_pal_init();
}

struct MemoryAllocator executorch_MemoryAllocator_new(uint32_t size, uint8_t *base_address)
{
    struct MemoryAllocator self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(&self);
    new (self_) executorch::runtime::MemoryAllocator(size, base_address);
    return self;
}
void *executorch_MemoryAllocator_allocate(struct MemoryAllocator *self, size_t size, size_t alignment)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(self);
    return self_->allocate(size, alignment);
}
struct HierarchicalAllocator executorch_HierarchicalAllocator_new(struct SpanSpanU8 buffers)
{
    auto buffers_ = *checked_reinterpret_cast<executorch::runtime::Span<executorch::runtime::Span<uint8_t>>>(&buffers);
    ET_CHECK((void *)buffers_.begin() == (void *)buffers.data);
    ET_CHECK(buffers_.size() == buffers.len);
    struct HierarchicalAllocator self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(&self);
    new (self_) executorch::runtime::HierarchicalAllocator(buffers_);
    return self;
}
void executorch_HierarchicalAllocator_destructor(struct HierarchicalAllocator *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(self);
    self_->~HierarchicalAllocator();
}
struct MemoryManager executorch_MemoryManager_new(
    struct MemoryAllocator *method_allocator,
    struct HierarchicalAllocator *planned_memory,
    struct MemoryAllocator *temp_allocator)
{
    auto method_allocator_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(method_allocator);
    auto planned_memory_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(planned_memory);
    auto temp_allocator_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(temp_allocator);

    struct MemoryManager self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryManager>(&self);
    new (self_) executorch::runtime::MemoryManager(method_allocator_, planned_memory_, temp_allocator_);
    return self;
}

// Loaders
static executorch::runtime::DataLoader *cast_data_loader_mut(struct DataLoaderRefMut loader)
{
    return reinterpret_cast<executorch::runtime::DataLoader *>(loader.ptr);
}
static struct DataLoaderRefMut cast_data_loader_mut(executorch::runtime::DataLoader *loader)
{
    return DataLoaderRefMut{.ptr = loader};
}
struct BufferDataLoader executorch_BufferDataLoader_new(const void *data, size_t size)
{
    struct BufferDataLoader loader;
    auto loader_ = checked_reinterpret_cast<executorch::extension::BufferDataLoader>(&loader);
    new (loader_) executorch::extension::BufferDataLoader(data, size);
    return loader;
}
struct DataLoaderRefMut executorch_BufferDataLoader_as_data_loader_mut(struct BufferDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::BufferDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
#if defined(EXECUTORCH_RS_DATA_LOADER)
enum Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, struct FileDataLoader *out)
{
    auto out_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(out);
    // return extract_result(std::move(executorch::extension::FileDataLoader::from(file_path, alignment)), out);
    auto res = executorch::extension::FileDataLoader::from(file_path, alignment);
    if (!res.ok())
        return static_cast<Error>(res.error());
    auto &loader = res.get();
    new (out_) executorch::extension::FileDataLoader(std::move(loader));
    return Error::Error_Ok;
}
void executorch_FileDataLoader_destructor(struct FileDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(self);
    self_->~FileDataLoader();
}
struct DataLoaderRefMut executorch_FileDataLoader_as_data_loader_mut(struct FileDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
enum Error executorch_MmapDataLoader_new(const char *file_path, enum MmapDataLoaderMlockConfig mlock_config, struct MmapDataLoader *out)
{
    auto mlock_config_ = static_cast<executorch::extension::MmapDataLoader::MlockConfig>(mlock_config);
    auto out_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(out);
    // return extract_result(executorch::extension::MmapDataLoader::from(file_path, mlock_config), out);
    auto res = executorch::extension::MmapDataLoader::from(file_path, mlock_config_);
    if (!res.ok())
        return static_cast<Error>(res.error());
    auto &loader = res.get();
    new (out_) executorch::extension::MmapDataLoader(std::move(loader));
    return Error::Error_Ok;
}
void executorch_MmapDataLoader_destructor(struct MmapDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(self);
    self_->~MmapDataLoader();
}
struct DataLoaderRefMut executorch_MmapDataLoader_as_data_loader_mut(struct MmapDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
#endif

// Tensor
static const executorch::aten::Tensor *cast_tensor(struct TensorRef tensor)
{
    return reinterpret_cast<const executorch::aten::Tensor *>(tensor.ptr);
}
static struct TensorRef cast_tensor(const executorch::aten::Tensor *tensor)
{
    return TensorRef{.ptr = tensor};
}
static executorch::aten::Tensor *cast_tensor_mut(struct TensorRefMut tensor)
{
    return reinterpret_cast<executorch::aten::Tensor *>(tensor.ptr);
}
// static TensorMut cast_tensor_mut(executorch::aten::Tensor *tensor)
// {
//     return reinterpret_cast<TensorMut>(tensor);
// }

bool executorch_is_valid_dim_order_and_strides(size_t dim, const SizesType *sizes, const DimOrderType *dim_order, const StridesType *strides)
{
    ET_CHECK_MSG(dim <= MAX_DIM, "dim > 16");

    StridesType computed_strides[MAX_DIM];
    auto error = executorch::runtime::dim_order_to_stride(
        sizes, dim_order, dim, &computed_strides[0]);
    if (error != executorch::runtime::Error::Ok)
        return false; // Invalid dim order

    for (size_t i = 0; i < dim; i++)
        if (computed_strides[i] != strides[i])
            return false;
    return true;
}
enum Error executorch_stride_to_dim_order(const StridesType *strides, size_t dims, DimOrderType *dim_order)
{
    return static_cast<Error>(executorch::runtime::stride_to_dim_order(strides, dims, dim_order));
}

void executorch_TensorImpl_new(
    struct TensorImpl *self,
    enum ScalarType type,
    size_t dim,
    SizesType *sizes,
    void *data,
    DimOrderType *dim_order,
    StridesType *strides,
    enum TensorShapeDynamism dynamism)
{
    auto self_ = checked_reinterpret_cast<executorch::aten::TensorImpl>(self);
    new (self_) executorch::aten::TensorImpl(
        static_cast<executorch::aten::ScalarType>(type),
        dim,
        static_cast<executorch::aten::SizesType *>(sizes),
        data,
        static_cast<executorch::aten::DimOrderType *>(dim_order),
        static_cast<executorch::aten::StridesType *>(strides),
        static_cast<executorch::aten::TensorShapeDynamism>(dynamism));
}
void executorch_Tensor_new(struct TensorRefMut self, struct TensorImpl *tensor_impl)
{
    auto self_ = cast_tensor_mut(self);
    auto tensor_impl_ = checked_reinterpret_cast<executorch::aten::TensorImpl>(tensor_impl);
    new (self_) executorch::aten::Tensor(tensor_impl_);
}
size_t executorch_Tensor_nbytes(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->nbytes();
}
size_t executorch_Tensor_size(struct TensorRef self, size_t dim)
{
    auto self_ = cast_tensor(self);
    return self_->size(dim);
}
size_t executorch_Tensor_dim(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->dim();
}
size_t executorch_Tensor_numel(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->numel();
}
enum ScalarType executorch_Tensor_scalar_type(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto ret = self_->scalar_type();
    return static_cast<ScalarType>(ret);
}
size_t executorch_Tensor_element_size(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->element_size();
}
struct ArrayRefSizesType executorch_Tensor_sizes(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto sizes = self_->sizes();
    return ArrayRefSizesType{
        .data = sizes.data(),
        .len = sizes.size(),
    };
}
struct ArrayRefDimOrderType executorch_Tensor_dim_order(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto dim_order = self_->dim_order();
    return ArrayRefDimOrderType{
        .data = dim_order.data(),
        .len = dim_order.size(),
    };
}
struct ArrayRefStridesType executorch_Tensor_strides(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto strides = self_->strides();
    return ArrayRefStridesType{
        .data = strides.data(),
        .len = strides.size(),
    };
}
const void *executorch_Tensor_const_data_ptr(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->const_data_ptr();
}
void *executorch_Tensor_mutable_data_ptr(struct TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->mutable_data_ptr();
}

int64_t executorch_Tensor_coordinate_to_index(struct TensorRef self, struct ArrayRefUsizeType coordinate)
{
    auto self_ = cast_tensor(self);
    auto ndim = (size_t)self_->dim();
    if (coordinate.len != ndim)
    {
        return -1;
    }

    auto sizes = self_->sizes();
    auto strides = self_->strides();
    auto dim_order = self_->dim_order();
    ET_CHECK_MSG(sizes.size() == ndim, "Sizes must have the same number of dimensions as the tensor");
    ET_CHECK_MSG(strides.size() == ndim, "Strides must have the same number of dimensions as the tensor");
    // TODO: support dim order
    ET_CHECK_MSG(
        dim_order.data() == nullptr || executorch::runtime::is_contiguous_dim_order(dim_order.data(), ndim),
        "Only contiguous dim order is supported for now");

    size_t index = 0;
    for (size_t d = 0; d < ndim; d++)
    {
        if (coordinate.data[d] >= (size_t)sizes[d])
        {
            return -1;
        }
        index += coordinate.data[d] * strides[d];
    }
    return index;
}
void executorch_Tensor_destructor(struct TensorRefMut self)
{
    auto self_ = cast_tensor_mut(self);
    self_->~Tensor();
}

// OptionalTensor
static const executorch::aten::optional<executorch::aten::Tensor> *cast_optional_tensor(struct OptionalTensorRef tensor)
{
    return reinterpret_cast<const executorch::aten::optional<executorch::aten::Tensor> *>(tensor.ptr);
}
static executorch::aten::optional<executorch::aten::Tensor> *cast_optional_tensor_mut(struct OptionalTensorRefMut tensor)
{
    return reinterpret_cast<executorch::aten::optional<executorch::aten::Tensor> *>(tensor.ptr);
}
struct TensorRef executorch_OptionalTensor_get(struct OptionalTensorRef self)
{
    auto self_ = cast_optional_tensor(self);
    if (!self_->has_value())
        return TensorRef{.ptr = nullptr};
    const executorch::aten::Tensor *tensor = &self_->value();
    return cast_tensor(tensor);
}

// EValue
static const executorch::runtime::EValue *cast_evalue(struct EValueRef evalue)
{
    return reinterpret_cast<const executorch::runtime::EValue *>(evalue.ptr);
}
static struct EValueRef cast_evalue(const executorch::runtime::EValue *evalue)
{
    return EValueRef{.ptr = evalue};
}
static executorch::runtime::EValue *cast_evalue_mut(struct EValueRefMut evalue)
{
    return reinterpret_cast<executorch::runtime::EValue *>(evalue.ptr);
}
// static EValueMut cast_evalue_mut(executorch::runtime::EValue *evalue)
// {
//     return evalue;
// }
void executorch_EValue_new_none(struct EValueRefMut self)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue();
}
void executorch_EValue_new_from_i64(struct EValueRefMut self, int64_t value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_i64_list(struct EValueRefMut self, struct BoxedEvalueListI64 value)
{
    auto self_ = cast_evalue_mut(self);
    ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
    auto wrapped_vals =
        const_cast<executorch::runtime::EValue **>(reinterpret_cast<const executorch::runtime::EValue *const *>(value.wrapped_vals.data));
    executorch::runtime::BoxedEvalueList<int64_t> list(
        wrapped_vals,
        value.unwrapped_vals.data,
        (int)value.wrapped_vals.len);
    new (self_) executorch::runtime::EValue(list);
}
void executorch_EValue_new_from_f64(struct EValueRefMut self, double value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_f64_list(struct EValueRefMut self, struct ArrayRefF64 value)
{
    auto self_ = cast_evalue_mut(self);
    executorch::aten::ArrayRef<double> value_(value.data, value.len);
    new (self_) executorch::runtime::EValue(value_);
}
void executorch_EValue_new_from_bool(struct EValueRefMut self, bool value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_bool_list(struct EValueRefMut self, struct ArrayRefBool value)
{
    auto self_ = cast_evalue_mut(self);
    executorch::aten::ArrayRef<bool> value_(value.data, value.len);
    new (self_) executorch::runtime::EValue(value_);
}
void executorch_EValue_new_from_string(struct EValueRefMut self, struct ArrayRefChar value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value.data, value.len);
}
void executorch_EValue_new_from_tensor(struct EValueRefMut self, struct TensorRef value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = cast_tensor(value);
    new (self_) executorch::runtime::EValue(*value_);
}
void executorch_EValue_new_from_tensor_list(struct EValueRefMut self, struct BoxedEvalueListTensor value)
{
    auto self_ = cast_evalue_mut(self);
    ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
    auto wrapped_vals =
        const_cast<executorch::runtime::EValue **>(reinterpret_cast<const executorch::runtime::EValue *const *>(value.wrapped_vals.data));
    executorch::runtime::BoxedEvalueList<executorch::aten::Tensor> list(
        wrapped_vals,
        cast_tensor_mut(value.unwrapped_vals.data),
        (int)value.wrapped_vals.len);
    new (self_) executorch::runtime::EValue(list);
}
void executorch_EValue_new_from_optional_tensor_list(struct EValueRefMut self, struct BoxedEvalueListOptionalTensor value)
{
    auto self_ = cast_evalue_mut(self);
    ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
    auto wrapped_vals =
        const_cast<executorch::runtime::EValue **>(reinterpret_cast<const executorch::runtime::EValue *const *>(value.wrapped_vals.data));
    auto unwrapped_vals = cast_optional_tensor_mut(value.unwrapped_vals.data);
    executorch::runtime::BoxedEvalueList<executorch::aten::optional<executorch::aten::Tensor>> list(
        wrapped_vals,
        unwrapped_vals,
        (int)value.wrapped_vals.len);
    new (self_) executorch::runtime::EValue(list);
}
enum Tag executorch_EValue_tag(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    return static_cast<Tag>(self_->tag);
}
int64_t executorch_EValue_as_i64(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toInt();
}
struct ArrayRefI64 executorch_EValue_as_i64_list(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toIntList();
    return ArrayRefI64{
        .data = list.data(),
        .len = list.size(),
    };
}
double executorch_EValue_as_f64(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toDouble();
}
struct ArrayRefF64 executorch_EValue_as_f64_list(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toDoubleList();
    return ArrayRefF64{
        .data = list.data(),
        .len = list.size(),
    };
}
bool executorch_EValue_as_bool(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toBool();
}
struct ArrayRefBool executorch_EValue_as_bool_list(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toBoolList();
    return ArrayRefBool{
        .data = list.data(),
        .len = list.size(),
    };
}
struct ArrayRefChar executorch_EValue_as_string(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto str = self_->toString();
    return ArrayRefChar{
        .data = str.data(),
        .len = str.size(),
    };
}
struct TensorRef executorch_EValue_as_tensor(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    const executorch::aten::Tensor *tensor = &self_->toTensor();
    return cast_tensor(tensor);
}
struct ArrayRefTensor executorch_EValue_as_tensor_list(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toTensorList();
    return ArrayRefTensor{
        .data = cast_tensor(list.data()),
        .len = list.size(),
    };
}
struct ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(struct EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toListOptionalTensor();
    return ArrayRefOptionalTensor{
        .data = OptionalTensorRef{.ptr = checked_reinterpret_cast<OptionalTensorStorage>(list.data())},
        .len = list.size(),
    };
}
void executorch_EValue_copy(struct EValueRef src, struct EValueRefMut dst)
{
    auto src_ = cast_evalue(src);
    auto dst_ = cast_evalue_mut(dst);
    new (dst_) executorch::runtime::EValue(*src_);
}
void executorch_EValue_destructor(struct EValueRefMut self)
{
    auto self_ = cast_evalue_mut(self);
    self_->~EValue();
}
void executorch_EValue_move(struct EValueRefMut src, struct EValueRefMut dst)
{
    auto src_ = cast_evalue_mut(src);
    auto dst_ = cast_evalue_mut(dst);
    new (dst_) executorch::runtime::EValue(std::move(*src_));
}

// Program
enum ProgramHeaderStatus executorch_Program_check_header(const void *data, size_t size)
{
    auto status = executorch::runtime::Program::check_header(data, size);
    return static_cast<ProgramHeaderStatus>(status);
}
enum Error executorch_Program_load(struct DataLoaderRefMut loader, enum ProgramVerification verification, struct Program *out)
{
    auto loader_ = cast_data_loader_mut(loader);
    auto verification_ = static_cast<executorch::runtime::Program::Verification>(verification);
    auto out_ = checked_reinterpret_cast<executorch::runtime::Program>(out);
    // return extract_result(executorch::runtime::Program::load(loader, verification), out);
    auto res = executorch::runtime::Program::load(loader_, verification_);
    if (!res.ok())
        return static_cast<Error>(res.error());
    auto &program = res.get();
    new (out_) executorch::runtime::Program(std::move(program));
    return Error::Error_Ok;
}
enum Error executorch_Program_load_method(const struct Program *self, const char *method_name, struct MemoryManager *memory_manager, struct EventTracerRefMut event_tracer, struct Method *out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    auto memory_manager_ = checked_reinterpret_cast<executorch::runtime::MemoryManager>(memory_manager);
    auto event_tracer_ = reinterpret_cast<executorch::runtime::EventTracer *>(event_tracer.ptr);
    auto out_ = checked_reinterpret_cast<executorch::runtime::Method>(out);

    auto res = self_->load_method(method_name, memory_manager_, event_tracer_);
    if (!res.ok())
        return static_cast<Error>(res.error());
    auto &method = res.get();
    new (out_) executorch::runtime::Method(std::move(method));
    return Error::Error_Ok;
}
enum Error executorch_Program_get_method_name(const struct Program *self, size_t method_index, const char **out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    return extract_result(self_->get_method_name(method_index), out);
}
enum Error executorch_Program_method_meta(const struct Program *self, const char *method_name, struct MethodMeta *method_meta_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    auto method_meta_out_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(method_meta_out);
    return extract_result(self_->method_meta(method_name), method_meta_out_);
}
size_t executorch_Program_num_methods(const struct Program *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    return self_->num_methods();
}
void executorch_Program_destructor(struct Program *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    self_->~Program();
}

// MethodMeta
size_t executorch_Method_inputs_size(const struct Method *self)
{
    auto *self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    return self_->inputs_size();
}
size_t executorch_Method_outputs_size(const struct Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    return self_->outputs_size();
}
enum Error executorch_Method_set_input(struct Method *self, struct EValueRef input_evalue, size_t input_idx)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    auto input_evalue_ = cast_evalue(input_evalue);
    executorch::runtime::Error ret = self_->set_input(*input_evalue_, input_idx);
    return static_cast<Error>(ret);
}
struct EValueRef executorch_Method_get_output(const struct Method *self, size_t i)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    const executorch::runtime::EValue *output = &self_->get_output(i);
    return cast_evalue(output);
}
enum Error executorch_Method_execute(struct Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    executorch::runtime::Error ret = self_->execute();
    return static_cast<Error>(ret);
}
void executorch_Method_destructor(struct Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    self_->~Method();
}
const char *executorch_MethodMeta_name(const struct MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->name();
    self_->num_inputs();
}
size_t executorch_MethodMeta_num_inputs(const struct MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_inputs();
}
size_t executorch_MethodMeta_num_outputs(const struct MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_outputs();
}
size_t executorch_MethodMeta_num_memory_planned_buffers(const struct MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_memory_planned_buffers();
}
enum Error executorch_MethodMeta_input_tag(const struct MethodMeta *self, size_t index, enum Tag *tag_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tag_out_ = checked_reinterpret_cast<executorch::runtime::Tag>(tag_out);
    return extract_result(self_->input_tag(index), tag_out_);
}
enum Error executorch_MethodMeta_output_tag(const struct MethodMeta *self, size_t index, enum Tag *tag_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tag_out_ = checked_reinterpret_cast<executorch::runtime::Tag>(tag_out);
    return extract_result(self_->output_tag(index), tag_out_);
}
enum Error executorch_MethodMeta_input_tensor_meta(const struct MethodMeta *self, size_t index, struct TensorInfo *tensor_info_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tensor_info_out_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(tensor_info_out);
    return extract_result(self_->input_tensor_meta(index), tensor_info_out_);
}
enum Error executorch_MethodMeta_output_tensor_meta(const struct MethodMeta *self, size_t index, struct TensorInfo *tensor_info_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tensor_info_out_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(tensor_info_out);
    return extract_result(self_->output_tensor_meta(index), tensor_info_out_);
}
enum Error executorch_MethodMeta_memory_planned_buffer_size(const struct MethodMeta *self, size_t index, int64_t *size_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return extract_result(self_->memory_planned_buffer_size(index), size_out);
}
bool executorch_MethodMeta_uses_backend(const struct MethodMeta *self, const char *backend_name)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->uses_backend(backend_name);
}
size_t executorch_MethodMeta_num_backends(const struct MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_backends();
}
enum Error executorch_MethodMeta_get_backend_name(const struct MethodMeta *self, size_t index, const char **backend_name_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return extract_result(self_->get_backend_name(index), backend_name_out);
}

// TensorInfo
struct ArrayRefI32 executorch_TensorInfo_sizes(const struct TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto sizes = self_->sizes();
    return ArrayRefI32{
        .data = sizes.data(),
        .len = sizes.size(),
    };
}
struct ArrayRefU8 executorch_TensorInfo_dim_order(const struct TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto dim_order = self_->dim_order();
    return ArrayRefU8{
        .data = dim_order.data(),
        .len = dim_order.size(),
    };
}
enum ScalarType executorch_TensorInfo_scalar_type(const struct TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto ret = self_->scalar_type();
    return static_cast<ScalarType>(ret);
}
size_t executorch_TensorInfo_nbytes(const struct TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    return self_->nbytes();
}

#if defined(EXECUTORCH_RS_ETDUMP)
// ETDumpGen
struct ETDumpGen executorch_ETDumpGen_new(struct SpanU8 buffer)
{
    struct ETDumpGen self;
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(&self);
    new (self_) executorch::etdump::ETDumpGen({buffer.data, buffer.len});
    return self;
}
struct ArrayRefU8 executorch_ETDumpGen_get_etdump_data(struct ETDumpGen *self)
{
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(self);
    auto res = self_->get_etdump_data();
    return ArrayRefU8{.data = (uint8_t *)res.buf, .len = res.size};
}
struct EventTracerRefMut executorch_ETDumpGen_as_event_tracer_mut(struct ETDumpGen *self)
{
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(self);
    auto tracer = static_cast<executorch::runtime::EventTracer *>(self_);
    return EventTracerRefMut{.ptr = tracer};
}
#endif
