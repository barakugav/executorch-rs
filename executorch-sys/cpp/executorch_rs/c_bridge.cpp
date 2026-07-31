
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

#if defined(EXECUTORCH_RS_FLAT_TENSOR)
#include "executorch/extension/flat_tensor/flat_tensor_data_map.h"
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

    static_assert(is_equal_layout<struct ET_ArrayRefF64, executorch::aten::ArrayRef<double>>());
    static_assert(is_equal_layout<struct ET_ArrayRefBool, executorch::aten::ArrayRef<bool>>());
    static_assert(is_equal_layout<struct ET_ArrayRefChar, executorch::aten::ArrayRef<char>>());
    static_assert(is_equal_layout<struct ET_BoxedEvalueListI64, executorch::runtime::BoxedEvalueList<int64_t>>());
    static_assert(is_equal_layout<struct ET_BoxedEvalueListTensor, executorch::runtime::BoxedEvalueList<executorch::aten::Tensor>>());
    static_assert(is_equal_layout<struct ET_BoxedEvalueListOptionalTensor, executorch::runtime::BoxedEvalueList<std::optional<executorch::aten::Tensor>>>());
    static_assert(is_equal_layout<struct ET_EValueStorage, executorch::runtime::EValue>());
    static_assert(is_equal_layout<struct ET_TensorStorage, executorch::aten::Tensor>());
    static_assert(is_equal_layout<struct ET_OptionalTensorStorage, executorch::aten::optional<executorch::aten::Tensor>>());

    static_assert(is_equal_layout<struct ET_TensorImpl, executorch::aten::TensorImpl>());
    static_assert(std::is_trivially_move_constructible_v<executorch::aten::TensorImpl>);

    static_assert(is_equal_layout<struct ET_FreeableBuffer, executorch::runtime::FreeableBuffer>());
    static_assert(is_equal_layout<struct ET_Program, executorch::runtime::Program>());
    // ET_Program is not trivially move constructible because it has a ET_FreeableBuffer field that
    // has a custom move constructor.
    // ET_FreeableBuffer has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::Program>);

    static_assert(is_equal_layout<struct ET_TensorInfo, executorch::runtime::TensorInfo>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::TensorInfo>);

    static_assert(is_equal_layout<struct ET_TensorLayout, executorch::runtime::TensorLayout>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::TensorLayout>);

    static_assert(is_equal_layout<struct ET_MethodMeta, executorch::runtime::MethodMeta>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MethodMeta>);

    static_assert(is_equal_layout<struct ET_Method, executorch::runtime::Method>());
    // ET_Method has a move constructor that just clean the old object to avoid double free.
    // Its OK to move it in Rust because the old object is forgotten.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::Method>);

#if defined(EXECUTORCH_RS_FLAT_TENSOR)
    static_assert(is_equal_layout<struct ET_FlatTensorDataMap, executorch::extension::FlatTensorDataMap>());
    // ET_FlatTensorDataMap is not trivially move constructible because it has a vtable with virtual
    // destructor inherited from DataLoader, but it has an empty implementation for it therefore
    // it is safe.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::FlatTensorDataMap>);
#endif

    static_assert(is_equal_layout<struct ET_BufferDataLoader, executorch::extension::BufferDataLoader>());
    // ET_BufferDataLoader is not trivially move constructible because it has a vtable with virtual
    // destructor inherited from DataLoader, but it has an empty implementation for it therefore
    // it is safe.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::BufferDataLoader>);

#if defined(EXECUTORCH_RS_DATA_LOADER)
    static_assert(is_equal_layout<struct ET_FileDataLoader, executorch::extension::FileDataLoader>());
    // ET_FileDataLoader has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::FileDataLoader>);

    static_assert(is_equal_layout<struct ET_MmapDataLoader, executorch::extension::MmapDataLoader>());
    // ET_MmapDataLoader has a custom move constructor and a destructor, but the move is trivial +cleaning
    // of the old object, which behave great with Rust move semantics as long as we only call the
    // destructor on the final object.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::extension::MmapDataLoader>);
#endif

    static_assert(is_equal_layout<struct ET_MemoryAllocator, executorch::runtime::MemoryAllocator>());
    // ET_MemoryAllocator is not trivially move constructible because it has a vtable with virtual
    // destructor, but when we have a concrete instance of it there is nothing virtual and no move
    // constructor, so it is safe to move it in Rust.
    //
    // static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MemoryAllocator>);

    static_assert(is_equal_layout<struct ET_HierarchicalAllocator, executorch::runtime::HierarchicalAllocator>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::HierarchicalAllocator>);

    static_assert(is_equal_layout<struct ET_MemoryManager, executorch::runtime::MemoryManager>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::MemoryManager>);

#if defined(EXECUTORCH_RS_ETDUMP)
    static_assert(is_equal_layout<struct ET_DumpGen, executorch::etdump::ETDumpGen>());
// ET_MemoryAllocator is not trivially move constructible because it has a vtable with virtual
// destructor, but when we have a concrete instance of it there is nothing virtual and no move
// constructor, so it is safe to move it in Rust.
//
// static_assert(std::is_trivially_move_constructible_v<executorch::etdump::ETDumpGen>);
#endif

    static_assert(is_equal_layout<executorch_timestamp_t, et_timestamp_t>());
    static_assert(std::is_trivially_move_constructible_v<et_timestamp_t>);
    static_assert(is_equal_layout<struct executorch_tick_ratio, et_tick_ratio_t>());
    static_assert(std::is_trivially_move_constructible_v<et_tick_ratio_t>);
    static_assert(is_equal_layout<enum executorch_pal_log_level, et_pal_log_level_t>());
    static_assert(std::is_trivially_move_constructible_v<et_pal_log_level_t>);
    static_assert(is_equal_layout<struct ExecutorchPalImpl, executorch::runtime::PalImpl>());
    static_assert(std::is_trivially_move_constructible_v<executorch::runtime::PalImpl>);
}

constexpr size_t MAX_DIM = 16;

using executorch_rs::checked_reinterpret_cast;

#if defined(EXECUTORCH_RS_STD)
void executorch_VecChar_destructor(struct ET_VecChar *vec)
{
    delete[] vec->data;
}
void executorch_VecVecChar_destructor(struct ET_VecVecChar *vec)
{
    for (size_t i = 0; i < vec->len; i++)
    {
        executorch_VecChar_destructor(&vec->data[i]);
    }
    delete[] vec->data;
}
void executorch_VecEValue_destructor(struct ET_VecEValue *vec)
{
    // Its safe to call the destructor of elements in `vec->data[len..cap]` because we created them with `new T[len]`
    // aka default constructor
    auto data = reinterpret_cast<executorch::runtime::EValue *>(vec->data.ptr);
    delete[] data;
}
#endif

template <typename T>
static enum ET_Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
{
    if (result.ok())
        *output = std::move(result.get());
    return static_cast<ET_Error>(result.error());
}

struct ET_MemoryAllocator executorch_MemoryAllocator_new(uint32_t size, uint8_t *base_address)
{
    struct ET_MemoryAllocator self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(&self);
    new (self_) executorch::runtime::MemoryAllocator(size, base_address);
    return self;
}
void *executorch_MemoryAllocator_allocate(struct ET_MemoryAllocator *self, size_t size, size_t alignment)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(self);
    return self_->allocate(size, alignment);
}
struct ET_HierarchicalAllocator executorch_HierarchicalAllocator_new(struct ET_SpanSpanU8 buffers)
{
    auto buffers_ = *checked_reinterpret_cast<executorch::runtime::Span<executorch::runtime::Span<uint8_t>>>(&buffers);
    ET_CHECK((void *)buffers_.begin() == (void *)buffers.data);
    ET_CHECK(buffers_.size() == buffers.len);
    struct ET_HierarchicalAllocator self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(&self);
    new (self_) executorch::runtime::HierarchicalAllocator(buffers_);
    return self;
}
void executorch_HierarchicalAllocator_destructor(struct ET_HierarchicalAllocator *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(self);
    self_->~HierarchicalAllocator();
}
struct ET_MemoryManager executorch_MemoryManager_new(
    struct ET_MemoryAllocator *method_allocator,
    struct ET_HierarchicalAllocator *planned_memory,
    struct ET_MemoryAllocator *temp_allocator)
{
    auto method_allocator_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(method_allocator);
    auto planned_memory_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(planned_memory);
    auto temp_allocator_ = checked_reinterpret_cast<executorch::runtime::MemoryAllocator>(temp_allocator);

    struct ET_MemoryManager self;
    auto self_ = checked_reinterpret_cast<executorch::runtime::MemoryManager>(&self);
    new (self_) executorch::runtime::MemoryManager(method_allocator_, planned_memory_, temp_allocator_);
    return self;
}

// Loaders
static executorch::runtime::DataLoader *cast_data_loader_mut(struct ET_DataLoaderRefMut loader)
{
    return reinterpret_cast<executorch::runtime::DataLoader *>(loader.ptr);
}
static struct ET_DataLoaderRefMut cast_data_loader_mut(executorch::runtime::DataLoader *loader)
{
    return ET_DataLoaderRefMut{.ptr = loader};
}
struct ET_BufferDataLoader executorch_BufferDataLoader_new(const void *data, size_t size)
{
    struct ET_BufferDataLoader loader;
    auto loader_ = checked_reinterpret_cast<executorch::extension::BufferDataLoader>(&loader);
    new (loader_) executorch::extension::BufferDataLoader(data, size);
    return loader;
}
struct ET_DataLoaderRefMut executorch_BufferDataLoader_as_data_loader_mut(struct ET_BufferDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::BufferDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
#if defined(EXECUTORCH_RS_DATA_LOADER)
enum ET_Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, struct ET_FileDataLoader *out)
{
    auto out_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(out);
    // return extract_result(std::move(executorch::extension::FileDataLoader::from(file_path, alignment)), out);
    auto res = executorch::extension::FileDataLoader::from(file_path, alignment);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    auto &loader = res.get();
    new (out_) executorch::extension::FileDataLoader(std::move(loader));
    return ET_Error::ET_Error_Ok;
}
void executorch_FileDataLoader_destructor(struct ET_FileDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(self);
    self_->~FileDataLoader();
}
struct ET_DataLoaderRefMut executorch_FileDataLoader_as_data_loader_mut(struct ET_FileDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::FileDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
enum ET_Error executorch_MmapDataLoader_new(const char *file_path, enum ET_MmapDataLoaderMlockConfig mlock_config, struct ET_MmapDataLoader *out)
{
    auto mlock_config_ = static_cast<executorch::extension::MmapDataLoader::MlockConfig>(mlock_config);
    auto out_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(out);
    // return extract_result(executorch::extension::MmapDataLoader::from(file_path, mlock_config), out);
    auto res = executorch::extension::MmapDataLoader::from(file_path, mlock_config_);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    auto &loader = res.get();
    new (out_) executorch::extension::MmapDataLoader(std::move(loader));
    return ET_Error::ET_Error_Ok;
}
void executorch_MmapDataLoader_destructor(struct ET_MmapDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(self);
    self_->~MmapDataLoader();
}
struct ET_DataLoaderRefMut executorch_MmapDataLoader_as_data_loader_mut(struct ET_MmapDataLoader *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::MmapDataLoader>(self);
    auto loader = static_cast<executorch::runtime::DataLoader *>(self_);
    return cast_data_loader_mut(loader);
}
#endif

// NamedDataMap
enum ET_Error executorch_NamedDataMap_get_tensor_layout(
    struct ET_NamedDataMapRef self,
    struct ET_ArrayRefChar key,
    struct ET_TensorLayout *out)
{
    auto self_ = reinterpret_cast<const executorch::runtime::NamedDataMap *>(self.ptr);
    std::string_view key_(key.data, key.len);
    auto res = self_->get_tensor_layout(key_);
    auto out_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(out);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    memcpy(out_, &res.get(), sizeof(executorch::runtime::TensorLayout));
    return ET_Error::ET_Error_Ok;
}
enum ET_Error executorch_NamedDataMap_get_num_keys(struct ET_NamedDataMapRef self, uint32_t *out)
{
    auto self_ = reinterpret_cast<const executorch::runtime::NamedDataMap *>(self.ptr);
    return extract_result(self_->get_num_keys(), out);
}
enum ET_Error executorch_NamedDataMap_get_key(
    struct ET_NamedDataMapRef self,
    uint32_t index,
    const char **out_data)
{
    auto self_ = reinterpret_cast<const executorch::runtime::NamedDataMap *>(self.ptr);
    auto res = self_->get_key(index);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    *out_data = res.get();
    return ET_Error::ET_Error_Ok;
}

#if defined(EXECUTORCH_RS_FLAT_TENSOR)
// ET_FlatTensorDataMap
enum ET_Error executorch_FlatTensorDataMap_load(struct ET_DataLoaderRefMut loader, struct ET_FlatTensorDataMap *out)
{
    auto loader_ = cast_data_loader_mut(loader);
    auto out_ = checked_reinterpret_cast<executorch::extension::FlatTensorDataMap>(out);
    // return extract_result(executorch::extension::FlatTensorDataMap::load(loader_), out_);
    auto res = executorch::extension::FlatTensorDataMap::load(loader_);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    auto &program = res.get();
    new (out_) executorch::extension::FlatTensorDataMap(std::move(program));
    return ET_Error::ET_Error_Ok;
}
struct ET_NamedDataMapRefMut executorch_FlatTensorDataMap_as_named_data_map_mut(struct ET_FlatTensorDataMap *self)
{
    auto self_ = checked_reinterpret_cast<executorch::extension::FlatTensorDataMap>(self);
    auto named_data_map = static_cast<executorch::runtime::NamedDataMap *>(self_);
    return ET_NamedDataMapRefMut{.ptr = named_data_map};
}
#endif

// Tensor
static const executorch::aten::Tensor *cast_tensor(struct ET_TensorRef tensor)
{
    return reinterpret_cast<const executorch::aten::Tensor *>(tensor.ptr);
}
static struct ET_TensorRef cast_tensor(const executorch::aten::Tensor *tensor)
{
    return ET_TensorRef{.ptr = tensor};
}
static executorch::aten::Tensor *cast_tensor_mut(struct ET_TensorRefMut tensor)
{
    return reinterpret_cast<executorch::aten::Tensor *>(tensor.ptr);
}
// static TensorMut cast_tensor_mut(executorch::aten::Tensor *tensor)
// {
//     return reinterpret_cast<TensorMut>(tensor);
// }

bool executorch_is_valid_dim_order_and_strides(size_t dim, const ET_SizesType *sizes, const ET_DimOrderType *dim_order, const ET_StridesType *strides)
{
    ET_CHECK_MSG(dim <= MAX_DIM, "dim > 16");

    ET_StridesType computed_strides[MAX_DIM];
    auto error = executorch::runtime::dim_order_to_stride(
        sizes, dim_order, dim, &computed_strides[0]);
    if (error != executorch::runtime::Error::Ok)
        return false; // Invalid dim order

    for (size_t i = 0; i < dim; i++)
        if (computed_strides[i] != strides[i])
            return false;
    return true;
}
enum ET_Error executorch_stride_to_dim_order(const ET_StridesType *strides, size_t dims, ET_DimOrderType *dim_order)
{
    return static_cast<ET_Error>(executorch::runtime::stride_to_dim_order(strides, dims, dim_order));
}

void executorch_TensorImpl_new(
    struct ET_TensorImpl *self,
    enum ET_ScalarType type,
    size_t dim,
    ET_SizesType *sizes,
    void *data,
    ET_DimOrderType *dim_order,
    ET_StridesType *strides,
    enum ET_TensorShapeDynamism dynamism)
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
void executorch_Tensor_new(struct ET_TensorRefMut self, struct ET_TensorImpl *tensor_impl)
{
    auto self_ = cast_tensor_mut(self);
    auto tensor_impl_ = checked_reinterpret_cast<executorch::aten::TensorImpl>(tensor_impl);
    new (self_) executorch::aten::Tensor(tensor_impl_);
}
size_t executorch_Tensor_nbytes(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->nbytes();
}
size_t executorch_Tensor_size(struct ET_TensorRef self, size_t dim)
{
    auto self_ = cast_tensor(self);
    return self_->size(dim);
}
size_t executorch_Tensor_dim(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->dim();
}
size_t executorch_Tensor_numel(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->numel();
}
enum ET_ScalarType executorch_Tensor_scalar_type(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto ret = self_->scalar_type();
    return static_cast<ET_ScalarType>(ret);
}
size_t executorch_Tensor_element_size(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->element_size();
}
struct ET_ArrayRefSizesType executorch_Tensor_sizes(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto sizes = self_->sizes();
    return ET_ArrayRefSizesType{
        .data = sizes.data(),
        .len = sizes.size(),
    };
}
struct ET_ArrayRefDimOrderType executorch_Tensor_dim_order(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto dim_order = self_->dim_order();
    return ET_ArrayRefDimOrderType{
        .data = dim_order.data(),
        .len = dim_order.size(),
    };
}
struct ET_ArrayRefStridesType executorch_Tensor_strides(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    auto strides = self_->strides();
    return ET_ArrayRefStridesType{
        .data = strides.data(),
        .len = strides.size(),
    };
}
const void *executorch_Tensor_const_data_ptr(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->const_data_ptr();
}
void *executorch_Tensor_mutable_data_ptr(struct ET_TensorRef self)
{
    auto self_ = cast_tensor(self);
    return self_->mutable_data_ptr();
}

int64_t executorch_Tensor_coordinate_to_index(struct ET_TensorRef self, struct ET_ArrayRefUsizeType coordinate)
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

    for (size_t d = 0; d < ndim; d++)
    {
        if (coordinate.data[d] >= (size_t)sizes[d])
        {
            return -1;
        }
    }
    return executorch_Tensor_coordinate_to_index_unchecked(self, coordinate);
}
int64_t executorch_Tensor_coordinate_to_index_unchecked(struct ET_TensorRef self, struct ET_ArrayRefUsizeType coordinate)
{
    auto self_ = cast_tensor(self);
    auto ndim = (size_t)self_->dim();
    auto strides = self_->strides();
    size_t index = 0;
    for (size_t d = 0; d < ndim; d++)
    {
        index += coordinate.data[d] * strides[d];
    }
    return index;
}
void executorch_Tensor_destructor(struct ET_TensorRefMut self)
{
    auto self_ = cast_tensor_mut(self);
    self_->~Tensor();
}

// OptionalTensor
static const executorch::aten::optional<executorch::aten::Tensor> *cast_optional_tensor(struct ET_OptionalTensorRef tensor)
{
    return reinterpret_cast<const executorch::aten::optional<executorch::aten::Tensor> *>(tensor.ptr);
}
// static executorch::aten::optional<executorch::aten::Tensor> *cast_optional_tensor_mut(struct ET_OptionalTensorRefMut tensor)
// {
//     return reinterpret_cast<executorch::aten::optional<executorch::aten::Tensor> *>(tensor.ptr);
// }
struct ET_TensorRef executorch_OptionalTensor_get(struct ET_OptionalTensorRef self)
{
    auto self_ = cast_optional_tensor(self);
    if (!self_->has_value())
        return ET_TensorRef{.ptr = nullptr};
    const executorch::aten::Tensor *tensor = &self_->value();
    return cast_tensor(tensor);
}

// ET_TensorLayout
// enum ET_Error executorch_TensorLayout_create(
//     struct ET_ArrayRefI32 sizes,
//     struct ET_ArrayRefU8 dim_order,
//     enum ET_ScalarType scalar_type,
//     struct ET_TensorLayout *out)
// {
//     auto sizes_ = *checked_reinterpret_cast<executorch::runtime::Span<const int32_t>>(&sizes);
//     auto dim_order_ = *checked_reinterpret_cast<executorch::runtime::Span<const uint8_t>>(&dim_order);
//     auto out_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(out);
//     auto scalar_type_ = static_cast<executorch::aten::ScalarType>(scalar_type);
//     auto res = executorch::runtime::TensorLayout::create(
//         sizes_,
//         dim_order_,
//         scalar_type_);
//     if (!res.ok())
//         return static_cast<ET_Error>(res.error());
//     auto &layout = res.get();
//     new (out_) executorch::runtime::TensorLayout(std::move(layout));
//     return ET_Error::ET_Error_Ok;
// }
struct ET_ArrayRefI32 executorch_TensorLayout_sizes(const struct ET_TensorLayout *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(self);
    auto sizes = self_->sizes();
    return ET_ArrayRefI32{
        .data = sizes.data(),
        .len = sizes.size(),
    };
}
struct ET_ArrayRefU8 executorch_TensorLayout_dim_order(const struct ET_TensorLayout *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(self);
    auto dim_order = self_->dim_order();
    return ET_ArrayRefU8{
        .data = dim_order.data(),
        .len = dim_order.size(),
    };
}
enum ET_ScalarType executorch_TensorLayout_scalar_type(const struct ET_TensorLayout *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(self);
    auto scalar_type = self_->scalar_type();
    return static_cast<ET_ScalarType>(scalar_type);
}
size_t executorch_TensorLayout_nbytes(const struct ET_TensorLayout *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorLayout>(self);
    return self_->nbytes();
}

// EValue
static const executorch::runtime::EValue *cast_evalue(struct ET_EValueRef evalue)
{
    return reinterpret_cast<const executorch::runtime::EValue *>(evalue.ptr);
}
static struct ET_EValueRef cast_evalue(const executorch::runtime::EValue *evalue)
{
    return ET_EValueRef{.ptr = evalue};
}
static executorch::runtime::EValue *cast_evalue_mut(struct ET_EValueRefMut evalue)
{
    return reinterpret_cast<executorch::runtime::EValue *>(evalue.ptr);
}
// static EValueMut cast_evalue_mut(executorch::runtime::EValue *evalue)
// {
//     return evalue;
// }
void executorch_EValue_new_none(struct ET_EValueRefMut self)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue();
}
void executorch_EValue_new_from_i64(struct ET_EValueRefMut self, int64_t value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_i64_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListI64 *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::runtime::BoxedEvalueList<int64_t>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::runtime::BoxedEvalueList<int64_t> *>(value_));
}
void executorch_EValue_new_from_f64(struct ET_EValueRefMut self, double value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_f64_list(struct ET_EValueRefMut self, const struct ET_ArrayRefF64 *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::aten::ArrayRef<double>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::aten::ArrayRef<double> *>(value_));
}
void executorch_EValue_new_from_bool(struct ET_EValueRefMut self, bool value)
{
    auto self_ = cast_evalue_mut(self);
    new (self_) executorch::runtime::EValue(value);
}
void executorch_EValue_new_from_bool_list(struct ET_EValueRefMut self, const struct ET_ArrayRefBool *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::aten::ArrayRef<bool>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::aten::ArrayRef<bool> *>(value_));
}
void executorch_EValue_new_from_string(struct ET_EValueRefMut self, const struct ET_ArrayRefChar *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::aten::ArrayRef<char>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::aten::ArrayRef<char> *>(value_));
}
void executorch_EValue_new_from_tensor(struct ET_EValueRefMut self, struct ET_TensorRef value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = cast_tensor(value);
    new (self_) executorch::runtime::EValue(*value_);
}
void executorch_EValue_new_from_tensor_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListTensor *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::runtime::BoxedEvalueList<executorch::aten::Tensor>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::runtime::BoxedEvalueList<executorch::aten::Tensor> *>(value_));
}
void executorch_EValue_new_from_optional_tensor_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListOptionalTensor *value)
{
    auto self_ = cast_evalue_mut(self);
    auto value_ = checked_reinterpret_cast<executorch::runtime::BoxedEvalueList<std::optional<executorch::aten::Tensor>>>(value);
    new (self_) executorch::runtime::EValue(const_cast<executorch::runtime::BoxedEvalueList<std::optional<executorch::aten::Tensor>> *>(value_));
}
enum ET_Tag executorch_EValue_tag(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    return static_cast<ET_Tag>(self_->tag);
}
int64_t executorch_EValue_as_i64(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toInt();
}
struct ET_ArrayRefI64 executorch_EValue_as_i64_list(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toIntList();
    return ET_ArrayRefI64{
        .data = list.data(),
        .len = list.size(),
    };
}
double executorch_EValue_as_f64(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toDouble();
}
struct ET_ArrayRefF64 executorch_EValue_as_f64_list(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toDoubleList();
    return ET_ArrayRefF64{
        .data = list.data(),
        .len = list.size(),
    };
}
bool executorch_EValue_as_bool(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    return self_->toBool();
}
struct ET_ArrayRefBool executorch_EValue_as_bool_list(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toBoolList();
    return ET_ArrayRefBool{
        .data = list.data(),
        .len = list.size(),
    };
}
struct ET_ArrayRefChar executorch_EValue_as_string(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto str = self_->toString();
    return ET_ArrayRefChar{
        .data = str.data(),
        .len = str.size(),
    };
}
struct ET_TensorRef executorch_EValue_as_tensor(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    const executorch::aten::Tensor *tensor = &self_->toTensor();
    return cast_tensor(tensor);
}
struct ET_ArrayRefTensor executorch_EValue_as_tensor_list(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toTensorList();
    return ET_ArrayRefTensor{
        .data = cast_tensor(list.data()),
        .len = list.size(),
    };
}
struct ET_ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(struct ET_EValueRef self)
{
    auto self_ = cast_evalue(self);
    auto list = self_->toListOptionalTensor();
    return ET_ArrayRefOptionalTensor{
        .data = ET_OptionalTensorRef{.ptr = checked_reinterpret_cast<ET_OptionalTensorStorage>(list.data())},
        .len = list.size(),
    };
}
void executorch_EValue_copy(struct ET_EValueRef src, struct ET_EValueRefMut dst)
{
    auto src_ = cast_evalue(src);
    auto dst_ = cast_evalue_mut(dst);
    new (dst_) executorch::runtime::EValue(*src_);
}
void executorch_EValue_destructor(struct ET_EValueRefMut self)
{
    auto self_ = cast_evalue_mut(self);
    self_->~EValue();
}
void executorch_EValue_move(struct ET_EValueRefMut src, struct ET_EValueRefMut dst)
{
    auto src_ = cast_evalue_mut(src);
    auto dst_ = cast_evalue_mut(dst);
    new (dst_) executorch::runtime::EValue(std::move(*src_));
}

// ET_Program
enum ET_ProgramHeaderStatus executorch_Program_check_header(const void *data, size_t size)
{
    auto status = executorch::runtime::Program::check_header(data, size);
    return static_cast<ET_ProgramHeaderStatus>(status);
}
enum ET_Error executorch_Program_load(struct ET_DataLoaderRefMut loader, enum ET_ProgramVerification verification, struct ET_Program *out)
{
    auto loader_ = cast_data_loader_mut(loader);
    auto verification_ = static_cast<executorch::runtime::Program::Verification>(verification);
    auto out_ = checked_reinterpret_cast<executorch::runtime::Program>(out);
    // return extract_result(executorch::runtime::Program::load(loader, verification), out);
    auto res = executorch::runtime::Program::load(loader_, verification_);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    auto &program = res.get();
    new (out_) executorch::runtime::Program(std::move(program));
    return ET_Error::ET_Error_Ok;
}
enum ET_Error executorch_Program_load_method(
    const struct ET_Program *self,
    const char *method_name,
    struct ET_MemoryManager *memory_manager,
    struct ET_EventTracerRefMut event_tracer,
    struct ET_NamedDataMapRef named_data_map,
    struct ET_Method *out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    auto memory_manager_ = checked_reinterpret_cast<executorch::runtime::MemoryManager>(memory_manager);
    auto event_tracer_ = reinterpret_cast<executorch::runtime::EventTracer *>(event_tracer.ptr);
    auto named_data_map_ = reinterpret_cast<const executorch::runtime::NamedDataMap *>(named_data_map.ptr);
    auto out_ = checked_reinterpret_cast<executorch::runtime::Method>(out);

    auto res = self_->load_method(method_name, memory_manager_, event_tracer_, named_data_map_);
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    auto &method = res.get();
    new (out_) executorch::runtime::Method(std::move(method));
    return ET_Error::ET_Error_Ok;
}
enum ET_Error executorch_Program_get_method_name(const struct ET_Program *self, size_t method_index, const char **out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    return extract_result(self_->get_method_name(method_index), out);
}
enum ET_Error executorch_Program_get_named_data_map(const struct ET_Program *self, struct ET_NamedDataMapRef *out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    auto res = self_->get_named_data_map();
    if (!res.ok())
        return static_cast<ET_Error>(res.error());
    *out = ET_NamedDataMapRef{.ptr = res.get()};
    return ET_Error::ET_Error_Ok;
}
enum ET_Error executorch_Program_method_meta(const struct ET_Program *self, const char *method_name, struct ET_MethodMeta *method_meta_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    auto method_meta_out_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(method_meta_out);
    return extract_result(self_->method_meta(method_name), method_meta_out_);
}
size_t executorch_Program_num_methods(const struct ET_Program *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    return self_->num_methods();
}
void executorch_Program_destructor(struct ET_Program *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Program>(self);
    self_->~Program();
}

// ET_MethodMeta
size_t executorch_Method_inputs_size(const struct ET_Method *self)
{
    auto *self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    return self_->inputs_size();
}
size_t executorch_Method_outputs_size(const struct ET_Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    return self_->outputs_size();
}
enum ET_Error executorch_Method_set_input(struct ET_Method *self, struct ET_EValueRef input_evalue, size_t input_idx)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    auto input_evalue_ = cast_evalue(input_evalue);
    executorch::runtime::Error ret = self_->set_input(*input_evalue_, input_idx);
    return static_cast<ET_Error>(ret);
}
struct ET_EValueRef executorch_Method_get_output(const struct ET_Method *self, size_t i)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    const executorch::runtime::EValue *output = &self_->get_output(i);
    return cast_evalue(output);
}
enum ET_Error executorch_Method_get_attribute(struct ET_Method *self, struct ET_ArrayRefChar name, struct ET_TensorRefMut out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    auto out_ = cast_tensor_mut(out);
    std::string_view name_(name.data, name.len);
    return extract_result(self_->get_attribute(name_), out_);
}
enum ET_Error executorch_Method_execute(struct ET_Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    executorch::runtime::Error ret = self_->execute();
    return static_cast<ET_Error>(ret);
}
void executorch_Method_destructor(struct ET_Method *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::Method>(self);
    self_->~Method();
}
const char *executorch_MethodMeta_name(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->name();
    self_->num_inputs();
}
size_t executorch_MethodMeta_num_inputs(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_inputs();
}
size_t executorch_MethodMeta_num_outputs(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_outputs();
}
size_t executorch_MethodMeta_num_memory_planned_buffers(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_memory_planned_buffers();
}
enum ET_Error executorch_MethodMeta_input_tag(const struct ET_MethodMeta *self, size_t index, enum ET_Tag *tag_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tag_out_ = checked_reinterpret_cast<executorch::runtime::Tag>(tag_out);
    return extract_result(self_->input_tag(index), tag_out_);
}
enum ET_Error executorch_MethodMeta_output_tag(const struct ET_MethodMeta *self, size_t index, enum ET_Tag *tag_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tag_out_ = checked_reinterpret_cast<executorch::runtime::Tag>(tag_out);
    return extract_result(self_->output_tag(index), tag_out_);
}
enum ET_Error executorch_MethodMeta_input_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tensor_info_out_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(tensor_info_out);
    return extract_result(self_->input_tensor_meta(index), tensor_info_out_);
}
enum ET_Error executorch_MethodMeta_output_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tensor_info_out_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(tensor_info_out);
    return extract_result(self_->output_tensor_meta(index), tensor_info_out_);
}
size_t executorch_MethodMeta_num_attributes(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_attributes();
}
enum ET_Error executorch_MethodMeta_attribute_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    auto tensor_info_out_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(tensor_info_out);
    return extract_result(self_->attribute_tensor_meta(index), tensor_info_out_);
}
enum ET_Error executorch_MethodMeta_memory_planned_buffer_size(const struct ET_MethodMeta *self, size_t index, int64_t *size_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return extract_result(self_->memory_planned_buffer_size(index), size_out);
}
bool executorch_MethodMeta_uses_backend(const struct ET_MethodMeta *self, const char *backend_name)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->uses_backend(backend_name);
}
size_t executorch_MethodMeta_num_backends(const struct ET_MethodMeta *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return self_->num_backends();
}
enum ET_Error executorch_MethodMeta_get_backend_name(const struct ET_MethodMeta *self, size_t index, const char **backend_name_out)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(self);
    return extract_result(self_->get_backend_name(index), backend_name_out);
}

// ET_TensorInfo
struct ET_ArrayRefI32 executorch_TensorInfo_sizes(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto sizes = self_->sizes();
    return ET_ArrayRefI32{
        .data = sizes.data(),
        .len = sizes.size(),
    };
}
struct ET_ArrayRefU8 executorch_TensorInfo_dim_order(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto dim_order = self_->dim_order();
    return ET_ArrayRefU8{
        .data = dim_order.data(),
        .len = dim_order.size(),
    };
}
enum ET_ScalarType executorch_TensorInfo_scalar_type(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto ret = self_->scalar_type();
    return static_cast<ET_ScalarType>(ret);
}
bool executorch_TensorInfo_is_memory_planned(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    return self_->is_memory_planned();
}
size_t executorch_TensorInfo_nbytes(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    return self_->nbytes();
}
struct ET_ArrayRefChar executorch_TensorInfo_name(const struct ET_TensorInfo *self)
{
    auto self_ = checked_reinterpret_cast<executorch::runtime::TensorInfo>(self);
    auto name = self_->name();
    return ET_ArrayRefChar{
        .data = name.data(),
        .len = name.size(),
    };
}

#if defined(EXECUTORCH_RS_ETDUMP)
// ET_DumpGen
struct ET_DumpGen executorch_ETDumpGen_new(struct ET_SpanU8 buffer)
{
    struct ET_DumpGen self;
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(&self);
    new (self_) executorch::etdump::ETDumpGen({buffer.data, buffer.len});
    return self;
}
struct ET_ArrayRefU8 executorch_ETDumpGen_get_etdump_data(struct ET_DumpGen *self)
{
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(self);
    auto res = self_->get_etdump_data();
    return ET_ArrayRefU8{.data = (uint8_t *)res.buf, .len = res.size};
}
struct ET_EventTracerRefMut executorch_ETDumpGen_as_event_tracer_mut(struct ET_DumpGen *self)
{
    auto self_ = checked_reinterpret_cast<executorch::etdump::ETDumpGen>(self);
    auto tracer = static_cast<executorch::runtime::EventTracer *>(self_);
    return ET_EventTracerRefMut{.ptr = tracer};
}
#endif

// Platform structs and functions

bool executorch_register_pal(ExecutorchPalImpl impl)
{
    executorch::runtime::PalImpl pal_impl = executorch::runtime::PalImpl::create(
        impl.init,
        impl.abort,
        impl.current_ticks,
        reinterpret_cast<pal_ticks_to_ns_multiplier_method>(impl.ticks_to_ns_multiplier),
        reinterpret_cast<pal_emit_log_message_method>(impl.emit_log_message),
        impl.allocate,
        impl.free,
        impl.source_filename);
    return executorch::runtime::register_pal(pal_impl);
}

const ExecutorchPalImpl *executorch_get_pal_impl()
{
    auto impl = executorch::runtime::get_pal_impl();
    return impl ? checked_reinterpret_cast<ExecutorchPalImpl>(impl) : nullptr;
}

void executorch_pal_init()
{
    executorch::runtime::pal_init();
}

void executorch_pal_abort()
{
    executorch::runtime::pal_abort();
}

executorch_timestamp_t executorch_pal_current_ticks()
{
    return executorch::runtime::pal_current_ticks();
}

struct executorch_tick_ratio executorch_pal_ticks_to_ns_multiplier()
{
    auto ratio = executorch::runtime::pal_ticks_to_ns_multiplier();
    return executorch_tick_ratio{
        .numerator = ratio.numerator,
        .denominator = ratio.denominator,
    };
}

void executorch_pal_emit_log_message(
    executorch_timestamp_t timestamp,
    enum executorch_pal_log_level level,
    const char *filename,
    const char *function,
    size_t line,
    const char *message,
    size_t length)
{
    executorch::runtime::pal_emit_log_message(
        timestamp,
        static_cast<et_pal_log_level_t>(level),
        filename,
        function,
        line,
        message,
        length);
}

void *executorch_pal_allocate(size_t size)
{
    return executorch::runtime::pal_allocate(size);
}

void executorch_pal_free(void *ptr)
{
    executorch::runtime::pal_free(ptr);
}
