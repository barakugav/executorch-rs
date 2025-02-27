
#include <cstddef>
#include <vector>
#include "executorch_rs/bridge.hpp"
#include "executorch/runtime/core/exec_aten/util/tensor_util.h"
#include "executorch/runtime/core/exec_aten/util/dim_order_util.h"
#include "executorch/runtime/platform/assert.h"

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_STD)
    void executorch_VecChar_destructor(VecChar *vec)
    {
        delete[] vec->data;
    }
    void executorch_VecVecChar_destructor(VecVecChar *vec)
    {
        for (size_t i = 0; i < vec->len; i++)
            executorch_VecChar_destructor(&vec->data[i]);
        delete[] vec->data;
    }
    void executorch_VecEValue_destructor(VecEValue *vec)
    {
        // Its safe to call the destructor of elements in `vec->data[len..cap]` because we created them with `new T[len]`
        // aka default constructor
        delete[] vec->data;
    }
#endif

    template <typename T>
    static executorch::runtime::Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
    {
        if (result.ok())
            *output = std::move(result.get());
        return result.error();
    }

    size_t executorch_Method_inputs_size(const Method *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Method *>(self);
        return self_->inputs_size();
    }
    size_t executorch_Method_outputs_size(const Method *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Method *>(self);
        return self_->outputs_size();
    }
    executorch::runtime::Error executorch_Method_set_input(Method *self, const EValue *input_evalue, size_t input_idx)
    {
        auto self_ = reinterpret_cast<executorch::runtime::Method *>(self);
        auto input_evalue_ = reinterpret_cast<const executorch::runtime::EValue *>(input_evalue);
        return self_->set_input(*input_evalue_, input_idx);
    }
    const EValue *executorch_Method_get_output(const Method *self, size_t i)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Method *>(self);
        const executorch::runtime::EValue *output = &self_->get_output(i);
        return reinterpret_cast<const EValue *>(output);
    }
    executorch::runtime::Error executorch_Method_execute(Method *self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::Method *>(self);
        return self_->execute();
    }
    void executorch_Method_destructor(Method *self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::Method *>(self);
        self_->~Method();
    }
    const char *executorch_MethodMeta_name(const MethodMeta *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return self_->name();
        self_->num_inputs();
    }
    size_t executorch_MethodMeta_num_inputs(const MethodMeta *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return self_->num_inputs();
    }
    size_t executorch_MethodMeta_num_outputs(const MethodMeta *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return self_->num_outputs();
    }
    size_t executorch_MethodMeta_num_memory_planned_buffers(const MethodMeta *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return self_->num_memory_planned_buffers();
    }
    executorch::runtime::Error executorch_MethodMeta_input_tag(const MethodMeta *self, size_t index, executorch::runtime::Tag *tag_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return extract_result(self_->input_tag(index), tag_out);
    }
    executorch::runtime::Error executorch_MethodMeta_output_tag(const MethodMeta *self, size_t index, executorch::runtime::Tag *tag_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return extract_result(self_->output_tag(index), tag_out);
    }
    executorch::runtime::Error executorch_MethodMeta_input_tensor_meta(const MethodMeta *self, size_t index, TensorInfo *tensor_info_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        auto tensor_info_out_ = reinterpret_cast<executorch::runtime::TensorInfo *>(tensor_info_out);
        return extract_result(self_->input_tensor_meta(index), tensor_info_out_);
    }
    executorch::runtime::Error executorch_MethodMeta_output_tensor_meta(const MethodMeta *self, size_t index, TensorInfo *tensor_info_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        auto tensor_info_out_ = reinterpret_cast<executorch::runtime::TensorInfo *>(tensor_info_out);
        return extract_result(self_->output_tensor_meta(index), tensor_info_out_);
    }
    executorch::runtime::Error executorch_MethodMeta_memory_planned_buffer_size(const MethodMeta *self, size_t index, int64_t *size_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::MethodMeta *>(self);
        return extract_result(self_->memory_planned_buffer_size(index), size_out);
    }

    executorch_rs::MemoryAllocator executorch_MemoryAllocator_new(uint32_t size, uint8_t *base_address)
    {
        executorch_rs::MemoryAllocator self;
        auto self_ = reinterpret_cast<executorch::runtime::MemoryAllocator *>(&self);
        new (self_) executorch::runtime::MemoryAllocator(size, base_address);
        return self;
    }
    void *MemoryAllocator_allocate(executorch_rs::MemoryAllocator &self, size_t size, size_t alignment)
    {
        auto self_ = reinterpret_cast<executorch::runtime::MemoryAllocator *>(&self);
        return self_->allocate(size, alignment);
    }
#if defined(EXECUTORCH_RS_STD)

    executorch_rs::MallocMemoryAllocator executorch_MallocMemoryAllocator_new()
    {
        executorch_rs::MallocMemoryAllocator self;
        auto self_ = reinterpret_cast<executorch::extension::MallocMemoryAllocator *>(&self);
        new (self_) executorch::extension::MallocMemoryAllocator();
        return self;
    }
    void executorch_MallocMemoryAllocator_destructor(executorch_rs::MallocMemoryAllocator &self)
    {
        auto self_ = reinterpret_cast<executorch::extension::MallocMemoryAllocator *>(&self);
        self_->~MallocMemoryAllocator();
    }
    const executorch_rs::MemoryAllocator *executorch_MallocMemoryAllocator_as_memory_allocator(const executorch_rs::MallocMemoryAllocator *self)
    {
        auto self_ = reinterpret_cast<const executorch::extension::MallocMemoryAllocator *>(self);
        auto memory_allocator = static_cast<const executorch::runtime::MemoryAllocator *>(self_);
        return reinterpret_cast<const executorch_rs::MemoryAllocator *>(memory_allocator);
    }
#endif
    executorch_rs::HierarchicalAllocator executorch_HierarchicalAllocator_new(SpanSpanU8 buffers)
    {
        executorch::runtime::Span<executorch::runtime::Span<uint8_t>> buffers_ = *reinterpret_cast<executorch::runtime::Span<executorch::runtime::Span<uint8_t>> *>(&buffers);
        ET_CHECK((void *)buffers_.begin() == (void *)buffers.data);
        ET_CHECK(buffers_.size() == buffers.len);
        executorch_rs::HierarchicalAllocator self;
        auto self_ = reinterpret_cast<executorch::runtime::HierarchicalAllocator *>(&self);
        new (self_) executorch::runtime::HierarchicalAllocator(buffers_);
        return self;
    }
    void executorch_HierarchicalAllocator_destructor(executorch_rs::HierarchicalAllocator &self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::HierarchicalAllocator *>(&self);
        self_->~HierarchicalAllocator();
    }
    executorch_rs::MemoryManager executorch_MemoryManager_new(
        executorch_rs::MemoryAllocator *method_allocator,
        executorch_rs::HierarchicalAllocator *planned_memory,
        executorch_rs::MemoryAllocator *temp_allocator)
    {
        auto method_allocator_ = reinterpret_cast<executorch::runtime::MemoryAllocator *>(method_allocator);
        auto planned_memory_ = reinterpret_cast<executorch::runtime::HierarchicalAllocator *>(planned_memory);
        auto temp_allocator_ = reinterpret_cast<executorch::runtime::MemoryAllocator *>(temp_allocator);

        executorch_rs::MemoryManager self;
        auto self_ = reinterpret_cast<executorch::runtime::MemoryManager *>(&self);
        new (self_) executorch::runtime::MemoryManager(method_allocator_, planned_memory_, temp_allocator_);
        return self;
    }

    // Loaders
    executorch_rs::BufferDataLoader executorch_BufferDataLoader_new(const void *data, size_t size)
    {
        executorch_rs::BufferDataLoader loader;
        auto loader_ = reinterpret_cast<executorch::extension::BufferDataLoader *>(&loader);
        new (loader_) executorch::extension::BufferDataLoader(data, size);
        return loader;
    }
    const executorch_rs::DataLoader *executorch_BufferDataLoader_as_data_loader(const executorch_rs::BufferDataLoader *self)
    {
        auto self_ = reinterpret_cast<const executorch::extension::BufferDataLoader *>(self);
        auto loader = static_cast<const executorch::runtime::DataLoader *>(self_);
        return reinterpret_cast<const executorch_rs::DataLoader *>(loader);
    }
#if defined(EXECUTORCH_RS_DATA_LOADER)
    executorch::runtime::Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, executorch_rs::FileDataLoader *out)
    {
        auto out_ = reinterpret_cast<executorch::extension::FileDataLoader *>(out);
        // return extract_result(std::move(executorch::extension::FileDataLoader::from(file_path, alignment)), out);
        auto res = executorch::extension::FileDataLoader::from(file_path, alignment);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out_) executorch::extension::FileDataLoader(std::move(loader));
        return executorch::runtime::Error::Ok;
    }
    void executorch_FileDataLoader_destructor(executorch_rs::FileDataLoader *self)
    {
        auto self_ = reinterpret_cast<executorch::extension::FileDataLoader *>(self);
        self_->~FileDataLoader();
    }
    const executorch_rs::DataLoader *executorch_FileDataLoader_as_data_loader(const executorch_rs::FileDataLoader *self)
    {
        auto self_ = reinterpret_cast<const executorch::extension::FileDataLoader *>(self);
        auto loader = static_cast<const executorch::runtime::DataLoader *>(self_);
        return reinterpret_cast<const executorch_rs::DataLoader *>(loader);
    }
    executorch::runtime::Error executorch_MmapDataLoader_new(const char *file_path, executorch::extension::MmapDataLoader::MlockConfig mlock_config, executorch_rs::MmapDataLoader *out)
    {
        auto out_ = reinterpret_cast<executorch::extension::MmapDataLoader *>(out);
        // return extract_result(executorch::extension::MmapDataLoader::from(file_path, mlock_config), out);
        auto res = executorch::extension::MmapDataLoader::from(file_path, mlock_config);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out_) executorch::extension::MmapDataLoader(std::move(loader));
        return executorch::runtime::Error::Ok;
    }
    void executorch_MmapDataLoader_destructor(executorch_rs::MmapDataLoader *self)
    {
        auto self_ = reinterpret_cast<executorch::extension::MmapDataLoader *>(self);
        self_->~MmapDataLoader();
    }
    const executorch_rs::DataLoader *executorch_MmapDataLoader_as_data_loader(const executorch_rs::MmapDataLoader *self)
    {
        auto self_ = reinterpret_cast<const executorch::extension::MmapDataLoader *>(self);
        auto loader = static_cast<const executorch::runtime::DataLoader *>(self_);
        return reinterpret_cast<const executorch_rs::DataLoader *>(loader);
    }
#endif

    // Program
    executorch::runtime::Program::HeaderStatus executorch_Program_check_header(const void *data, size_t size)
    {
        return executorch::runtime::Program::check_header(data, size);
    }
    executorch::runtime::Error executorch_Program_load(executorch_rs::DataLoader *loader, executorch::runtime::Program::Verification verification, Program *out)
    {
        auto loader_ = reinterpret_cast<executorch::runtime::DataLoader *>(loader);
        auto out_ = reinterpret_cast<executorch::runtime::Program *>(out);
        // return extract_result(executorch::runtime::Program::load(loader, verification), out);
        auto res = executorch::runtime::Program::load(loader_, verification);
        if (!res.ok())
            return res.error();
        auto &program = res.get();
        new (out_) executorch::runtime::Program(std::move(program));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error executorch_Program_load_method(const Program *self, const char *method_name, executorch_rs::MemoryManager *memory_manager, executorch::runtime::EventTracer *event_tracer, Method *out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Program *>(self);
        auto memory_manager_ = reinterpret_cast<executorch::runtime::MemoryManager *>(memory_manager);
        auto out_ = reinterpret_cast<executorch::runtime::Method *>(out);
        // return extract_result(std::move(self.load_method(method_name, memory_manager, event_tracer)), out);
        auto res = self_->load_method(method_name, memory_manager_, event_tracer);
        if (!res.ok())
            return res.error();
        auto &method = res.get();
        new (out_) executorch::runtime::Method(std::move(method));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error executorch_Program_get_method_name(const Program *self, size_t method_index, const char **out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Program *>(self);
        return extract_result(self_->get_method_name(method_index), out);
    }
    executorch::runtime::Error executorch_Program_method_meta(const Program *self, const char *method_name, MethodMeta *method_meta_out)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Program *>(self);
        auto method_meta_out_ = reinterpret_cast<executorch::runtime::MethodMeta *>(method_meta_out);
        return extract_result(self_->method_meta(method_name), method_meta_out_);
    }
    size_t executorch_Program_num_methods(const Program *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::Program *>(self);
        return self_->num_methods();
    }
    void executorch_Program_destructor(Program *self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::Program *>(self);
        self_->~Program();
    }

    // TensorInfo
    ArrayRefI32 executorch_TensorInfo_sizes(const TensorInfo *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::TensorInfo *>(self);
        auto sizes = self_->sizes();
        return ArrayRefI32{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefU8 executorch_TensorInfo_dim_order(const TensorInfo *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::TensorInfo *>(self);
        auto dim_order = self_->dim_order();
        return ArrayRefU8{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }
    executorch::aten::ScalarType executorch_TensorInfo_scalar_type(const TensorInfo *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::TensorInfo *>(self);
        return self_->scalar_type();
    }
    size_t executorch_TensorInfo_nbytes(const TensorInfo *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::TensorInfo *>(self);
        return self_->nbytes();
    }

    // Tensor
    void executorch_TensorImpl_new(
        TensorImpl *self,
        executorch::aten::ScalarType type,
        ssize_t dim,
        executorch::aten::SizesType *sizes,
        void *data,
        executorch::aten::DimOrderType *dim_order,
        executorch::aten::StridesType *strides,
        executorch::aten::TensorShapeDynamism dynamism)
    {
        auto self_ = reinterpret_cast<executorch::aten::TensorImpl *>(self);
        new (self_) executorch::aten::TensorImpl(
            type,
            dim,
            sizes,
            data,
            dim_order,
            strides,
            dynamism);
    }
    void executorch_Tensor_new(Tensor *self, TensorImpl *tensor_impl)
    {
        auto self_ = reinterpret_cast<executorch::aten::Tensor *>(self);
        auto tensor_impl_ = reinterpret_cast<executorch::aten::TensorImpl *>(tensor_impl);
        new (self_) executorch::aten::Tensor(tensor_impl_);
    }
    size_t executorch_Tensor_nbytes(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->nbytes();
    }
    ssize_t executorch_Tensor_size(const Tensor *self, ssize_t dim)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->size(dim);
    }
    ssize_t executorch_Tensor_dim(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->dim();
    }
    ssize_t executorch_Tensor_numel(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->numel();
    }
    executorch::aten::ScalarType executorch_Tensor_scalar_type(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->scalar_type();
    }
    ssize_t executorch_Tensor_element_size(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->element_size();
    }
    ArrayRefSizesType executorch_Tensor_sizes(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        auto sizes = self_->sizes();
        return ArrayRefSizesType{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefDimOrderType executorch_Tensor_dim_order(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        auto dim_order = self_->dim_order();
        return ArrayRefDimOrderType{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }
    ArrayRefStridesType executorch_Tensor_strides(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        auto strides = self_->strides();
        return ArrayRefStridesType{
            .data = strides.data(),
            .len = strides.size(),
        };
    }
    const void *executorch_Tensor_const_data_ptr(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->const_data_ptr();
    }
    void *executorch_Tensor_mutable_data_ptr(const Tensor *self)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
        return self_->mutable_data_ptr();
    }

    ssize_t executorch_Tensor_coordinate_to_index(const Tensor *self, ArrayRefUsizeType coordinate)
    {
        auto self_ = reinterpret_cast<const executorch::aten::Tensor *>(self);
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
    void executorch_Tensor_destructor(Tensor *self)
    {
        auto self_ = reinterpret_cast<executorch::aten::Tensor *>(self);
        self_->~Tensor();
    }

    void executorch_EValue_new_none(EValue *self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        new (self_) executorch::runtime::EValue();
    }
    void executorch_EValue_new_from_i64(EValue *self, int64_t value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        new (self_) executorch::runtime::EValue(value);
    }
    void executorch_EValue_new_from_i64_list(EValue *self, BoxedEvalueListI64 value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
        executorch::runtime::BoxedEvalueList<int64_t> list(
            reinterpret_cast<executorch::runtime::EValue **>(const_cast<EValue **>(value.wrapped_vals.data)),
            value.unwrapped_vals.data,
            (int)value.wrapped_vals.len);
        new (self_) executorch::runtime::EValue(list);
    }
    void executorch_EValue_new_from_f64(EValue *self, double value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        new (self_) executorch::runtime::EValue(value);
    }
    void executorch_EValue_new_from_f64_list(EValue *self, ArrayRefF64 value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        executorch::aten::ArrayRef<double> value_(value.data, value.len);
        new (self_) executorch::runtime::EValue(value_);
    }
    void executorch_EValue_new_from_bool(EValue *self, bool value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        new (self_) executorch::runtime::EValue(value);
    }
    void executorch_EValue_new_from_bool_list(EValue *self, ArrayRefBool value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        executorch::aten::ArrayRef<bool> value_(value.data, value.len);
        new (self_) executorch::runtime::EValue(value_);
    }
    void executorch_EValue_new_from_string(EValue *self, ArrayRefChar value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        new (self_) executorch::runtime::EValue(value.data, value.len);
    }
    void executorch_EValue_new_from_tensor(EValue *self, const Tensor *value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        auto value_ = reinterpret_cast<const executorch::aten::Tensor *>(value);
        new (self_) executorch::runtime::EValue(*value_);
    }
    void executorch_EValue_new_from_tensor_list(EValue *self, BoxedEvalueListTensor value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
        executorch::runtime::BoxedEvalueList<executorch::aten::Tensor> list(
            reinterpret_cast<executorch::runtime::EValue **>(const_cast<EValue **>(value.wrapped_vals.data)),
            reinterpret_cast<executorch::aten::Tensor *>(value.unwrapped_vals.data),
            (int)value.wrapped_vals.len);
        new (self_) executorch::runtime::EValue(list);
    }
    void executorch_EValue_new_from_optional_tensor_list(EValue *self, BoxedEvalueListOptionalTensor value)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        ET_CHECK(value.wrapped_vals.len == value.unwrapped_vals.len);
        auto unwrapped_vals = reinterpret_cast<executorch::aten::optional<executorch::aten::Tensor> *>(value.unwrapped_vals.data);
        executorch::runtime::BoxedEvalueList<executorch::aten::optional<executorch::aten::Tensor>> list(
            reinterpret_cast<executorch::runtime::EValue **>(const_cast<EValue **>(value.wrapped_vals.data)),
            unwrapped_vals,
            (int)value.wrapped_vals.len);
        new (self_) executorch::runtime::EValue(list);
    }
    executorch::runtime::Tag executorch_EValue_tag(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        return self_->tag;
    }
    int64_t executorch_EValue_as_i64(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        return self_->toInt();
    }
    ArrayRefI64 executorch_EValue_as_i64_list(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto list = self_->toIntList();
        return ArrayRefI64{
            .data = list.data(),
            .len = list.size(),
        };
    }
    double executorch_EValue_as_f64(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        return self_->toDouble();
    }
    ArrayRefF64 executorch_EValue_as_f64_list(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto list = self_->toDoubleList();
        return ArrayRefF64{
            .data = list.data(),
            .len = list.size(),
        };
    }
    bool executorch_EValue_as_bool(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        return self_->toBool();
    }
    ArrayRefBool executorch_EValue_as_bool_list(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto list = self_->toBoolList();
        return ArrayRefBool{
            .data = list.data(),
            .len = list.size(),
        };
    }
    ArrayRefChar executorch_EValue_as_string(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto str = self_->toString();
        return ArrayRefChar{
            .data = str.data(),
            .len = str.size(),
        };
    }
    const Tensor *executorch_EValue_as_tensor(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        const executorch::aten::Tensor *tensor = &self_->toTensor();
        return reinterpret_cast<const Tensor *>(tensor);
    }
    ArrayRefTensor executorch_EValue_as_tensor_list(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto list = self_->toTensorList();
        return ArrayRefTensor{
            .data = reinterpret_cast<const Tensor *>(list.data()),
            .len = list.size(),
        };
    }
    ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(const EValue *self)
    {
        auto self_ = reinterpret_cast<const executorch::runtime::EValue *>(self);
        auto list = self_->toListOptionalTensor();
        return ArrayRefOptionalTensor{
            .data = reinterpret_cast<const OptionalTensor *>(list.data()),
            .len = list.size(),
        };
    }
    void executorch_EValue_copy(const EValue *src, EValue *dst)
    {
        auto src_ = reinterpret_cast<const executorch::runtime::EValue *>(src);
        auto dst_ = reinterpret_cast<executorch::runtime::EValue *>(dst);
        new (dst_) executorch::runtime::EValue(*src_);
    }
    void executorch_EValue_destructor(EValue *self)
    {
        auto self_ = reinterpret_cast<executorch::runtime::EValue *>(self);
        self_->~EValue();
    }
    void executorch_EValue_move(EValue *src, EValue *dst)
    {
        auto src_ = reinterpret_cast<executorch::runtime::EValue *>(src);
        auto dst_ = reinterpret_cast<executorch::runtime::EValue *>(dst);
        new (dst_) executorch::runtime::EValue(std::move(*src_));
    }

}
