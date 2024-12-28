
#include <cstddef>
#include <vector>
#include <cassert>
#include "executorch_rs/bridge.hpp"
#include "executorch/runtime/core/exec_aten/util/tensor_util.h"

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_STD)
    template <typename T>
    T *vec_to_array(std::vector<T> &&vec)
    {
        T *arr = new T[vec.size()];
        std::move(vec.begin(), vec.end(), arr);
        return arr;
    }
    VecChar VecChar_new(std::vector<char> &&vec)
    {
        return VecChar{
            .data = vec_to_array(std::move(vec)),
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
    void VecChar_destructor(VecChar *vec)
    {
        delete[] vec->data;
    }
    VecVecChar VecVecChar_new(std::vector<VecChar> &&vec)
    {
        return VecVecChar{
            .data = vec_to_array(std::move(vec)),
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
    void VecVecChar_destructor(VecVecChar *vec)
    {
        for (size_t i = 0; i < vec->len; i++)
            VecChar_destructor(&vec->data[i]);
        delete[] vec->data;
    }
    VecEValue VecEValue_new(std::vector<executorch::runtime::EValue> &&vec)
    {
        return VecEValue{
            .data = vec_to_array(std::move(vec)),
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
    void VecEValue_destructor(VecEValue *vec)
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

    void Program_destructor(executorch::runtime::Program &self)
    {
        self.~Program();
    }

    executorch::runtime::Error MethodMeta_input_tag(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::Tag *tag_out)
    {
        return extract_result(self.input_tag(index), tag_out);
    }
    executorch::runtime::Error MethodMeta_output_tag(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::Tag *tag_out)
    {
        return extract_result(self.output_tag(index), tag_out);
    }
    executorch::runtime::Error MethodMeta_input_tensor_meta(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::TensorInfo *tensor_info_out)
    {
        return extract_result(self.input_tensor_meta(index), tensor_info_out);
    }
    executorch::runtime::Error MethodMeta_output_tensor_meta(const executorch::runtime::MethodMeta &self, size_t index, executorch::runtime::TensorInfo *tensor_info_out)
    {
        return extract_result(self.output_tensor_meta(index), tensor_info_out);
    }
    executorch::runtime::Error MethodMeta_memory_planned_buffer_size(const executorch::runtime::MethodMeta &self, size_t index, int64_t *size_out)
    {
        return extract_result(self.memory_planned_buffer_size(index), size_out);
    }

    executorch::runtime::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address)
    {
        return executorch::runtime::MemoryAllocator(size, base_address);
    }
    void *MemoryAllocator_allocate(executorch::runtime::MemoryAllocator &self, size_t size, size_t alignment)
    {
        return self.allocate(size, alignment);
    }
#if defined(EXECUTORCH_RS_STD)
    executorch::extension::MallocMemoryAllocator MallocMemoryAllocator_new()
    {
        return executorch::extension::MallocMemoryAllocator();
    }
    void MallocMemoryAllocator_destructor(executorch::extension::MallocMemoryAllocator &self)
    {
        self.~MallocMemoryAllocator();
    }
#endif
    executorch::runtime::HierarchicalAllocator HierarchicalAllocator_new(SpanSpanU8 buffers)
    {
        executorch::runtime::Span<executorch::runtime::Span<uint8_t>> buffers_ = *reinterpret_cast<executorch::runtime::Span<executorch::runtime::Span<uint8_t>> *>(&buffers);
        assert((void *)buffers_.begin() == (void *)buffers.data);
        assert(buffers_.size() == buffers.len);
        return executorch::runtime::HierarchicalAllocator(buffers_);
    }
    void HierarchicalAllocator_destructor(executorch::runtime::HierarchicalAllocator &self)
    {
        self.~HierarchicalAllocator();
    }
    executorch::runtime::Error FileDataLoader_new(const char *file_path, size_t alignment, executorch::extension::FileDataLoader *out)
    {
        // return extract_result(std::move(executorch::extension::FileDataLoader::from(file_path, alignment)), out);
        auto res = executorch::extension::FileDataLoader::from(file_path, alignment);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out) executorch::extension::FileDataLoader(std::move(loader));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error MmapDataLoader_new(const char *file_path, executorch::extension::MmapDataLoader::MlockConfig mlock_config, executorch::extension::MmapDataLoader *out)
    {
        // return extract_result(executorch::extension::MmapDataLoader::from(file_path, mlock_config), out);
        auto res = executorch::extension::MmapDataLoader::from(file_path, mlock_config);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out) executorch::extension::MmapDataLoader(std::move(loader));
        return executorch::runtime::Error::Ok;
    }

    // Program
    executorch::runtime::Error Program_load(executorch::runtime::DataLoader *loader, executorch::runtime::Program::Verification verification, executorch::runtime::Program *out)
    {
        // return extract_result(executorch::runtime::Program::load(loader, verification), out);
        auto res = executorch::runtime::Program::load(loader, verification);
        if (!res.ok())
            return res.error();
        auto &program = res.get();
        new (out) executorch::runtime::Program(std::move(program));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error Program_load_method(const executorch::runtime::Program &self, const char *method_name, executorch::runtime::MemoryManager *memory_manager, executorch::runtime::EventTracer *event_tracer, executorch::runtime::Method *out)
    {
        // return extract_result(std::move(self.load_method(method_name, memory_manager, event_tracer)), out);
        auto res = self.load_method(method_name, memory_manager, event_tracer);
        if (!res.ok())
            return res.error();
        auto &method = res.get();
        new (out) executorch::runtime::Method(std::move(method));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error Program_get_method_name(const executorch::runtime::Program &self, size_t method_index, const char **out)
    {
        return extract_result(self.get_method_name(method_index), out);
    }
    executorch::runtime::Error Program_method_meta(const executorch::runtime::Program &self, const char *method_name, executorch::runtime::MethodMeta *method_meta_out)
    {
        return extract_result(self.method_meta(method_name), method_meta_out);
    }

    // TensorInfo
    ArrayRefI32 TensorInfo_sizes(const executorch::runtime::TensorInfo &self)
    {
        auto sizes = self.sizes();
        return ArrayRefI32{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefU8 TensorInfo_dim_order(const executorch::runtime::TensorInfo &self)
    {
        auto dim_order = self.dim_order();
        return ArrayRefU8{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }

    // Tensor
    void Tensor_new(executorch::aten::Tensor *self, executorch::aten::TensorImpl *tensor_impl)
    {
        new (self) executorch::aten::Tensor(tensor_impl);
    }
    size_t Tensor_nbytes(const executorch::aten::Tensor &self)
    {
        return self.nbytes();
    }
    ssize_t Tensor_size(const executorch::aten::Tensor &self, ssize_t dim)
    {
        return self.size(dim);
    }
    ssize_t Tensor_dim(const executorch::aten::Tensor &self)
    {
        return self.dim();
    }
    ssize_t Tensor_numel(const executorch::aten::Tensor &self)
    {
        return self.numel();
    }
    executorch::aten::ScalarType Tensor_scalar_type(const executorch::aten::Tensor &self)
    {
        return self.scalar_type();
    }
    ssize_t Tensor_element_size(const executorch::aten::Tensor &self)
    {
        return self.element_size();
    }
    ArrayRefSizesType Tensor_sizes(const executorch::aten::Tensor &self)
    {
        auto sizes = self.sizes();
        return ArrayRefSizesType{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefDimOrderType Tensor_dim_order(const executorch::aten::Tensor &self)
    {
        auto dim_order = self.dim_order();
        return ArrayRefDimOrderType{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }
    ArrayRefStridesType Tensor_strides(const executorch::aten::Tensor &self)
    {
        auto strides = self.strides();
        return ArrayRefStridesType{
            .data = strides.data(),
            .len = strides.size(),
        };
    }
    const void *Tensor_const_data_ptr(const executorch::aten::Tensor &self)
    {
        return self.const_data_ptr();
    }
    void *Tensor_mutable_data_ptr(const executorch::aten::Tensor &self)
    {
        return self.mutable_data_ptr();
    }
    size_t Tensor_coordinate_to_index(const executorch::aten::Tensor &self, const size_t *coordinate)
    {
        return executorch::runtime::coordinateToIndex(self, coordinate);
    }
    void Tensor_destructor(executorch::aten::Tensor &self)
    {
        self.~Tensor();
    }

    void EValue_new_from_i64(executorch::runtime::EValue *self, int64_t value)
    {
        new (self) executorch::runtime::EValue(value);
    }
    void EValue_new_from_f64(executorch::runtime::EValue *self, double value)
    {
        new (self) executorch::runtime::EValue(value);
    }
    void EValue_new_from_f64_arr(executorch::runtime::EValue *self, ArrayRefF64 value)
    {
        executorch::aten::ArrayRef<double> value_(value.data, value.len);
        new (self) executorch::runtime::EValue(value_);
    }
    void EValue_new_from_bool(executorch::runtime::EValue *self, bool value)
    {
        new (self) executorch::runtime::EValue(value);
    }
    void EValue_new_from_bool_arr(executorch::runtime::EValue *self, ArrayRefBool value)
    {
        executorch::aten::ArrayRef<bool> value_(value.data, value.len);
        new (self) executorch::runtime::EValue(value_);
    }
    void EValue_new_from_chars(executorch::runtime::EValue *self, ArrayRefChar value)
    {
        new (self) executorch::runtime::EValue(value.data, value.len);
    }
    void EValue_new_from_tensor(executorch::runtime::EValue *self, const executorch::aten::Tensor *value)
    {
        new (self) executorch::runtime::EValue(*value);
    }
    int64_t EValue_as_i64(const executorch::runtime::EValue &self)
    {
        return self.payload.copyable_union.as_int;
    }
    double EValue_as_f64(const executorch::runtime::EValue &self)
    {
        return self.payload.copyable_union.as_double;
    }
    bool EValue_as_bool(const executorch::runtime::EValue &self)
    {
        return self.payload.copyable_union.as_bool;
    }
    ArrayRefChar EValue_as_string(const executorch::runtime::EValue &self)
    {
        auto str = self.payload.copyable_union.as_string;
        return ArrayRefChar{
            .data = str.data(),
            .len = str.size(),
        };
    }
    ArrayRefBool EValue_as_bool_list(const executorch::runtime::EValue &self)
    {
        auto bool_list = self.payload.copyable_union.as_bool_list;
        return ArrayRefBool{
            .data = bool_list.data(),
            .len = bool_list.size(),
        };
    }
    ArrayRefF64 EValue_as_f64_list(const executorch::runtime::EValue &self)
    {
        auto f64_list = self.payload.copyable_union.as_double_list;
        return ArrayRefF64{
            .data = f64_list.data(),
            .len = f64_list.size(),
        };
    }
    void EValue_copy(const executorch::runtime::EValue *src, executorch::runtime::EValue *dst)
    {
        new (dst) executorch::runtime::EValue(*src);
    }
    void EValue_destructor(executorch::runtime::EValue &self)
    {
        self.~EValue();
    }
    void EValue_move(executorch::runtime::EValue *src, executorch::runtime::EValue *dst)
    {
        new (dst) executorch::runtime::EValue(std::move(*src));
    }
    // executorch::aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const executorch::runtime::EValue<int64_t> &self)
    // {
    //     return self.get();
    // }
    // executorch::aten::ArrayRef<executorch::aten::Tensor> BoxedEvalueList_Tensor_get(const executorch::runtime::EValue<executorch::aten::Tensor> &self)
    // {
    //     return self.get();
    // }

    executorch::extension::BufferDataLoader BufferDataLoader_new(const void *data, size_t size)
    {
        return executorch::extension::BufferDataLoader(data, size);
    }

#if defined(EXECUTORCH_RS_MODULE)
    void Module_new(executorch::extension::Module *self, ArrayRefChar file_path, const executorch::extension::Module::LoadMode load_mode, executorch::runtime::EventTracer *event_tracer)
    {
        std::string file_path_str(file_path.data, file_path.data + file_path.len);
        std::unique_ptr<executorch::runtime::EventTracer> event_tracer2(event_tracer);
        new (self) executorch::extension::Module(file_path_str, load_mode, std::move(event_tracer2));
    }
    void Module_destructor(executorch::extension::Module &self)
    {
        self.~Module();
    }
    executorch::runtime::Error Module_method_names(executorch::extension::Module &self, VecVecChar *method_names_out)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        std::vector<VecChar> method_names_vec;
        for (const std::string &method_name : method_names)
        {
            std::vector<char> method_name_vec(method_name.begin(), method_name.end());
            method_names_vec.push_back(VecChar_new(std::move(method_name_vec)));
        }
        *method_names_out = VecVecChar_new(std::move(method_names_vec));
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error Module_load_method(executorch::extension::Module &self, ArrayRefChar method_name)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        return self.load_method(method_name_str);
    }
    bool Module_is_method_loaded(const executorch::extension::Module &self, ArrayRefChar method_name)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        return self.is_method_loaded(method_name_str);
    }
    executorch::runtime::Error Module_method_meta(executorch::extension::Module &self, executorch::runtime::ArrayRef<char> method_name, executorch::runtime::MethodMeta *method_meta_out)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return extract_result(self.method_meta(method_name_str), method_meta_out);
    }
    executorch::runtime::Error Module_execute(executorch::extension::Module &self, ArrayRefChar method_name, ArrayRefEValue inputs, VecEValue *outputs)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        std::vector<executorch::runtime::EValue> inputs_vec(inputs.data, inputs.data + inputs.len);
        std::vector<executorch::runtime::EValue> outputs_ = ET_UNWRAP(self.execute(method_name_str, inputs_vec));
        *outputs = VecEValue_new(std::move(outputs_));
        return executorch::runtime::Error::Ok;
    }
#endif
}
