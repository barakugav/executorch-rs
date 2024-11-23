
#include <cstddef>
#include <vector>
#include <cassert>
#include "executorch_rs_ext/api_utils.hpp"
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
    VecEValue VecEValue_new(std::vector<torch::executor::EValue> &&vec)
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
    static torch::executor::Error extract_result(const torch::executor::Result<T> &&result, T *output)
    {
        if (result.ok())
            *output = std::move(result.get());
        return result.error();
    }

    void Program_destructor(torch::executor::Program &self)
    {
        self.~Program();
    }

    torch::executor::Error MethodMeta_input_tag(const torch::executor::MethodMeta &self, size_t index, torch::executor::Tag *tag_out)
    {
        return extract_result(self.input_tag(index), tag_out);
    }
    torch::executor::Error MethodMeta_output_tag(const torch::executor::MethodMeta &self, size_t index, torch::executor::Tag *tag_out)
    {
        return extract_result(self.output_tag(index), tag_out);
    }
    torch::executor::Error MethodMeta_input_tensor_meta(const torch::executor::MethodMeta &self, size_t index, torch::executor::TensorInfo *tensor_info_out)
    {
        return extract_result(self.input_tensor_meta(index), tensor_info_out);
    }
    torch::executor::Error MethodMeta_output_tensor_meta(const torch::executor::MethodMeta &self, size_t index, torch::executor::TensorInfo *tensor_info_out)
    {
        return extract_result(self.output_tensor_meta(index), tensor_info_out);
    }
    torch::executor::Error MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta &self, size_t index, int64_t *size_out)
    {
        return extract_result(self.memory_planned_buffer_size(index), size_out);
    }

    torch::executor::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address)
    {
        return torch::executor::MemoryAllocator(size, base_address);
    }
    void *MemoryAllocator_allocate(torch::executor::MemoryAllocator &self, size_t size, size_t alignment)
    {
        return self.allocate(size, alignment);
    }
#if defined(EXECUTORCH_RS_STD)
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new()
    {
        return torch::executor::util::MallocMemoryAllocator();
    }
    void MallocMemoryAllocator_destructor(torch::executor::util::MallocMemoryAllocator &self)
    {
        self.~MallocMemoryAllocator();
    }
#endif
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(SpanSpanU8 buffers)
    {
        torch::executor::Span<torch::executor::Span<uint8_t>> buffers_ = *reinterpret_cast<torch::executor::Span<torch::executor::Span<uint8_t>> *>(&buffers);
        assert((void *)buffers_.begin() == (void *)buffers.data);
        assert(buffers_.size() == buffers.len);
        return torch::executor::HierarchicalAllocator(buffers_);
    }
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator &self)
    {
        self.~HierarchicalAllocator();
    }
    torch::executor::Error FileDataLoader_new(const char *file_path, size_t alignment, torch::executor::util::FileDataLoader *out)
    {
        // return extract_result(std::move(torch::executor::util::FileDataLoader::from(file_path, alignment)), out);
        auto res = torch::executor::util::FileDataLoader::from(file_path, alignment);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out) torch::executor::util::FileDataLoader(std::move(loader));
        return torch::executor::Error::Ok;
    }
    torch::executor::Error MmapDataLoader_new(const char *file_path, torch::executor::util::MmapDataLoader::MlockConfig mlock_config, torch::executor::util::MmapDataLoader *out)
    {
        // return extract_result(torch::executor::util::MmapDataLoader::from(file_path, mlock_config), out);
        auto res = torch::executor::util::MmapDataLoader::from(file_path, mlock_config);
        if (!res.ok())
            return res.error();
        auto &loader = res.get();
        new (out) torch::executor::util::MmapDataLoader(std::move(loader));
        return torch::executor::Error::Ok;
    }

    // Program
    torch::executor::Error Program_load(torch::executor::DataLoader *loader, torch::executor::Program::Verification verification, torch::executor::Program *out)
    {
        // return extract_result(torch::executor::Program::load(loader, verification), out);
        auto res = torch::executor::Program::load(loader, verification);
        if (!res.ok())
            return res.error();
        auto &program = res.get();
        new (out) torch::executor::Program(std::move(program));
        return torch::executor::Error::Ok;
    }
    torch::executor::Error Program_load_method(const torch::executor::Program &self, const char *method_name, torch::executor::MemoryManager *memory_manager, torch::executor::EventTracer *event_tracer, torch::executor::Method *out)
    {
        // return extract_result(std::move(self.load_method(method_name, memory_manager, event_tracer)), out);
        auto res = self.load_method(method_name, memory_manager, event_tracer);
        if (!res.ok())
            return res.error();
        auto &method = res.get();
        new (out) torch::executor::Method(std::move(method));
        return torch::executor::Error::Ok;
    }
    torch::executor::Error Program_get_method_name(const torch::executor::Program &self, size_t method_index, const char **out)
    {
        return extract_result(self.get_method_name(method_index), out);
    }
    torch::executor::Error Program_method_meta(const torch::executor::Program &self, const char *method_name, torch::executor::MethodMeta *method_meta_out)
    {
        return extract_result(self.method_meta(method_name), method_meta_out);
    }

    // TensorInfo
    ArrayRefI32 TensorInfo_sizes(const torch::executor::TensorInfo &self)
    {
        auto sizes = self.sizes();
        return ArrayRefI32{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefU8 TensorInfo_dim_order(const torch::executor::TensorInfo &self)
    {
        auto dim_order = self.dim_order();
        return ArrayRefU8{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }

    // Tensor
    void Tensor_new(exec_aten::Tensor *self, exec_aten::TensorImpl *tensor_impl)
    {
        new (self) exec_aten::Tensor(tensor_impl);
    }
    size_t Tensor_nbytes(const exec_aten::Tensor &self)
    {
        return self.nbytes();
    }
    ssize_t Tensor_size(const exec_aten::Tensor &self, ssize_t dim)
    {
        return self.size(dim);
    }
    ssize_t Tensor_dim(const exec_aten::Tensor &self)
    {
        return self.dim();
    }
    ssize_t Tensor_numel(const exec_aten::Tensor &self)
    {
        return self.numel();
    }
    exec_aten::ScalarType Tensor_scalar_type(const exec_aten::Tensor &self)
    {
        return self.scalar_type();
    }
    ssize_t Tensor_element_size(const exec_aten::Tensor &self)
    {
        return self.element_size();
    }
    ArrayRefSizesType Tensor_sizes(const exec_aten::Tensor &self)
    {
        auto sizes = self.sizes();
        return ArrayRefSizesType{
            .data = sizes.data(),
            .len = sizes.size(),
        };
    }
    ArrayRefDimOrderType Tensor_dim_order(const exec_aten::Tensor &self)
    {
        auto dim_order = self.dim_order();
        return ArrayRefDimOrderType{
            .data = dim_order.data(),
            .len = dim_order.size(),
        };
    }
    ArrayRefStridesType Tensor_strides(const exec_aten::Tensor &self)
    {
        auto strides = self.strides();
        return ArrayRefStridesType{
            .data = strides.data(),
            .len = strides.size(),
        };
    }
    const void *Tensor_const_data_ptr(const exec_aten::Tensor &self)
    {
        return self.const_data_ptr();
    }
    void *Tensor_mutable_data_ptr(const exec_aten::Tensor &self)
    {
        return self.mutable_data_ptr();
    }
    size_t Tensor_coordinate_to_index(const exec_aten::Tensor &self, const size_t *coordinate)
    {
        return torch::executor::coordinateToIndex(self, coordinate);
    }
    void Tensor_destructor(exec_aten::Tensor &self)
    {
        self.~Tensor();
    }

    void EValue_new_from_i64(torch::executor::EValue *self, int64_t value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_f64(torch::executor::EValue *self, double value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_f64_arr(torch::executor::EValue *self, ArrayRefF64 value)
    {
        exec_aten::ArrayRef<double> value_(value.data, value.len);
        new (self) torch::executor::EValue(value_);
    }
    void EValue_new_from_bool(torch::executor::EValue *self, bool value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_bool_arr(torch::executor::EValue *self, ArrayRefBool value)
    {
        exec_aten::ArrayRef<bool> value_(value.data, value.len);
        new (self) torch::executor::EValue(value_);
    }
    void EValue_new_from_chars(torch::executor::EValue *self, ArrayRefChar value)
    {
        new (self) torch::executor::EValue(value.data, value.len);
    }
    void EValue_new_from_tensor(torch::executor::EValue *self, const exec_aten::Tensor *value)
    {
        new (self) torch::executor::EValue(*value);
    }
    int64_t EValue_as_i64(const torch::executor::EValue &self)
    {
        return self.payload.copyable_union.as_int;
    }
    double EValue_as_f64(const torch::executor::EValue &self)
    {
        return self.payload.copyable_union.as_double;
    }
    bool EValue_as_bool(const torch::executor::EValue &self)
    {
        return self.payload.copyable_union.as_bool;
    }
    ArrayRefChar EValue_as_string(const torch::executor::EValue &self)
    {
        auto str = self.payload.copyable_union.as_string;
        return ArrayRefChar{
            .data = str.data(),
            .len = str.size(),
        };
    }
    ArrayRefBool EValue_as_bool_list(const torch::executor::EValue &self)
    {
        auto bool_list = self.payload.copyable_union.as_bool_list;
        return ArrayRefBool{
            .data = bool_list.data(),
            .len = bool_list.size(),
        };
    }
    ArrayRefF64 EValue_as_f64_list(const torch::executor::EValue &self)
    {
        auto f64_list = self.payload.copyable_union.as_double_list;
        return ArrayRefF64{
            .data = f64_list.data(),
            .len = f64_list.size(),
        };
    }
    void EValue_copy(const torch::executor::EValue *src, torch::executor::EValue *dst)
    {
        new (dst) torch::executor::EValue(*src);
    }
    void EValue_destructor(torch::executor::EValue &self)
    {
        self.~EValue();
    }
    void EValue_move(torch::executor::EValue *src, torch::executor::EValue *dst)
    {
        new (dst) torch::executor::EValue(std::move(*src));
    }
    // exec_aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const torch::executor::BoxedEvalueList<int64_t> &self)
    // {
    //     return self.get();
    // }
    // exec_aten::ArrayRef<exec_aten::Tensor> BoxedEvalueList_Tensor_get(const torch::executor::BoxedEvalueList<exec_aten::Tensor> &self)
    // {
    //     return self.get();
    // }

    torch::executor::util::BufferDataLoader BufferDataLoader_new(const void *data, size_t size)
    {
        return torch::executor::util::BufferDataLoader(data, size);
    }

#if defined(EXECUTORCH_RS_MODULE)
    void Module_new(torch::executor::Module *self, ArrayRefChar file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer)
    {
        std::string file_path_str(file_path.data, file_path.data + file_path.len);
        std::unique_ptr<torch::executor::EventTracer> event_tracer2(event_tracer);
        new (self) torch::executor::Module(file_path_str, mlock_config, std::move(event_tracer2));
    }
    void Module_destructor(torch::executor::Module &self)
    {
        self.~Module();
    }
    torch::executor::Error Module_method_names(torch::executor::Module &self, VecVecChar *method_names_out)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        std::vector<VecChar> method_names_vec;
        for (const std::string &method_name : method_names)
        {
            std::vector<char> method_name_vec(method_name.begin(), method_name.end());
            method_names_vec.push_back(VecChar_new(std::move(method_name_vec)));
        }
        *method_names_out = VecVecChar_new(std::move(method_names_vec));
        return torch::executor::Error::Ok;
    }
    torch::executor::Error Module_load_method(torch::executor::Module &self, ArrayRefChar method_name)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        return self.load_method(method_name_str);
    }
    bool Module_is_method_loaded(const torch::executor::Module &self, ArrayRefChar method_name)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        return self.is_method_loaded(method_name_str);
    }
    torch::executor::Error Module_method_meta(torch::executor::Module &self, torch::executor::ArrayRef<char> method_name, torch::executor::MethodMeta *method_meta_out)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return extract_result(self.method_meta(method_name_str), method_meta_out);
    }
    torch::executor::Error Module_execute(torch::executor::Module &self, ArrayRefChar method_name, ArrayRefEValue inputs, VecEValue *outputs)
    {
        std::string method_name_str(method_name.data, method_name.data + method_name.len);
        std::vector<torch::executor::EValue> inputs_vec(inputs.data, inputs.data + inputs.len);
        std::vector<torch::executor::EValue> outputs_ = ET_UNWRAP(self.execute(method_name_str, inputs_vec));
        *outputs = VecEValue_new(std::move(outputs_));
        return torch::executor::Error::Ok;
    }
#endif
}
