
#include <cstddef>
#include <vector>

#include "executorch_rs_ext/api_utils.hpp"
#include "executorch/runtime/core/exec_aten/util/tensor_util.h"

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_STD)
    template <typename T>
    Vec<T> crate_Vec(std::vector<T> &&vec)
    {
        size_t len = vec.size();
        T *arr = new T[len];
        std::move(vec.begin(), vec.end(), arr);
        return Vec<T>{
            .data = arr,
            .len = len,
            .cap = len,
        };
    }

// Its safe to call the destructor of elements in `vec->data[len..cap]` because we created them with `new T[len]`
// aka default constructor
#define VEC_DESTRUCTOR_IMPL(T, name)          \
    void Vec_##name##_destructor(Vec<T> *vec) \
    {                                         \
        delete[] vec->data;                   \
    }

    VEC_DESTRUCTOR_IMPL(char, char)
    VEC_DESTRUCTOR_IMPL(Vec<char>, Vec_char)
    VEC_DESTRUCTOR_IMPL(torch::executor::EValue, EValue)
#endif

    static_assert(sizeof(Result_i64) == sizeof(torch::executor::Result<int64_t>), "Result_i64 size mismatch");
    // static_assert(offsetof(Result_i64, value_) == offsetof(torch::executor::Result<int64_t>, value_), "Result_i64 value_ offset mismatch");
    // static_assert(offsetof(Result_i64, error_) == offsetof(torch::executor::Result<int64_t>, error_), "Result_i64 error_ offset mismatch");
    // static_assert(offsetof(Result_i64, hasValue_) == offsetof(torch::executor::Result<int64_t>, hasValue_), "Result_i64 hasValue_ offset mismatch");
    Result_i64 crate_Result_i64(const torch::executor::Result<int64_t> &result)
    {
        Result_i64 result2;
        memcpy(&result2, &result, sizeof(Result_i64));
        return result2;
    }

    static_assert(sizeof(Result_MethodMeta) == sizeof(torch::executor::Result<torch::executor::MethodMeta>), "Result_MethodMeta size mismatch");
    // static_assert(offsetof(Result_MethodMeta, value_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, value_), "Result_MethodMeta value_ offset mismatch");
    // static_assert(offsetof(Result_MethodMeta, error_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, error_), "Result_MethodMeta error_ offset mismatch");
    // static_assert(offsetof(Result_MethodMeta, hasValue_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, hasValue_), "Result_MethodMeta hasValue_ offset mismatch");
    Result_MethodMeta crate_Result_MethodMeta(const torch::executor::Result<torch::executor::MethodMeta> &result)
    {
        Result_MethodMeta result2;
        memcpy(&result2, &result, sizeof(Result_MethodMeta));
        return result2;
    }

    Result_MethodMeta Program_method_meta(const torch::executor::Program &self, const char *method_name)
    {
        return crate_Result_MethodMeta(self.method_meta(method_name));
    }
    void Program_destructor(torch::executor::Program &self)
    {
        self.~Program();
    }

    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta &self, size_t index)
    {
        return crate_Result_i64(self.memory_planned_buffer_size(index));
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
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers)
    {
        return torch::executor::HierarchicalAllocator(buffers);
    }
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator &self)
    {
        self.~HierarchicalAllocator();
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
    exec_aten::ArrayRef<exec_aten::SizesType> Tensor_sizes(const exec_aten::Tensor &self)
    {
        return self.sizes();
    }
    exec_aten::ArrayRef<exec_aten::DimOrderType> Tensor_dim_order(const exec_aten::Tensor &self)
    {
        return self.dim_order();
    }
    exec_aten::ArrayRef<exec_aten::StridesType> Tensor_strides(const exec_aten::Tensor &self)
    {
        return self.strides();
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
    void EValue_new_from_f64_arr(torch::executor::EValue *self, exec_aten::ArrayRef<double> value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_bool(torch::executor::EValue *self, bool value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_bool_arr(torch::executor::EValue *self, exec_aten::ArrayRef<bool> value)
    {
        new (self) torch::executor::EValue(value);
    }
    void EValue_new_from_chars(torch::executor::EValue *self, exec_aten::ArrayRef<char> value)
    {
        new (self) torch::executor::EValue(value.begin(), value.end() - value.begin());
    }
    void EValue_new_from_tensor(torch::executor::EValue *self, const exec_aten::Tensor *value)
    {
        new (self) torch::executor::EValue(*value);
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
    void Module_new(torch::executor::Module *self, torch::executor::ArrayRef<char> file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer)
    {
        std::string file_path_str(file_path.begin(), file_path.end());
        std::unique_ptr<torch::executor::EventTracer> event_tracer2(event_tracer);
        new (self) torch::executor::Module(file_path_str, mlock_config, std::move(event_tracer2));
    }
    void Module_destructor(torch::executor::Module &self)
    {
        self.~Module();
    }
    torch::executor::Result<Vec<Vec<char>>> Module_method_names(torch::executor::Module &self)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        std::vector<Vec<char>> method_names_vec;
        for (const std::string &method_name : method_names)
        {
            std::vector<char> method_name_vec(method_name.begin(), method_name.end());
            method_names_vec.push_back(crate_Vec(std::move(method_name_vec)));
        }
        return crate_Vec(std::move(method_names_vec));
    }
    torch::executor::Error Module_load_method(torch::executor::Module &self, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return self.load_method(method_name_str);
    }
    bool Module_is_method_loaded(const torch::executor::Module &self, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return self.is_method_loaded(method_name_str);
    }
    Result_MethodMeta Module_method_meta(torch::executor::Module &self, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return crate_Result_MethodMeta(self.method_meta(method_name_str));
    }
    torch::executor::Result<Vec<torch::executor::EValue>> Module_execute(torch::executor::Module &self, torch::executor::ArrayRef<char> method_name, torch::executor::ArrayRef<torch::executor::EValue> inputs)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        std::vector<torch::executor::EValue> inputs_vec(inputs.begin(), inputs.end());
        std::vector<torch::executor::EValue> outputs = ET_UNWRAP(self.execute(method_name_str, inputs_vec));
        return crate_Vec(std::move(outputs));
    }
#endif
}
