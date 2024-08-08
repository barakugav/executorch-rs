
#include <cstddef>

#include "executorch_rs_ext/api_utils.hpp"

namespace executorch_rs
{
    namespace
    {
        template <typename T>
        struct ManuallyDrop
        {
            union
            {
                T value;
            };
            ManuallyDrop(T &&value) : value(std::move(value)) {}
            ~ManuallyDrop() {}
        };

        template <typename T>
        RawVec<T> crate_RawVec(std::vector<T> &&vec)
        {
            auto vec2 = ManuallyDrop<std::vector<T>>(std::move(vec));
            return RawVec<T>{
                .data = vec2.value.data(),
                .len = vec2.value.size(),
                .cap = vec2.value.capacity(),
            };
        }

        static_assert(sizeof(Result_i64) == sizeof(torch::executor::Result<int64_t>), "Result_i64 size mismatch");
        // static_assert(offsetof(Result_i64, value_) == offsetof(torch::executor::Result<int64_t>, value_), "Result_i64 value_ offset mismatch");
        // static_assert(offsetof(Result_i64, error_) == offsetof(torch::executor::Result<int64_t>, error_), "Result_i64 error_ offset mismatch");
        // static_assert(offsetof(Result_i64, hasValue_) == offsetof(torch::executor::Result<int64_t>, hasValue_), "Result_i64 hasValue_ offset mismatch");
        Result_i64 crate_Result_i64(const torch::executor::Result<int64_t> &result)
        {
            Result_i64 result2{
                .error_ = torch::executor::Error::Ok,
                .hasValue_ = false,
            };
            memcpy(&result2, &result, sizeof(Result_i64));
            return result2;
        }

        static_assert(sizeof(Result_MethodMeta) == sizeof(torch::executor::Result<torch::executor::MethodMeta>), "Result_MethodMeta size mismatch");
        // static_assert(offsetof(Result_MethodMeta, value_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, value_), "Result_MethodMeta value_ offset mismatch");
        // static_assert(offsetof(Result_MethodMeta, error_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, error_), "Result_MethodMeta error_ offset mismatch");
        // static_assert(offsetof(Result_MethodMeta, hasValue_) == offsetof(torch::executor::Result<torch::executor::MethodMeta>, hasValue_), "Result_MethodMeta hasValue_ offset mismatch");
        Result_MethodMeta crate_Result_MethodMeta(const torch::executor::Result<torch::executor::MethodMeta> &result)
        {
            Result_MethodMeta result2{
                .error_ = torch::executor::Error::Ok,
                .hasValue_ = false,
            };
            memcpy(&result2, &result, sizeof(Result_MethodMeta));
            return result2;
        }
    }

    Result_MethodMeta Program_method_meta(const torch::executor::Program *program, const char *method_name)
    {
        return crate_Result_MethodMeta(program->method_meta(method_name));
    }
    void Program_destructor(torch::executor::Program *program)
    {
        program->~Program();
    }

    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta *method_meta, size_t index)
    {
        return crate_Result_i64(method_meta->memory_planned_buffer_size(index));
    }

    torch::executor::MemoryAllocator MemoryAllocator_new(uint32_t size, uint8_t *base_address)
    {
        return torch::executor::MemoryAllocator(size, base_address);
    }
    void *MemoryAllocator_allocate(torch::executor::MemoryAllocator *allocator, size_t size, size_t alignment)
    {
        return allocator->allocate(size, alignment);
    }
#if defined(EXECUTORCH_RS_STD)
    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new()
    {
        return torch::executor::util::MallocMemoryAllocator();
    }
    void MallocMemoryAllocator_destructor(torch::executor::util::MallocMemoryAllocator *allocator)
    {
        allocator->~MallocMemoryAllocator();
    }
#endif
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers)
    {
        return torch::executor::HierarchicalAllocator(buffers);
    }
    void HierarchicalAllocator_destructor(torch::executor::HierarchicalAllocator *allocator)
    {
        allocator->~HierarchicalAllocator();
    }

    // Tensor

    size_t Tensor_nbytes(const exec_aten::Tensor *tensor)
    {
        return tensor->nbytes();
    }
    ssize_t Tensor_size(const exec_aten::Tensor *tensor, ssize_t dim)
    {
        return tensor->size(dim);
    }
    ssize_t Tensor_dim(const exec_aten::Tensor *tensor)
    {
        return tensor->dim();
    }
    ssize_t Tensor_numel(const exec_aten::Tensor *tensor)
    {
        return tensor->numel();
    }
    exec_aten::ScalarType Tensor_scalar_type(const exec_aten::Tensor *tensor)
    {
        return tensor->scalar_type();
    }
    ssize_t Tensor_element_size(const exec_aten::Tensor *tensor)
    {
        return tensor->element_size();
    }
    const exec_aten::ArrayRef<exec_aten::SizesType> Tensor_sizes(const exec_aten::Tensor *tensor)
    {
        return tensor->sizes();
    }
    const exec_aten::ArrayRef<exec_aten::DimOrderType> Tensor_dim_order(const exec_aten::Tensor *tensor)
    {
        return tensor->dim_order();
    }
    const exec_aten::ArrayRef<exec_aten::StridesType> Tensor_strides(const exec_aten::Tensor *tensor)
    {
        return tensor->strides();
    }
    const void *Tensor_const_data_ptr(const exec_aten::Tensor *tensor)
    {
        return tensor->const_data_ptr();
    }
    void *Tensor_mutable_data_ptr(const exec_aten::Tensor *tensor)
    {
        return tensor->mutable_data_ptr();
    }
    void Tensor_destructor(exec_aten::Tensor *tensor)
    {
        tensor->~Tensor();
    }

    void EValue_destructor(torch::executor::EValue *evalue)
    {
        evalue->~EValue();
    }
    const exec_aten::ArrayRef<int64_t> BoxedEvalueList_i64_get(const torch::executor::BoxedEvalueList<int64_t> *list)
    {
        return list->get();
    }
    const exec_aten::ArrayRef<exec_aten::Tensor> BoxedEvalueList_Tensor_get(const torch::executor::BoxedEvalueList<exec_aten::Tensor> *list)
    {
        return list->get();
    }

#if defined(EXECUTORCH_RS_DATA_LOADER)
    torch::executor::util::BufferDataLoader BufferDataLoader_new(const void *data, size_t size)
    {
        return torch::executor::util::BufferDataLoader(data, size);
    }
#endif

#if defined(EXECUTORCH_RS_MODULE)
    torch::executor::Module Module_new(torch::executor::ArrayRef<char> file_path, torch::executor::Module::MlockConfig mlock_config, torch::executor::EventTracer *event_tracer)
    {
        std::string file_path_str(file_path.begin(), file_path.end());
        std::unique_ptr<torch::executor::EventTracer> event_tracer2(event_tracer);
        return torch::executor::Module(file_path_str, mlock_config, std::move(event_tracer2));
    }
    void Module_destructor(torch::executor::Module *module)
    {
        module->~Module();
    }
    torch::executor::Result<RawVec<RawVec<char>>> Module_method_names(torch::executor::Module *module)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(module->method_names());
        std::vector<RawVec<char>> method_names_vec;
        for (const std::string &method_name : method_names)
        {
            std::vector<char> method_name_vec(method_name.begin(), method_name.end());
            method_names_vec.push_back(crate_RawVec(std::move(method_name_vec)));
        }
        return crate_RawVec(std::move(method_names_vec));
    }
    torch::executor::Error Module_load_method(torch::executor::Module *module, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return module->load_method(method_name_str);
    }
    bool Module_is_method_loaded(const torch::executor::Module *module, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return module->is_method_loaded(method_name_str);
    }
    Result_MethodMeta Module_method_meta(torch::executor::Module *module, torch::executor::ArrayRef<char> method_name)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        return crate_Result_MethodMeta(module->method_meta(method_name_str));
    }
    torch::executor::Result<RawVec<torch::executor::EValue>> Module_execute(torch::executor::Module *module, torch::executor::ArrayRef<char> method_name, torch::executor::ArrayRef<torch::executor::EValue> inputs)
    {
        std::string method_name_str(method_name.begin(), method_name.end());
        std::vector<torch::executor::EValue> inputs_vec(inputs.begin(), inputs.end());
        std::vector<torch::executor::EValue> outputs = ET_UNWRAP(module->execute(method_name_str, inputs_vec));
        return crate_RawVec(std::move(outputs));
    }
#endif
}
