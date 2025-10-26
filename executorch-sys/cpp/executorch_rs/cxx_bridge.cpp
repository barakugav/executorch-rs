
#include <cstddef>
#include "executorch_rs/cxx_bridge.hpp"
#include "executorch_rs/layout.hpp"

#if defined(EXECUTORCH_RS_TENSOR_PTR)
#include "executorch/extension/tensor/tensor_ptr.h"
#endif

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
    VecEValue VecEValue_new(std::vector<executorch::runtime::EValue> &&vec)
    {
        executorch::runtime::EValue *arr = vec_to_array(std::move(vec));
        return VecEValue{
            .data = EValueRefMut{.ptr = arr},
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
#endif

    template <typename T>
    static Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
    {
        if (result.ok())
            *output = std::move(result.get());
        return static_cast<Error>(result.error());
    }

#if defined(EXECUTORCH_RS_STD)
    std::unique_ptr<executorch::extension::MallocMemoryAllocator> MallocMemoryAllocator_new()
    {
        return std::make_unique<executorch::extension::MallocMemoryAllocator>();
    }
    struct MemoryAllocator *MallocMemoryAllocator_as_memory_allocator(executorch::extension::MallocMemoryAllocator &self)
    {
        auto allocator = static_cast<executorch::runtime::MemoryAllocator *>(&self);
        return reinterpret_cast<struct MemoryAllocator *>(allocator);
    }
#endif

#if defined(EXECUTORCH_RS_TENSOR_PTR)
    std::shared_ptr<executorch::aten::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        ScalarType scalar_type,
        TensorShapeDynamism dynamism,
        rust::Box<executorch_rs::cxx_util::RustAny> allocation)
    {
        // std::function must be copyable, so we need to wrap the allocation in a shared_ptr
        std::shared_ptr<rust::Box<executorch_rs::cxx_util::RustAny>> allocation_ptr =
            std::make_shared<rust::Box<executorch_rs::cxx_util::RustAny>>(std::move(allocation));

        return executorch::extension::make_tensor_ptr(
            std::move(*sizes),
            data,
            std::move(*dim_order),
            std::move(*strides),
            static_cast<executorch::aten::ScalarType>(scalar_type),
            static_cast<executorch::aten::TensorShapeDynamism>(dynamism),
            [allocation_ptr = allocation_ptr](void *) mutable {});
    }
#endif

#if defined(EXECUTORCH_RS_MODULE)
    std::unique_ptr<executorch::extension::Module> Module_new(
        const std::string &file_path,
        rust::Slice<const rust::Str> data_files,
        const ModuleLoadMode load_mode,
        std::unique_ptr<executorch::runtime::EventTracer> event_tracer)
    {
        std::vector<std::string> data_files_;
        for (const auto &data_file : data_files)
        {
            data_files_.emplace_back(data_file);
        }
        auto load_mode_ = static_cast<executorch::extension::Module::LoadMode>(load_mode);
        return std::make_unique<executorch::extension::Module>(
            file_path,
            data_files_,
            load_mode_,
            std::move(event_tracer));
    }

    Error Module_load(executorch::extension::Module &self, ProgramVerification verification)
    {
        auto verification_ = static_cast<executorch::runtime::Program::Verification>(verification);
        auto ret = self.load(verification_);
        return static_cast<Error>(ret);
    }
    static executorch::runtime::Error Module_num_methods_(executorch::extension::Module &self, size_t &method_num_out)
    {
        method_num_out = ET_UNWRAP(self.num_methods());
        return executorch::runtime::Error::Ok;
    }
    Error Module_num_methods(executorch::extension::Module &self, size_t &method_num_out)
    {
        return static_cast<Error>(Module_num_methods_(self, method_num_out));
    }
    static executorch::runtime::Error Module_method_names_(executorch::extension::Module &self, rust::Vec<rust::String> &method_names_out)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        for (const std::string &method_name : method_names)
        {
            method_names_out.emplace_back(method_name);
        }
        return executorch::runtime::Error::Ok;
    }
    Error Module_method_names(executorch::extension::Module &self, rust::Vec<rust::String> &method_names_out)
    {
        return static_cast<Error>(Module_method_names_(self, method_names_out));
    }
    Error Module_load_method(executorch::extension::Module &self, const std::string &method_name, HierarchicalAllocator *planned_memory, executorch::runtime::EventTracer *event_tracer)
    {
        auto planned_memory_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(planned_memory);
        auto ret = self.load_method(method_name, planned_memory_, event_tracer);
        return static_cast<Error>(ret);
    }
    bool Module_unload_method(executorch::extension::Module &self, const std::string &method_name)
    {
        return self.unload_method(method_name);
    }
    bool Module_is_method_loaded(const executorch::extension::Module &self, const std::string &method_name)
    {
        return self.is_method_loaded(method_name);
    }
    Error Module_method_meta(executorch::extension::Module &self, const std::string &method_name, MethodMeta *method_meta_out)
    {
        auto method_meta_out_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(method_meta_out);
        return extract_result(self.method_meta(method_name), method_meta_out_);
    }
    static executorch::runtime::Error Module_execute_(executorch::extension::Module &self, const std::string &method_name, ArrayRefEValue inputs, VecEValue *outputs)
    {
        auto inputs_data = reinterpret_cast<const executorch::runtime::EValue *>(inputs.data.ptr);
        std::vector<executorch::runtime::EValue> inputs_vec(inputs_data, inputs_data + inputs.len);
        auto err = self.set_inputs(method_name, inputs_vec);
        if (err != executorch::runtime::Error::Ok)
            return err;
        std::vector<executorch::runtime::EValue> outputs_ = ET_UNWRAP(self.execute(method_name));
        *outputs = VecEValue_new(std::move(outputs_));
        return executorch::runtime::Error::Ok;
    }
    Error Module_execute(executorch::extension::Module &self, const std::string &method_name, ArrayRefEValue inputs, VecEValue *outputs)
    {
        return static_cast<Error>(Module_execute_(self, method_name, inputs, outputs));
    }
#endif
}
