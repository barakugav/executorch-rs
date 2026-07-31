
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
    ET_VecEValue VecEValue_new(std::vector<executorch::runtime::EValue> &&vec)
    {
        executorch::runtime::EValue *arr = vec_to_array(std::move(vec));
        return ET_VecEValue{
            .data = ET_EValueRefMut{.ptr = arr},
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
#endif

    template <typename T>
    static ET_Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
    {
        if (result.ok())
            *output = std::move(result.get());
        return static_cast<ET_Error>(result.error());
    }

#if defined(EXECUTORCH_RS_STD)
    std::unique_ptr<executorch::extension::MallocMemoryAllocator> MallocMemoryAllocator_new()
    {
        return std::make_unique<executorch::extension::MallocMemoryAllocator>();
    }
    struct ET_MemoryAllocator *MallocMemoryAllocator_as_memory_allocator(executorch::extension::MallocMemoryAllocator &self)
    {
        auto allocator = static_cast<executorch::runtime::MemoryAllocator *>(&self);
        return reinterpret_cast<struct ET_MemoryAllocator *>(allocator);
    }
    std::unique_ptr<struct ET_MemoryAllocator> MallocMemoryAllocator_into_memory_allocator_unique_ptr(std::unique_ptr<executorch::extension::MallocMemoryAllocator> self)
    {
        std::unique_ptr<executorch::runtime::MemoryAllocator> ptr = std::move(self);
        return std::unique_ptr<struct ET_MemoryAllocator>(reinterpret_cast<struct ET_MemoryAllocator *>(ptr.release()));
    }

    std::unique_ptr<struct ET_MemoryAllocator> BufferMemoryAllocator_into_memory_allocator_unique_ptr(struct ET_MemoryAllocator &self)
    {
        auto ptr = std::make_unique<struct ET_MemoryAllocator>(std::move(self));
        self.~ET_MemoryAllocator();
        return ptr;
    }

#endif

#if defined(EXECUTORCH_RS_TENSOR_PTR)
    std::shared_ptr<executorch::aten::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        ET_ScalarType scalar_type,
        ET_TensorShapeDynamism dynamism,
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

    std::shared_ptr<executorch::aten::Tensor> TensorPtr_clone(
        const executorch::aten::Tensor &tensor,
        ET_ScalarType scalar_type)
    {
        return executorch::extension::clone_tensor_ptr(
            tensor, static_cast<executorch::aten::ScalarType>(scalar_type));
    }
#endif

#if defined(EXECUTORCH_RS_MODULE)
    std::unique_ptr<executorch::extension::Module> Module_new(
        const std::string &file_path,
        rust::Slice<const rust::Str> data_files,
        const ET_ModuleLoadMode load_mode,
        std::unique_ptr<executorch::runtime::EventTracer> event_tracer,
        std::unique_ptr<struct ET_MemoryAllocator> memory_allocator,
        std::unique_ptr<struct ET_MemoryAllocator> temp_allocator)
    {
        std::unique_ptr<executorch::runtime::MemoryAllocator> memory_allocator_(reinterpret_cast<executorch::runtime::MemoryAllocator *>(memory_allocator.release()));
        std::unique_ptr<executorch::runtime::MemoryAllocator> temp_allocator_(reinterpret_cast<executorch::runtime::MemoryAllocator *>(temp_allocator.release()));

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
            std::move(event_tracer),
            std::move(memory_allocator_),
            std::move(temp_allocator_));
    }

    ET_Error Module_load(executorch::extension::Module &self, ET_ProgramVerification verification)
    {
        auto verification_ = static_cast<executorch::runtime::Program::Verification>(verification);
        auto ret = self.load(verification_);
        return static_cast<ET_Error>(ret);
    }
    bool Module_is_loaded(const executorch::extension::Module &self)
    {
        return self.is_loaded();
    }
    static executorch::runtime::Error Module_num_methods_(executorch::extension::Module &self, size_t *method_num_out)
    {
        *method_num_out = ET_UNWRAP(self.num_methods());
        return executorch::runtime::Error::Ok;
    }
    ET_Error Module_num_methods(executorch::extension::Module &self, size_t *method_num_out)
    {
        return static_cast<ET_Error>(Module_num_methods_(self, method_num_out));
    }
    static executorch::runtime::Error Module_method_names_(executorch::extension::Module &self, rust::Vec<rust::String> *method_names_out)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        new (method_names_out) rust::Vec<rust::String>();
        for (const std::string &method_name : method_names)
        {
            method_names_out->emplace_back(method_name);
        }
        return executorch::runtime::Error::Ok;
    }
    ET_Error Module_method_names(executorch::extension::Module &self, rust::Vec<rust::String> *method_names_out)
    {
        return static_cast<ET_Error>(Module_method_names_(self, method_names_out));
    }
    ET_Error Module_load_method(executorch::extension::Module &self, const std::string &method_name, ET_HierarchicalAllocator *planned_memory, executorch::runtime::EventTracer *event_tracer)
    {
        auto planned_memory_ = checked_reinterpret_cast<executorch::runtime::HierarchicalAllocator>(planned_memory);
        auto ret = self.load_method(method_name, planned_memory_, event_tracer);
        return static_cast<ET_Error>(ret);
    }
    bool Module_unload_method(executorch::extension::Module &self, const std::string &method_name)
    {
        return self.unload_method(method_name);
    }
    bool Module_is_method_loaded(const executorch::extension::Module &self, const std::string &method_name)
    {
        return self.is_method_loaded(method_name);
    }
    ET_Error Module_method_meta(executorch::extension::Module &self, const std::string &method_name, ET_MethodMeta *method_meta_out)
    {
        auto method_meta_out_ = checked_reinterpret_cast<executorch::runtime::MethodMeta>(method_meta_out);
        return extract_result(self.method_meta(method_name), method_meta_out_);
    }
    static executorch::runtime::Error Module_execute_(executorch::extension::Module &self, const std::string &method_name, ET_ArrayRefEValue inputs, ET_VecEValue *outputs)
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
    ET_Error Module_execute(executorch::extension::Module &self, const std::string &method_name, ET_ArrayRefEValue inputs, ET_VecEValue *outputs)
    {
        return static_cast<ET_Error>(Module_execute_(self, method_name, inputs, outputs));
    }
#endif
}
