
#include <cstddef>
#include "executorch_rs/cxx_bridge.hpp"

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
            .data = reinterpret_cast<EValue *>(arr),
            .len = vec.size(),
            .cap = vec.size(),
        };
    }
#endif

    template <typename T>
    static executorch::runtime::Error extract_result(const executorch::runtime::Result<T> &&result, T *output)
    {
        if (result.ok())
            *output = std::move(result.get());
        return result.error();
    }

#if defined(EXECUTORCH_RS_TENSOR_PTR)
    std::shared_ptr<executorch_rs::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        executorch::aten::ScalarType scalar_type,
        executorch::aten::TensorShapeDynamism dynamism,
        rust::Box<executorch_rs::cxx_util::RustAny> allocation)
    {
        // std::function must be copyable, so we need to wrap the allocation in a shared_ptr
        std::shared_ptr<rust::Box<executorch_rs::cxx_util::RustAny>> allocation_ptr =
            std::make_shared<rust::Box<executorch_rs::cxx_util::RustAny>>(std::move(allocation));

        auto tensor = executorch::extension::make_tensor_ptr(
            std::move(*sizes),
            data,
            std::move(*dim_order),
            std::move(*strides),
            scalar_type,
            dynamism,
            [allocation_ptr = allocation_ptr](void *) mutable {});
        return std::reinterpret_pointer_cast<executorch_rs::Tensor>(tensor);
    }
#endif

#if defined(EXECUTORCH_RS_MODULE)
    std::unique_ptr<executorch::extension::Module> Module_new(
        const std::string &file_path,
        const executorch::extension::Module::LoadMode load_mode
        // executorch::runtime::EventTracer *event_tracer
    )
    {
        return std::make_unique<executorch::extension::Module>(file_path, load_mode);
    }

    executorch::runtime::Error Module_load(executorch::extension::Module &self, executorch::runtime::Program::Verification verification)
    {
        return self.load(verification);
    }
    executorch::runtime::Error Module_method_names(executorch::extension::Module &self, rust::Vec<rust::String> &method_names_out)
    {
        std::unordered_set<std::string> method_names = ET_UNWRAP(self.method_names());
        for (const std::string &method_name : method_names)
        {
            method_names_out.emplace_back(method_name);
        }
        return executorch::runtime::Error::Ok;
    }
    executorch::runtime::Error Module_load_method(executorch::extension::Module &self, rust::Str method_name)
    {
        return self.load_method((std::string)method_name);
    }
    bool Module_is_method_loaded(const executorch::extension::Module &self, rust::Str method_name)
    {
        return self.is_method_loaded((std::string)method_name);
    }
    executorch::runtime::Error Module_method_meta(executorch::extension::Module &self, rust::Str method_name, MethodMeta *method_meta_out)
    {
        auto method_meta_out_ = reinterpret_cast<executorch::runtime::MethodMeta *>(method_meta_out);
        return extract_result(self.method_meta((std::string)method_name), method_meta_out_);
    }
    executorch::runtime::Error Module_execute(executorch::extension::Module &self, rust::Str method_name, ArrayRefEValue inputs, VecEValue *outputs)
    {
        auto inputs_data = reinterpret_cast<const executorch::runtime::EValue *>(inputs.data);
        std::vector<executorch::runtime::EValue> inputs_vec(inputs_data, inputs_data + inputs.len);
        std::vector<executorch::runtime::EValue> outputs_ = ET_UNWRAP(self.execute((std::string)method_name, inputs_vec));
        *outputs = VecEValue_new(std::move(outputs_));
        return executorch::runtime::Error::Ok;
    }
#endif
}
