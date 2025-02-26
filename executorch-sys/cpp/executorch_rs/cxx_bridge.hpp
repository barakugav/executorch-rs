#pragma once

#include "executorch_rs/defines.h"
#include "executorch_rs/bridge.hpp"

#include <cstddef>
#include <cstdint>
#include "executorch/runtime/core/exec_aten/exec_aten.h"

#if defined(EXECUTORCH_RS_MODULE)
#include "executorch-sys/src/cxx_bridge/module.rs.h"
#include "executorch/extension/module/module.h"
#endif

#if defined(EXECUTORCH_RS_TENSOR_PTR)
#include "executorch-sys/src/cxx_bridge/tensor_ptr.rs.h"
#include <vector>
#endif

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_TENSOR_PTR)
    std::shared_ptr<executorch_rs::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        executorch::aten::ScalarType scalar_type,
        executorch::aten::TensorShapeDynamism dynamism,
        rust::Box<executorch_rs::cxx_util::RustAny> allocation);
#endif

#if defined(EXECUTORCH_RS_MODULE)
    std::unique_ptr<executorch::extension::Module> Module_new(
        rust::Str file_path,
        const executorch::extension::Module::LoadMode load_mode
        // executorch::runtime::EventTracer *event_tracer
    );

    executorch::runtime::Error Module_load(executorch::extension::Module &self, executorch::runtime::Program::Verification verification);
    executorch::runtime::Error Module_method_names(executorch::extension::Module &self, rust::Vec<rust::String> &method_names_out);
    executorch::runtime::Error Module_load_method(executorch::extension::Module &self, rust::Str method_name);
    bool Module_is_method_loaded(const executorch::extension::Module &self, rust::Str method_name);
    executorch::runtime::Error Module_method_meta(executorch::extension::Module &self, rust::Str method_name, MethodMeta *method_meta_out);
    executorch::runtime::Error Module_execute(executorch::extension::Module &self, rust::Str method_name, ArrayRefEValue inputs, VecEValue *outputs);
#endif
}
