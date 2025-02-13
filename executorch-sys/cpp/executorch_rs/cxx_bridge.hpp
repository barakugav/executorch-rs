#pragma once

#include "executorch_rs/defines.h"

#include <cstddef>
#include <cstdint>
#include "executorch/runtime/core/exec_aten/exec_aten.h"

#include "executorch-sys/src/cxx_bridge.rs.h"

namespace executorch_rs
{
#if defined(EXECUTORCH_RS_TESTOR_PTR)
    std::shared_ptr<executorch::aten::Tensor> TensorPtr_new(
        std::unique_ptr<std::vector<int32_t>> sizes,
        uint8_t *data,
        std::unique_ptr<std::vector<uint8_t>> dim_order,
        std::unique_ptr<std::vector<int32_t>> strides,
        executorch::aten::ScalarType scalar_type,
        executorch::aten::TensorShapeDynamism dynamism,
        rust::Box<executorch_rs::cxx_util::RustAny> allocation);
#endif
}
