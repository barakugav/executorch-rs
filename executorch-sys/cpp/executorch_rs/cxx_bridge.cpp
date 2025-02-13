
#include <cstddef>
#include "executorch_rs/cxx_bridge.hpp"

#if defined(EXECUTORCH_RS_TESTOR_PTR)
#include "executorch/extension/tensor/tensor_ptr.h"
#endif

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
            scalar_type,
            dynamism,
            [allocation_ptr = allocation_ptr](void *) mutable {});
    }
#endif
}
