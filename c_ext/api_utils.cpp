
#include <cstddef>

#include "c_ext/api_utils.hpp"

namespace executorch_rs
{

    namespace
    {

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

    Result_i64 MethodMeta_memory_planned_buffer_size(const torch::executor::MethodMeta *method_meta, size_t index)
    {
        return crate_Result_i64(method_meta->memory_planned_buffer_size(index));
    }

    torch::executor::util::MallocMemoryAllocator MallocMemoryAllocator_new()
    {
        return torch::executor::util::MallocMemoryAllocator();
    }
    torch::executor::HierarchicalAllocator HierarchicalAllocator_new(torch::executor::Span<torch::executor::Span<uint8_t>> buffers)
    {
        return torch::executor::HierarchicalAllocator(buffers);
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
}
