
#include <cstddef>

#include "executorch_rs/api_utils.hpp"

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

}