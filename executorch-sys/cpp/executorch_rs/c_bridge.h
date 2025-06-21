#pragma once

#if defined(EXECUTORCH_RS_MODULE) && !defined(EXECUTORCH_RS_STD)
#error "EXECUTORCH_RS_MODULE requires EXECUTORCH_RS_STD"
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * ExecuTorch Error type.
     */
    enum Error
    {
        /*
         * System errors.
         */

        /// Status indicating a successful operation.
        Error_Ok = 0x00,

        /// An internal error occurred.
        Error_Internal = 0x01,

        /// Status indicating the executor is in an invalid state for a target
        /// operation
        Error_InvalidState = 0x2,

        /// Status indicating there are no more steps of execution to run
        Error_EndOfMethod = 0x03,

        /*
         * Logical errors.
         */

        /// Operation is not supported in the current context.
        Error_NotSupported = 0x10,

        /// Operation is not yet implemented.
        Error_NotImplemented = 0x11,

        /// User provided an invalid argument.
        Error_InvalidArgument = 0x12,

        /// Object is an invalid type for the operation.
        Error_InvalidType = 0x13,

        /// Operator(s) missing in the operator registry.
        Error_OperatorMissing = 0x14,

        /*
         * Resource errors.
         */

        /// Requested resource could not be found.
        Error_NotFound = 0x20,

        /// Could not allocate the requested memory.
        Error_MemoryAllocationFailed = 0x21,

        /// Could not access a resource.
        Error_AccessFailed = 0x22,

        /// Error caused by the contents of a program.
        Error_InvalidProgram = 0x23,

        /// Error caused by the contents of external data.
        Error_InvalidExternalData = 0x24,

        /// Does not have enough resources to perform the requested operation.
        Error_OutOfResources = 0x25,

        /*
         * Delegate errors.
         */

        /// Init stage: Backend receives an incompatible delegate version.
        Error_DelegateInvalidCompatibility = 0x30,
        /// Init stage: Backend fails to allocate memory.
        Error_DelegateMemoryAllocationFailed = 0x31,
        /// Execute stage: The handle is invalid.
        Error_DelegateInvalidHandle = 0x32,

    };

    /**
     * Describes the presence of an ExecuTorch program header.
     */
    enum ProgramHeaderStatus
    {
        /**
         * An ExecuTorch program header is present, and its version is compatible
         * with this version of the runtime.
         */
        ProgramHeaderStatus_CompatibleVersion,

        /**
         * An ExecuTorch program header is present, but its version is not
         * compatible with this version of the runtime.
         */
        ProgramHeaderStatus_IncompatibleVersion,

        /**
         * An ExecuTorch program header is not present.
         */
        ProgramHeaderStatus_NotPresent,

        /**
         * The data provided was too short to find the program header.
         */
        ProgramHeaderStatus_ShortData,
    };

    /**
     * Types of validation that the Program can do before parsing the data.
     */
    enum ProgramVerification
    {
        /**
         * Do minimal verification of the data, ensuring that the header appears
         * correct.
         *
         * Has minimal runtime overhead.
         */
        ProgramVerification_Minimal,
        /**
         * Do full verification of the data, ensuring that internal pointers are
         * self-consistent and that the data has not been truncated or obviously
         * corrupted. May not catch all types of corruption, but should guard
         * against illegal memory operations during parsing.
         *
         * Will have higher runtime overhead, scaling with the complexity of the
         * proram data.
         */
        ProgramVerification_InternalConsistency,
    };

    /**
     * Describes how and whether to lock loaded pages with `mlock()`.
     *
     * Using `mlock()` typically loads all of the pages immediately, and will
     * typically ensure that they are not swapped out. The actual behavior
     * will depend on the host system.
     */
    enum MmapDataLoaderMlockConfig
    {
        /// Do not call `mlock()` on loaded pages.
        ModuleLoadMode_NoMlock,
        /// Call `mlock()` on loaded pages, failing if it fails.
        ModuleLoadMode_UseMlock,
        /// Call `mlock()` on loaded pages, ignoring errors if it fails.
        ModuleLoadMode_UseMlockIgnoreErrors,
    };

    /**
     * Enum to define loading behavior.
     */
    enum ModuleLoadMode
    {
        /// Load the whole file as a buffer.
        ModuleLoadMode_File,
        /// Use mmap to load pages into memory.
        ModuleLoadMode_Mmap,
        /// Use memory locking and handle errors.
        ModuleLoadMode_MmapUseMlock,
        /// Use memory locking and ignore errors.
        ModuleLoadMode_MmapUseMlockIgnoreErrors,
    };

    enum Tag
    {
        Tag_None,
        Tag_Tensor,
        Tag_String,
        Tag_Double,
        Tag_Int,
        Tag_Bool,
        Tag_ListBool,
        Tag_ListDouble,
        Tag_ListInt,
        Tag_ListTensor,
        Tag_ListScalar,
        Tag_ListOptionalTensor,
    };

    enum ScalarType
    {
        ScalarType_Byte,
        ScalarType_Char,
        ScalarType_Short,
        ScalarType_Int,
        ScalarType_Long,
        ScalarType_Half,
        ScalarType_Float,
        ScalarType_Double,
        ScalarType_ComplexHalf,
        ScalarType_ComplexFloat,
        ScalarType_ComplexDouble,
        ScalarType_Bool,
        ScalarType_QInt8,
        ScalarType_QUInt8,
        ScalarType_QInt32,
        ScalarType_BFloat16,
        ScalarType_QUInt4x2,
        ScalarType_QUInt2x4,
        ScalarType_Bits1x8,
        ScalarType_Bits2x4,
        ScalarType_Bits4x2,
        ScalarType_Bits8,
        ScalarType_Bits16,
        ScalarType_Float8_e5m2,
        ScalarType_Float8_e4m3fn,
        ScalarType_Float8_e5m2fnuz,
        ScalarType_Float8_e4m3fnuz,
        ScalarType_UInt16,
        ScalarType_UInt32,
        ScalarType_UInt64,
    };

    /**
     * The type used for elements of `Tensor.sizes()`.
     */
    typedef int32_t SizesType;

    /**
     * The type used for elements of `Tensor.dim_order()`.
     */
    typedef uint8_t DimOrderType;

    /**
     * The type used for elements of `Tensor.strides()`.
     */
    typedef int32_t StridesType;

    /**
     * The resizing capabilities of a Tensor.
     *
     * The rank of an ExecuTorch Tensors can never change, but shape sometimes can.
     */
    enum TensorShapeDynamism
    {
        /// Cannot change shape.
        TensorShapeDynamism_STATIC = 0,
        /// Shape cannot exceed initial capacity.
        TensorShapeDynamism_DYNAMIC_BOUND = 1,
        /// No restriction on shape and capacity.
        TensorShapeDynamism_DYNAMIC_UNBOUND = 2,
    };

    struct EValueStorage
    {
        size_t _blob[4];
    };
    struct EValueRef
    {
        const void *ptr;
    };
    struct EValueRefMut
    {
        void *ptr;
    };
    struct TensorStorage
    {
        size_t _blob[1];
    };
    struct TensorRef
    {
        const void *ptr;
    };
    struct TensorRefMut
    {
        void *ptr;
    };
    struct TensorImpl
    {
        size_t _blob[8];
    };
    struct Program
    {

        // program_data_ (4)
        //   free_fn_
        //   free_fn_context_
        //   data_
        //   size_
        // loader_
        // internal_program_
        // segment_base_offset_
        // constant_segment_data_ (4)
        //   free_fn_
        //   free_fn_context_
        //   data_
        //   size_
        size_t _blob1[11];
        // pte_data_map_
        struct // optional<PteDataMap>
        {
            union
            {
                char _blob2_opt_dummy;
                // vtable
                // loader_
                // segment_base_offset_
                // named_data_
                // segments_
                size_t _blob2_opt_val[5];
            };
            bool _blob2_opt_flag;
        };
    };
    struct TensorInfo
    {
        size_t _blob[6];
    };
    struct MethodMeta
    {
        size_t _blob[1];
    };
    struct Method
    {

        // step_state_ (2)
        // program_
        // memory_manager_
        // temp_allocator_
        // serialization_plan_
        // event_tracer_
        // n_value_
        // values_
        // n_delegate_
        // delegates_
        // n_chains_
        // chains_
        // external_constants_
        // n_external_constants_
        size_t _blob1[15];
        // init_state_;
        uint8_t _blob2[1];
    };
    struct DataLoaderRefMut
    {
        void *ptr;
    };

    struct BufferDataLoader
    {
        size_t _blob[3];
    };
#if defined(EXECUTORCH_RS_DATA_LOADER)
    struct FileDataLoader
    {
        size_t _blob[5];
    };
    struct MmapDataLoader
    {
        size_t _blob_1[4];
        int _blob_2[2];
    };
#endif

    struct MemoryAllocator
    {
        size_t _blob_1[4];
        uint32_t _blob_2[2];
    };
    struct HierarchicalAllocator
    {
        size_t _blob[34];
    };
    struct MemoryManager
    {
        size_t _blob[3];
    };

    struct OptionalTensorStorage
    {
        union
        {
            char _dummy;
            struct TensorStorage _val;
        };
        bool _flag;
    };
    struct OptionalTensorRef
    {
        const void *ptr;
    };
    struct OptionalTensorRefMut
    {
        void *ptr;
    };

#if defined(EXECUTORCH_RS_STD)
    struct VecChar
    {
        char *data;
        size_t len;
        size_t cap;
    };
    void executorch_VecChar_destructor(struct VecChar *vec);

    struct VecVecChar
    {
        struct VecChar *data;
        size_t len;
        size_t cap;
    };
    void executorch_VecVecChar_destructor(struct VecVecChar *vec);

    struct VecEValue
    {
        struct EValueRefMut data;
        size_t len;
        size_t cap;
    };
    void executorch_VecEValue_destructor(struct VecEValue *vec);
#endif

    struct ArrayRefChar
    {
        const char *data;
        size_t len;
    };
    struct ArrayRefBool
    {
        const bool *data;
        size_t len;
    };
    struct ArrayRefU8
    {
        const uint8_t *data;
        size_t len;
    };
    struct ArrayRefI32
    {
        const int32_t *data;
        size_t len;
    };
    struct ArrayRefI64
    {
        const int64_t *data;
        size_t len;
    };
    struct ArrayRefF64
    {
        const double *data;
        size_t len;
    };
    struct ArrayRefUsizeType
    {
        const size_t *data;
        size_t len;
    };
    struct ArrayRefSizesType
    {
        const SizesType *data;
        size_t len;
    };
    struct ArrayRefDimOrderType
    {
        const DimOrderType *data;
        size_t len;
    };
    struct ArrayRefStridesType
    {
        const StridesType *data;
        size_t len;
    };
    struct ArrayRefTensor
    {
        struct TensorRef data;
        size_t len;
    };
    struct ArrayRefOptionalTensor
    {
        struct OptionalTensorRef data;
        size_t len;
    };
    struct ArrayRefEValue
    {
        struct EValueRef data;
        size_t len;
    };
    struct ArrayRefEValuePtr
    {
        const struct EValueRef *data;
        size_t len;
    };
    struct SpanU8
    {
        uint8_t *data;
        size_t len;
    };
    struct SpanSpanU8
    {
        struct SpanU8 *data;
        size_t len;
    };
    // struct SpanEValue
    // {
    //     EValue *data;
    //     size_t len;
    // };
    struct SpanI64
    {
        int64_t *data;
        size_t len;
    };
    struct SpanTensor
    {
        struct TensorRefMut data;
        size_t len;
    };
    struct SpanOptionalTensor
    {
        struct OptionalTensorRefMut data;
        size_t len;
    };
    struct BoxedEvalueListI64
    {
        struct ArrayRefEValuePtr wrapped_vals;
        struct SpanI64 unwrapped_vals;
    };
    struct BoxedEvalueListTensor
    {
        struct ArrayRefEValuePtr wrapped_vals;
        struct SpanTensor unwrapped_vals;
    };
    struct BoxedEvalueListOptionalTensor
    {
        struct ArrayRefEValuePtr wrapped_vals;
        struct SpanOptionalTensor unwrapped_vals;
    };

    struct EventTracerRefMut
    {
        void *ptr;
    };
#if defined(EXECUTORCH_RS_ETDUMP)
    struct ETDumpGen
    {
        // vtable
        size_t _blob0[1];
        // kUnsetChainId
        // debug_handle_
        int _blob1[2];
        // event_tracer_enable_debugging_
        // log_intermediate_tensors_
        bool _blob2[2];
        // bundled_input_index_
        // event_tracer_debug_level_
        // event_tracer_profiling_level_
        int _blob3[3];
        // builder_
        // num_blocks_
        // data_sink_
        // buffer_data_sink_ (5)
        //   DataSinkBase vtable
        //   BufferDataSink::debug_buffer_ (2)
        //   BufferDataSink::offset_
        //   BufferDataSink::alignment_
        size_t _blob4[8];
        // bundled_input_index_
        // state_
        int _blob5[2];
        // alloc_ (6)
        size_t _blob6[6];
    };
#endif

    void executorch_pal_init();

    struct MemoryAllocator executorch_MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *executorch_MemoryAllocator_allocate(struct MemoryAllocator *self, size_t size, size_t alignment);
    struct HierarchicalAllocator executorch_HierarchicalAllocator_new(struct SpanSpanU8 buffers);
    void executorch_HierarchicalAllocator_destructor(struct HierarchicalAllocator *self);
    struct MemoryManager executorch_MemoryManager_new(
        struct MemoryAllocator *method_allocator,
        struct HierarchicalAllocator *planned_memory,
        struct MemoryAllocator *temp_allocator);

    // Loaders
    struct BufferDataLoader executorch_BufferDataLoader_new(const void *data, size_t size);
    struct DataLoaderRefMut executorch_BufferDataLoader_as_data_loader_mut(struct BufferDataLoader *self);
#if defined(EXECUTORCH_RS_DATA_LOADER)
    enum Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, struct FileDataLoader *out);
    void executorch_FileDataLoader_destructor(struct FileDataLoader *self);
    struct DataLoaderRefMut executorch_FileDataLoader_as_data_loader_mut(struct FileDataLoader *self);
    enum Error executorch_MmapDataLoader_new(const char *file_path, enum MmapDataLoaderMlockConfig mlock_config, struct MmapDataLoader *out);
    void executorch_MmapDataLoader_destructor(struct MmapDataLoader *self);
    struct DataLoaderRefMut executorch_MmapDataLoader_as_data_loader_mut(struct MmapDataLoader *self);

#endif

    bool executorch_is_valid_dim_order_and_strides(size_t dim, const SizesType *sizes, const DimOrderType *dim_order, const StridesType *strides);
    enum Error executorch_stride_to_dim_order(const StridesType *strides, size_t dims, DimOrderType *dim_order);

    // Tensor
    void executorch_TensorImpl_new(
        struct TensorImpl *self,
        enum ScalarType type,
        size_t dim,
        SizesType *sizes,
        void *data,
        DimOrderType *dim_order,
        StridesType *strides,
        enum TensorShapeDynamism dynamism);
    void executorch_Tensor_new(struct TensorRefMut self, struct TensorImpl *tensor_impl);
    size_t executorch_Tensor_nbytes(struct TensorRef self);
    size_t executorch_Tensor_size(struct TensorRef self, size_t dim);
    size_t executorch_Tensor_dim(struct TensorRef self);
    size_t executorch_Tensor_numel(struct TensorRef self);
    enum ScalarType executorch_Tensor_scalar_type(struct TensorRef self);
    size_t executorch_Tensor_element_size(struct TensorRef self);
    struct ArrayRefSizesType executorch_Tensor_sizes(struct TensorRef self);
    struct ArrayRefDimOrderType executorch_Tensor_dim_order(struct TensorRef self);
    struct ArrayRefStridesType executorch_Tensor_strides(struct TensorRef self);
    const void *executorch_Tensor_const_data_ptr(struct TensorRef self);
    void *executorch_Tensor_mutable_data_ptr(struct TensorRef self);
    int64_t executorch_Tensor_coordinate_to_index(struct TensorRef self, struct ArrayRefUsizeType coordinate);
    void executorch_Tensor_destructor(struct TensorRefMut self);

    // OptionalTensor
    struct TensorRef executorch_OptionalTensor_get(struct OptionalTensorRef self);

    // EValue
    void executorch_EValue_new_none(struct EValueRefMut self);
    void executorch_EValue_new_from_i64(struct EValueRefMut self, int64_t value);
    void executorch_EValue_new_from_i64_list(struct EValueRefMut self, struct BoxedEvalueListI64 value);
    void executorch_EValue_new_from_f64(struct EValueRefMut self, double value);
    void executorch_EValue_new_from_f64_list(struct EValueRefMut self, struct ArrayRefF64 value);
    void executorch_EValue_new_from_bool(struct EValueRefMut self, bool value);
    void executorch_EValue_new_from_bool_list(struct EValueRefMut self, struct ArrayRefBool value);
    void executorch_EValue_new_from_string(struct EValueRefMut self, struct ArrayRefChar value);
    void executorch_EValue_new_from_tensor(struct EValueRefMut self, struct TensorRef value);
    void executorch_EValue_new_from_tensor_list(struct EValueRefMut self, struct BoxedEvalueListTensor value);
    void executorch_EValue_new_from_optional_tensor_list(struct EValueRefMut self, struct BoxedEvalueListOptionalTensor value);
    enum Tag executorch_EValue_tag(struct EValueRef self);
    int64_t executorch_EValue_as_i64(struct EValueRef self);
    struct ArrayRefI64 executorch_EValue_as_i64_list(struct EValueRef self);
    double executorch_EValue_as_f64(struct EValueRef self);
    struct ArrayRefF64 executorch_EValue_as_f64_list(struct EValueRef self);
    bool executorch_EValue_as_bool(struct EValueRef self);
    struct ArrayRefBool executorch_EValue_as_bool_list(struct EValueRef self);
    struct ArrayRefChar executorch_EValue_as_string(struct EValueRef self);
    struct TensorRef executorch_EValue_as_tensor(struct EValueRef self);
    struct ArrayRefTensor executorch_EValue_as_tensor_list(struct EValueRef self);
    struct ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(struct EValueRef self);
    void executorch_EValue_copy(struct EValueRef src, struct EValueRefMut dst);
    void executorch_EValue_destructor(struct EValueRefMut self);
    void executorch_EValue_move(struct EValueRefMut src, struct EValueRefMut dst);

    // Program
    enum ProgramHeaderStatus executorch_Program_check_header(const void *data, size_t size);
    enum Error executorch_Program_load(struct DataLoaderRefMut loader, enum ProgramVerification verification, struct Program *out);
    enum Error executorch_Program_load_method(const struct Program *self, const char *method_name, struct MemoryManager *memory_manager, struct EventTracerRefMut event_tracer, struct Method *out);
    enum Error executorch_Program_get_method_name(const struct Program *self, size_t method_index, const char **out);
    enum Error executorch_Program_method_meta(const struct Program *self, const char *method_name, struct MethodMeta *method_meta_out);
    size_t executorch_Program_num_methods(const struct Program *self);
    void executorch_Program_destructor(struct Program *self);

    // MethodMeta
    size_t executorch_Method_inputs_size(const struct Method *self);
    size_t executorch_Method_outputs_size(const struct Method *self);
    enum Error executorch_Method_set_input(struct Method *self, struct EValueRef input_evalue, size_t input_idx);
    struct EValueRef executorch_Method_get_output(const struct Method *self, size_t i);
    enum Error executorch_Method_execute(struct Method *self);
    void executorch_Method_destructor(struct Method *self);
    const char *executorch_MethodMeta_name(const struct MethodMeta *self);
    size_t executorch_MethodMeta_num_inputs(const struct MethodMeta *self);
    size_t executorch_MethodMeta_num_outputs(const struct MethodMeta *self);
    size_t executorch_MethodMeta_num_memory_planned_buffers(const struct MethodMeta *self);
    enum Error executorch_MethodMeta_input_tag(const struct MethodMeta *self, size_t index, enum Tag *tag_out);
    enum Error executorch_MethodMeta_output_tag(const struct MethodMeta *self, size_t index, enum Tag *tag_out);
    enum Error executorch_MethodMeta_input_tensor_meta(const struct MethodMeta *self, size_t index, struct TensorInfo *tensor_info_out);
    enum Error executorch_MethodMeta_output_tensor_meta(const struct MethodMeta *self, size_t index, struct TensorInfo *tensor_info_out);
    enum Error executorch_MethodMeta_memory_planned_buffer_size(const struct MethodMeta *self, size_t index, int64_t *size_out);
    bool executorch_MethodMeta_uses_backend(const struct MethodMeta *self, const char *backend_name);
    size_t executorch_MethodMeta_num_backends(const struct MethodMeta *self);
    enum Error executorch_MethodMeta_get_backend_name(const struct MethodMeta *self, size_t index, const char **backend_name_out);

    // TensorInfo
    struct ArrayRefI32 executorch_TensorInfo_sizes(const struct TensorInfo *self);
    struct ArrayRefU8 executorch_TensorInfo_dim_order(const struct TensorInfo *self);
    enum ScalarType executorch_TensorInfo_scalar_type(const struct TensorInfo *self);
    size_t executorch_TensorInfo_nbytes(const struct TensorInfo *self);

#if defined(EXECUTORCH_RS_ETDUMP)
    // ETDumpGen
    struct ETDumpGen executorch_ETDumpGen_new(struct SpanU8 buffer);
    struct ArrayRefU8 executorch_ETDumpGen_get_etdump_data(struct ETDumpGen *self);
    struct EventTracerRefMut executorch_ETDumpGen_as_event_tracer_mut(struct ETDumpGen *self);
#endif

#ifdef __cplusplus
} // end of extern "C" block
#endif
