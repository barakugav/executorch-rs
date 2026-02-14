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
    enum Error : uint32_t
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

        /// Registration error: Exceeding the maximum number of kernels.
        Error_RegistrationExceedingMaxKernels = 0x15,

        /// Registration error: The kernel is already registered.
        Error_RegistrationAlreadyRegistered = 0x16,

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
    enum ProgramHeaderStatus : uint32_t
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
    enum ProgramVerification : uint8_t
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
    enum MmapDataLoaderMlockConfig : uint32_t
    {
        /// Do not call `mlock()` on loaded pages.
        MmapDataLoaderMlockConfig_NoMlock,
        /// Call `mlock()` on loaded pages, failing if it fails.
        MmapDataLoaderMlockConfig_UseMlock,
        /// Call `mlock()` on loaded pages, ignoring errors if it fails.
        MmapDataLoaderMlockConfig_UseMlockIgnoreErrors,
    };

    /**
     * Enum to define loading behavior.
     */
    enum ModuleLoadMode : uint32_t
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

    enum Tag : uint32_t
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

    enum ScalarType : int8_t
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
    enum TensorShapeDynamism : uint8_t
    {
        /// Cannot change shape.
        TensorShapeDynamism_STATIC = 0,
        /// Shape cannot exceed initial capacity.
        TensorShapeDynamism_DYNAMIC_BOUND = 1,
        /// No restriction on shape and capacity.
        TensorShapeDynamism_DYNAMIC_UNBOUND = 2,
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
    struct EValueStorage
    {
        union
        {
            // as_int
            int64_t _blob1;
            // as_double
            double _blob2;
            // as_bool
            bool _blob3;
            // as_string
            size_t _blob4;
            // as_double_list
            size_t _blob5;
            // as_bool_list
            size_t _blob6;
            // as_int_list
            size_t _blob7;
            // as_tensor_list
            size_t _blob8;
            // as_list_optional_tensor
            size_t _blob9;
            // as_tensor
            struct TensorStorage _blob10;
        };
        // tag
        uint32_t _blob11;
    };
    struct EValueRef
    {
        const void *ptr;
    };
    struct EValueRefMut
    {
        void *ptr;
    };
    struct FreeableBuffer
    {
        union
        {
            struct
            {
                size_t _blob1[2];
            };
            struct
            {
                uint64_t _blob2;
                size_t _blob3;
            };
        };
        uint8_t _blob4;
        size_t _blob5[2];
    };
    struct Program
    {

        // program_data_
        struct FreeableBuffer _blob1;
        // loader_
        // internal_program_
        // segment_base_offset_
        size_t _blob2[3];
        // constant_segment_data_
        struct FreeableBuffer _blob3;
        // pte_data_map_
        struct // optional<PteDataMap>
        {
            union
            {
                char _blob4_opt_dummy;
                // vtable
                // loader_
                // segment_base_offset_
                // named_data_
                // segments_
                size_t _blob4_opt_val[5];
            };
            bool _blob4_opt_flag;
        };
    };
    struct TensorInfo
    {

        // sizes_ (2)
        // dim_order_ (2)
        // name_ (2)
        size_t _blob1[6];
        // scalar_type_
        uint8_t _blob2;
        // is_memory_planned_
        bool _blob3;
        // nbytes_
        size_t _blob4;
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
        // input_set_
        // n_delegate_
        // delegates_
        // n_chains_
        // chains_
        // merged_data_map_
        // external_constants_
        // n_external_constants_
        size_t _blob1[17];
        // init_state_;
        uint8_t _blob2[1];
    };

    struct TensorLayout
    {
        // sizes_ (2)
        // dim_order_ (2)
        size_t _blob1[4];
        // scalar_type_
        int8_t _blob2;
        // nbytes_
        size_t _blob3;
    };

    struct DataLoaderRefMut
    {
        void *ptr;
    };
    struct NamedDataMapRef
    {
        const void *ptr;
    };
    struct NamedDataMapRefMut
    {
        void *ptr;
    };
#if defined(EXECUTORCH_RS_FLAT_TENSOR)
    struct FlatTensorDataMap
    {
        // vtable
        size_t _blob0[1];
        // header_
        uint64_t _blob1[4];
        // flat_tensor_data_
        struct FreeableBuffer _blob2;
        // flat_tensor_
        // loader_
        size_t _blob3[2];
    };
#endif

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
    // struct SpanEValue
    // {
    //     EValue *data;
    //     size_t len;
    // };
    struct BoxedEvalueListI64
    {
        struct ArrayRefEValuePtr wrapped_vals;
        int64_t *unwrapped_vals;
    };
    struct BoxedEvalueListTensor
    {
        struct ArrayRefEValuePtr wrapped_vals;
        struct TensorRefMut unwrapped_vals;
    };
    struct BoxedEvalueListOptionalTensor
    {
        struct ArrayRefEValuePtr wrapped_vals;
        struct OptionalTensorRefMut unwrapped_vals;
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
        // filter_
        size_t _blob7;
    };
#endif

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

    // NamedDataMap
    enum Error executorch_NamedDataMap_get_tensor_layout(
        struct NamedDataMapRef self,
        struct ArrayRefChar key,
        struct TensorLayout *out);
    enum Error executorch_NamedDataMap_get_num_keys(struct NamedDataMapRef self, uint32_t *out);
    enum Error executorch_NamedDataMap_get_key(
        struct NamedDataMapRef self,
        uint32_t index,
        const char **out_data);

#if defined(EXECUTORCH_RS_FLAT_TENSOR)
    // FlatTensorDataMap
    enum Error executorch_FlatTensorDataMap_load(struct DataLoaderRefMut loader, struct FlatTensorDataMap *out);
    struct NamedDataMapRefMut executorch_FlatTensorDataMap_as_named_data_map_mut(struct FlatTensorDataMap *self);
#endif

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
    int64_t executorch_Tensor_coordinate_to_index_unchecked(struct TensorRef self, struct ArrayRefUsizeType coordinate);
    void executorch_Tensor_destructor(struct TensorRefMut self);

    // OptionalTensor
    struct TensorRef executorch_OptionalTensor_get(struct OptionalTensorRef self);

    // TensorLayout
    // enum Error executorch_TensorLayout_create(
    //     struct ArrayRefI32 sizes,
    //     struct ArrayRefU8 dim_order,
    //     enum ScalarType scalar_type,
    //     struct TensorLayout *out);
    struct ArrayRefI32 executorch_TensorLayout_sizes(const struct TensorLayout *self);
    struct ArrayRefU8 executorch_TensorLayout_dim_order(const struct TensorLayout *self);
    enum ScalarType executorch_TensorLayout_scalar_type(const struct TensorLayout *self);
    size_t executorch_TensorLayout_nbytes(const struct TensorLayout *self);

    // EValue
    void executorch_EValue_new_none(struct EValueRefMut self);
    void executorch_EValue_new_from_i64(struct EValueRefMut self, int64_t value);
    void executorch_EValue_new_from_i64_list(struct EValueRefMut self, const struct BoxedEvalueListI64 *value);
    void executorch_EValue_new_from_f64(struct EValueRefMut self, double value);
    void executorch_EValue_new_from_f64_list(struct EValueRefMut self, const struct ArrayRefF64 *value);
    void executorch_EValue_new_from_bool(struct EValueRefMut self, bool value);
    void executorch_EValue_new_from_bool_list(struct EValueRefMut self, const struct ArrayRefBool *value);
    void executorch_EValue_new_from_string(struct EValueRefMut self, const struct ArrayRefChar *value);
    void executorch_EValue_new_from_tensor(struct EValueRefMut self, struct TensorRef value);
    void executorch_EValue_new_from_tensor_list(struct EValueRefMut self, const struct BoxedEvalueListTensor *value);
    void executorch_EValue_new_from_optional_tensor_list(struct EValueRefMut self, const struct BoxedEvalueListOptionalTensor *value);
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
    enum Error executorch_Program_load_method(
        const struct Program *self,
        const char *method_name,
        struct MemoryManager *memory_manager,
        struct EventTracerRefMut event_tracer,
        struct NamedDataMapRef named_data_map,
        struct Method *out);
    enum Error executorch_Program_get_method_name(const struct Program *self, size_t method_index, const char **out);
    enum Error executorch_Program_get_named_data_map(const struct Program *self, struct NamedDataMapRef *out);
    enum Error executorch_Program_method_meta(const struct Program *self, const char *method_name, struct MethodMeta *method_meta_out);
    size_t executorch_Program_num_methods(const struct Program *self);
    void executorch_Program_destructor(struct Program *self);

    // MethodMeta
    size_t executorch_Method_inputs_size(const struct Method *self);
    size_t executorch_Method_outputs_size(const struct Method *self);
    enum Error executorch_Method_set_input(struct Method *self, struct EValueRef input_evalue, size_t input_idx);
    struct EValueRef executorch_Method_get_output(const struct Method *self, size_t i);
    enum Error executorch_Method_get_attribute(struct Method *self, struct ArrayRefChar name, struct TensorRefMut out);
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
    size_t executorch_MethodMeta_num_attributes(const struct MethodMeta *self);
    enum Error executorch_MethodMeta_attribute_tensor_meta(const struct MethodMeta *self, size_t index, struct TensorInfo *tensor_info_out);
    enum Error executorch_MethodMeta_memory_planned_buffer_size(const struct MethodMeta *self, size_t index, int64_t *size_out);
    bool executorch_MethodMeta_uses_backend(const struct MethodMeta *self, const char *backend_name);
    size_t executorch_MethodMeta_num_backends(const struct MethodMeta *self);
    enum Error executorch_MethodMeta_get_backend_name(const struct MethodMeta *self, size_t index, const char **backend_name_out);

    // TensorInfo
    struct ArrayRefI32 executorch_TensorInfo_sizes(const struct TensorInfo *self);
    struct ArrayRefU8 executorch_TensorInfo_dim_order(const struct TensorInfo *self);
    enum ScalarType executorch_TensorInfo_scalar_type(const struct TensorInfo *self);
    bool executorch_TensorInfo_is_memory_planned(const struct TensorInfo *self);
    size_t executorch_TensorInfo_nbytes(const struct TensorInfo *self);
    struct ArrayRefChar executorch_TensorInfo_name(const struct TensorInfo *self);

#if defined(EXECUTORCH_RS_ETDUMP)
    // ETDumpGen
    struct ETDumpGen executorch_ETDumpGen_new(struct SpanU8 buffer);
    struct ArrayRefU8 executorch_ETDumpGen_get_etdump_data(struct ETDumpGen *self);
    struct EventTracerRefMut executorch_ETDumpGen_as_event_tracer_mut(struct ETDumpGen *self);
#endif

    // Platform structs and functions

    /// Platform timestamp in system ticks.
    typedef uint64_t executorch_timestamp_t;

    /**
     * Represents the conversion ratio from system ticks to nanoseconds.
     * To convert, use nanoseconds = ticks * numerator / denominator.
     */
    struct executorch_tick_ratio
    {
        uint64_t numerator;
        uint64_t denominator;
    };

    /**
     * Severity level of a log message. Values must map to printable 7-bit ASCII
     * uppercase letters.
     */
    enum executorch_pal_log_level : uint32_t
    {
        EXECUTORCH_PAL_LOG_LEVEL_DEBUG = 'D',
        EXECUTORCH_PAL_LOG_LEVEL_INFO = 'I',
        EXECUTORCH_PAL_LOG_LEVEL_ERROR = 'E',
        EXECUTORCH_PAL_LOG_LEVEL_FATAL = 'F',
        EXECUTORCH_PAL_LOG_LEVEL_UNKNOWN = '?', // Exception to the "uppercase letter" rule.
    };

    struct ExecutorchPalImpl
    {
        void (*init)();
        void (*abort)();
        executorch_timestamp_t (*current_ticks)();
        struct executorch_tick_ratio (*ticks_to_ns_multiplier)();
        void (*emit_log_message)(
            executorch_timestamp_t timestamp,
            enum executorch_pal_log_level level,
            const char *filename,
            const char *function,
            size_t line,
            const char *message,
            size_t length);
        void *(*allocate)(size_t size);
        void (*free)(void *ptr);

        // An optional metadata field, indicating the name of the source
        // file that registered the PAL implementation.
        const char *source_filename;
    };

    /**
     * Override the PAL functions with user implementations. Any null entries in the
     * table are unchanged and will keep the default implementation.
     *
     * Returns true if the registration was successful, false otherwise.
     */
    bool executorch_register_pal(struct ExecutorchPalImpl impl);

    /**
     * Returns the PAL function table, which contains function pointers to the
     * active implementation of each PAL function.
     */
    const struct ExecutorchPalImpl *executorch_get_pal_impl();

    /**
     * Initialize the platform abstraction layer.
     *
     * This function should be called before any other function provided by the PAL
     * to initialize any global state. Typically overridden by PAL implementer.
     */
    void executorch_pal_init();

    /**
     * Immediately abort execution, setting the device into an error state, if
     * available.
     */
    void executorch_pal_abort();

    /**
     * Return a monotonically non-decreasing timestamp in system ticks.
     *
     * @retval Timestamp value in system ticks.
     */
    executorch_timestamp_t executorch_pal_current_ticks();

    /**
     * Return the conversion rate from system ticks to nanoseconds as a fraction.
     * To convert a system ticks to nanoseconds, multiply the tick count by the
     * numerator and then divide by the denominator:
     *   nanoseconds = ticks * numerator / denominator
     *
     * The utility method executorch::runtime::ticks_to_ns(executorch_timestamp_t) can also
     * be used to perform the conversion for a given tick count. It is defined in
     * torch/executor/runtime/platform/clock.h.
     *
     * @retval The ratio of nanoseconds to system ticks.
     */
    struct executorch_tick_ratio executorch_pal_ticks_to_ns_multiplier();

    /**
     * Severity level of a log message. Values must map to printable 7-bit ASCII
     * uppercase letters.
     */
    void executorch_pal_emit_log_message(
        executorch_timestamp_t timestamp,
        enum executorch_pal_log_level level,
        const char *filename,
        const char *function,
        size_t line,
        const char *message,
        size_t length);

    /**
     * NOTE: Core runtime code must not call this directly. It may only be called by
     * a MemoryAllocator wrapper.
     *
     * Allocates size bytes of memory.
     *
     * @param[in] size Number of bytes to allocate.
     * @returns the allocated memory, or nullptr on failure. Must be freed using
     *     et_pal_free().
     */
    void *executorch_pal_allocate(size_t size);

    /**
     * Frees memory allocated by et_pal_allocate().
     *
     * @param[in] ptr Pointer to memory to free. May be nullptr.
     */
    void executorch_pal_free(void *ptr);

#ifdef __cplusplus
} // end of extern "C" block
#endif
