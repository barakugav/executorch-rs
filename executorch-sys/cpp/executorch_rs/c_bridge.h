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
     * ExecuTorch ET_Error type.
     */
    enum ET_Error : uint32_t
    {
        /*
         * System errors.
         */

        /// Status indicating a successful operation.
        ET_Error_Ok = 0x00,

        /// An internal error occurred.
        ET_Error_Internal = 0x01,

        /// Status indicating the executor is in an invalid state for a target
        /// operation
        ET_Error_InvalidState = 0x02,

        /// Status indicating there are no more steps of execution to run
        ET_Error_EndOfMethod = 0x03,

        /// Status indicating a resource has already been loaded.
        ET_Error_AlreadyLoaded = 0x04,

        /*
         * Logical errors.
         */

        /// Operation is not supported in the current context.
        ET_Error_NotSupported = 0x10,

        /// Operation is not yet implemented.
        ET_Error_NotImplemented = 0x11,

        /// User provided an invalid argument.
        ET_Error_InvalidArgument = 0x12,

        /// Object is an invalid type for the operation.
        ET_Error_InvalidType = 0x13,

        /// Operator(s) missing in the operator registry.
        ET_Error_OperatorMissing = 0x14,

        /// Registration error: Exceeding the maximum number of kernels.
        ET_Error_RegistrationExceedingMaxKernels = 0x15,

        /// Registration error: The kernel is already registered.
        ET_Error_RegistrationAlreadyRegistered = 0x16,

        /*
         * Resource errors.
         */

        /// Requested resource could not be found.
        ET_Error_NotFound = 0x20,

        /// Could not allocate the requested memory.
        ET_Error_MemoryAllocationFailed = 0x21,

        /// Could not access a resource.
        ET_Error_AccessFailed = 0x22,

        /// ET_Error caused by the contents of a program.
        ET_Error_InvalidProgram = 0x23,

        /// ET_Error caused by the contents of external data.
        ET_Error_InvalidExternalData = 0x24,

        /// Does not have enough resources to perform the requested operation.
        ET_Error_OutOfResources = 0x25,

        /*
         * Delegate errors.
         */

        /// Init stage: Backend receives an incompatible delegate version.
        ET_Error_DelegateInvalidCompatibility = 0x30,
        /// Init stage: Backend fails to allocate memory.
        ET_Error_DelegateMemoryAllocationFailed = 0x31,
        /// Execute stage: The handle is invalid.
        ET_Error_DelegateInvalidHandle = 0x32,

    };

    /**
     * Describes the presence of an ExecuTorch program header.
     */
    enum ET_ProgramHeaderStatus : uint32_t
    {
        /**
         * An ExecuTorch program header is present, and its version is compatible
         * with this version of the runtime.
         */
        ET_ProgramHeaderStatus_CompatibleVersion,

        /**
         * An ExecuTorch program header is present, but its version is not
         * compatible with this version of the runtime.
         */
        ET_ProgramHeaderStatus_IncompatibleVersion,

        /**
         * An ExecuTorch program header is not present.
         */
        ET_ProgramHeaderStatus_NotPresent,

        /**
         * The data provided was too short to find the program header.
         */
        ET_ProgramHeaderStatus_ShortData,
    };

    /**
     * Types of validation that the ET_Program can do before parsing the data.
     */
    enum ET_ProgramVerification : uint8_t
    {
        /**
         * Do minimal verification of the data, ensuring that the header appears
         * correct.
         *
         * Has minimal runtime overhead.
         */
        ET_ProgramVerification_Minimal,
        /**
         * Do full verification of the data, ensuring that internal pointers are
         * self-consistent and that the data has not been truncated or obviously
         * corrupted. May not catch all types of corruption, but should guard
         * against illegal memory operations during parsing.
         *
         * Will have higher runtime overhead, scaling with the complexity of the
         * proram data.
         */
        ET_ProgramVerification_InternalConsistency,
    };

    /**
     * Describes how and whether to lock loaded pages with `mlock()`.
     *
     * Using `mlock()` typically loads all of the pages immediately, and will
     * typically ensure that they are not swapped out. The actual behavior
     * will depend on the host system.
     */
    enum ET_MmapDataLoaderMlockConfig : uint32_t
    {
        /// Do not call `mlock()` on loaded pages.
        ET_MmapDataLoaderMlockConfig_NoMlock,
        /// Call `mlock()` on loaded pages, failing if it fails.
        ET_MmapDataLoaderMlockConfig_UseMlock,
        /// Call `mlock()` on loaded pages, ignoring errors if it fails.
        ET_MmapDataLoaderMlockConfig_UseMlockIgnoreErrors,
        /// Use madvise(MADV_WILLNEED | MADV_SEQUENTIAL) instead of mlock.
        /// Tells the kernel to prefetch pages eagerly and optimize for
        /// sequential reads, without pinning them in RAM.
        ET_MmapDataLoaderMlockConfig_UseMadvise,
    };

    /**
     * Enum to define loading behavior.
     */
    enum ET_ModuleLoadMode : uint32_t
    {
        /// Load the whole file as a buffer.
        ET_ModuleLoadMode_File,
        /// Use mmap to load pages into memory.
        ET_ModuleLoadMode_Mmap,
        /// Use memory locking and handle errors.
        ET_ModuleLoadMode_MmapUseMlock,
        /// Use memory locking and ignore errors.
        ET_ModuleLoadMode_MmapUseMlockIgnoreErrors,
        /// Use mmap with madvise(MADV_WILLNEED | MADV_SEQUENTIAL) hints.
        ET_ModuleLoadMode_MmapUseMadvise,
    };

    enum ET_Tag : uint32_t
    {
        ET_Tag_None,
        ET_Tag_Tensor,
        ET_Tag_String,
        ET_Tag_Double,
        ET_Tag_Int,
        ET_Tag_Bool,
        ET_Tag_ListBool,
        ET_Tag_ListDouble,
        ET_Tag_ListInt,
        ET_Tag_ListTensor,
        ET_Tag_ListScalar,
        ET_Tag_ListOptionalTensor,
    };

    enum ET_ScalarType : int8_t
    {
        ET_ScalarType_Byte,
        ET_ScalarType_Char,
        ET_ScalarType_Short,
        ET_ScalarType_Int,
        ET_ScalarType_Long,
        ET_ScalarType_Half,
        ET_ScalarType_Float,
        ET_ScalarType_Double,
        ET_ScalarType_ComplexHalf,
        ET_ScalarType_ComplexFloat,
        ET_ScalarType_ComplexDouble,
        ET_ScalarType_Bool,
        ET_ScalarType_QInt8,
        ET_ScalarType_QUInt8,
        ET_ScalarType_QInt32,
        ET_ScalarType_BFloat16,
        ET_ScalarType_QUInt4x2,
        ET_ScalarType_QUInt2x4,
        ET_ScalarType_Bits1x8,
        ET_ScalarType_Bits2x4,
        ET_ScalarType_Bits4x2,
        ET_ScalarType_Bits8,
        ET_ScalarType_Bits16,
        ET_ScalarType_Float8_e5m2,
        ET_ScalarType_Float8_e4m3fn,
        ET_ScalarType_Float8_e5m2fnuz,
        ET_ScalarType_Float8_e4m3fnuz,
        ET_ScalarType_UInt16,
        ET_ScalarType_UInt32,
        ET_ScalarType_UInt64,
    };

    /// Represents the type of compute device.
    /// Note: ExecuTorch Device is distinct from PyTorch Device.
    enum ET_DeviceType : int8_t
    {
        ET_DeviceType_CPU = 0,
        ET_DeviceType_CUDA = 1,
    };

    /**
     * The type used for elements of `Tensor.sizes()`.
     */
    typedef int32_t ET_SizesType;

    /**
     * The type used for elements of `Tensor.dim_order()`.
     */
    typedef uint8_t ET_DimOrderType;

    /**
     * The type used for elements of `Tensor.strides()`.
     */
    typedef int32_t ET_StridesType;

    /**
     * The resizing capabilities of a Tensor.
     *
     * The rank of an ExecuTorch Tensors can never change, but shape sometimes can.
     */
    enum ET_TensorShapeDynamism : uint8_t
    {
        /// Cannot change shape.
        ET_TensorShapeDynamism_STATIC = 0,
        /// Shape cannot exceed initial capacity.
        ET_TensorShapeDynamism_DYNAMIC_BOUND = 1,
        /// No restriction on shape and capacity.
        ET_TensorShapeDynamism_DYNAMIC_UNBOUND = 2,
    };

    struct ET_TensorStorage
    {
        size_t _blob[1];
    };
    struct ET_TensorRef
    {
        const void *ptr;
    };
    struct ET_TensorRefMut
    {
        void *ptr;
    };
    struct ET_TensorImpl
    {
        size_t _blob[8];
    };
    struct ET_EValueStorage
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
            struct ET_TensorStorage _blob10;
        };
        // tag
        uint32_t _blob11;
    };
    struct ET_EValueRef
    {
        const void *ptr;
    };
    struct ET_EValueRefMut
    {
        void *ptr;
    };
    struct ET_FreeableBuffer
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
    struct ET_Program
    {

        // program_data_
        struct ET_FreeableBuffer _blob1;
        // loader_
        // internal_program_
        // segment_base_offset_
        size_t _blob2[3];
        // constant_segment_data_
        struct ET_FreeableBuffer _blob3;
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
    struct ET_Device
    {
        enum ET_DeviceType type;
        int8_t index;
    };

    struct ET_BackendOption
    {
        char _blob1[64];
        struct // OptionValue
        {
            union
            {
                bool _blob2;
                int _blob3;
                char _blob4[256];
            };
            uint8_t _blob5; // tag
        };
    };
    struct ET_LoadBackendOptionsMap
    {

        struct
        {
            char _blob1[64];
            size_t _blob2[2];
        } _blob3[8]; // entries
        size_t _blob4;
    };
    struct ET_TensorInfo
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
    struct ET_MethodMeta
    {
        size_t _blob[1];
    };
    struct ET_Method
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
        // kernel_registry_ (2)
        size_t _blob1[19];
        // init_state_;
        uint8_t _blob2[1];
    };

    struct ET_TensorLayout
    {
        // sizes_ (2)
        // dim_order_ (2)
        size_t _blob1[4];
        // scalar_type_
        int8_t _blob2;
        // nbytes_
        size_t _blob3;
    };

    struct ET_DataLoaderRefMut
    {
        void *ptr;
    };
    struct ET_NamedDataMapRef
    {
        const void *ptr;
    };
    struct ET_NamedDataMapRefMut
    {
        void *ptr;
    };
#if defined(EXECUTORCH_RS_FLAT_TENSOR)
    struct ET_FlatTensorDataMap
    {
        // vtable
        size_t _blob0[1];
        // header_
        uint64_t _blob1[4];
        // flat_tensor_data_
        struct ET_FreeableBuffer _blob2;
        // flat_tensor_
        // loader_
        size_t _blob3[2];
    };
#endif

    struct ET_BufferDataLoader
    {
        size_t _blob[3];
    };
#if defined(EXECUTORCH_RS_DATA_LOADER)
    struct ET_FileDataLoader
    {
        size_t _blob[5];
    };
    struct ET_MmapDataLoader
    {
        size_t _blob_1[4];
        int _blob_2[2];
    };
#endif

    struct ET_MemoryAllocator
    {
        size_t _blob_1[4];
        uint32_t _blob_2[2];
    };
    struct ET_HierarchicalAllocator
    {
        size_t _blob[34];
    };
    struct ET_MemoryManager
    {
        size_t _blob[3];
    };

    struct ET_OptionalTensorStorage
    {
        union
        {
            char _dummy;
            struct ET_TensorStorage _val;
        };
        bool _flag;
    };
    struct ET_OptionalTensorRef
    {
        const void *ptr;
    };
    struct ET_OptionalTensorRefMut
    {
        void *ptr;
    };

#if defined(EXECUTORCH_RS_STD)
    struct ET_VecChar
    {
        char *data;
        size_t len;
        size_t cap;
    };
    void executorch_VecChar_destructor(struct ET_VecChar *vec);

    struct ET_VecVecChar
    {
        struct ET_VecChar *data;
        size_t len;
        size_t cap;
    };
    void executorch_VecVecChar_destructor(struct ET_VecVecChar *vec);

    struct ET_VecEValue
    {
        struct ET_EValueRefMut data;
        size_t len;
        size_t cap;
    };
    void executorch_VecEValue_destructor(struct ET_VecEValue *vec);
#endif

    struct ET_ArrayRefChar
    {
        const char *data;
        size_t len;
    };
    struct ET_ArrayRefBool
    {
        const bool *data;
        size_t len;
    };
    struct ET_ArrayRefU8
    {
        const uint8_t *data;
        size_t len;
    };
    struct ET_ArrayRefI32
    {
        const int32_t *data;
        size_t len;
    };
    struct ET_ArrayRefI64
    {
        const int64_t *data;
        size_t len;
    };
    struct ET_ArrayRefF64
    {
        const double *data;
        size_t len;
    };
    struct ET_ArrayRefUsizeType
    {
        const size_t *data;
        size_t len;
    };
    struct ET_ArrayRefSizesType
    {
        const ET_SizesType *data;
        size_t len;
    };
    struct ET_ArrayRefDimOrderType
    {
        const ET_DimOrderType *data;
        size_t len;
    };
    struct ET_ArrayRefStridesType
    {
        const ET_StridesType *data;
        size_t len;
    };
    struct ET_ArrayRefTensor
    {
        struct ET_TensorRef data;
        size_t len;
    };
    struct ET_ArrayRefOptionalTensor
    {
        struct ET_OptionalTensorRef data;
        size_t len;
    };
    struct ET_ArrayRefEValue
    {
        struct ET_EValueRef data;
        size_t len;
    };
    struct ET_ArrayRefEValuePtr
    {
        const struct ET_EValueRef *data;
        size_t len;
    };
    struct ET_SpanU8
    {
        uint8_t *data;
        size_t len;
    };
    struct ET_SpanSpanU8
    {
        struct ET_SpanU8 *data;
        size_t len;
    };
    struct ET_SpanI64
    {
        int64_t *data;
        size_t len;
    };
    struct ET_SpanTensor
    {
        struct ET_TensorRefMut data;
        size_t len;
    };
    struct ET_SpanOptionalTensor
    {
        struct ET_OptionalTensorRefMut data;
        size_t len;
    };
    // struct SpanEValue
    // {
    //     EValue *data;
    //     size_t len;
    // };
    struct ET_BoxedEvalueListI64
    {
        struct ET_ArrayRefEValuePtr wrapped_vals;
        int64_t *unwrapped_vals;
    };
    struct ET_BoxedEvalueListTensor
    {
        struct ET_ArrayRefEValuePtr wrapped_vals;
        struct ET_TensorRefMut unwrapped_vals;
    };
    struct ET_BoxedEvalueListOptionalTensor
    {
        struct ET_ArrayRefEValuePtr wrapped_vals;
        struct ET_OptionalTensorRefMut unwrapped_vals;
    };

    struct ET_EventTracerRefMut
    {
        void *ptr;
    };
#if defined(EXECUTORCH_RS_ETDUMP)
    struct ET_DumpGen
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

    struct ET_MemoryAllocator executorch_MemoryAllocator_new(uint32_t size, uint8_t *base_address);
    void *executorch_MemoryAllocator_allocate(struct ET_MemoryAllocator *self, size_t size, size_t alignment);
    struct ET_HierarchicalAllocator executorch_HierarchicalAllocator_new(struct ET_SpanSpanU8 buffers);
    void executorch_HierarchicalAllocator_destructor(struct ET_HierarchicalAllocator *self);
    struct ET_MemoryManager executorch_MemoryManager_new(
        struct ET_MemoryAllocator *method_allocator,
        struct ET_HierarchicalAllocator *planned_memory,
        struct ET_MemoryAllocator *temp_allocator);

    // Loaders
    struct ET_BufferDataLoader executorch_BufferDataLoader_new(const void *data, size_t size);
    struct ET_DataLoaderRefMut executorch_BufferDataLoader_as_data_loader_mut(struct ET_BufferDataLoader *self);
#if defined(EXECUTORCH_RS_DATA_LOADER)
    enum ET_Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, struct ET_FileDataLoader *out);
    void executorch_FileDataLoader_destructor(struct ET_FileDataLoader *self);
    struct ET_DataLoaderRefMut executorch_FileDataLoader_as_data_loader_mut(struct ET_FileDataLoader *self);
    enum ET_Error executorch_MmapDataLoader_new(const char *file_path, enum ET_MmapDataLoaderMlockConfig mlock_config, struct ET_MmapDataLoader *out);
    void executorch_MmapDataLoader_destructor(struct ET_MmapDataLoader *self);
    struct ET_DataLoaderRefMut executorch_MmapDataLoader_as_data_loader_mut(struct ET_MmapDataLoader *self);

#endif

    bool executorch_is_valid_dim_order_and_strides(size_t dim, const ET_SizesType *sizes, const ET_DimOrderType *dim_order, const ET_StridesType *strides);
    enum ET_Error executorch_stride_to_dim_order(const ET_StridesType *strides, size_t dims, ET_DimOrderType *dim_order);

    // NamedDataMap
    enum ET_Error executorch_NamedDataMap_get_tensor_layout(
        struct ET_NamedDataMapRef self,
        struct ET_ArrayRefChar key,
        struct ET_TensorLayout *out);
    enum ET_Error executorch_NamedDataMap_get_num_keys(struct ET_NamedDataMapRef self, uint32_t *out);
    enum ET_Error executorch_NamedDataMap_get_key(
        struct ET_NamedDataMapRef self,
        uint32_t index,
        const char **out_data);

#if defined(EXECUTORCH_RS_FLAT_TENSOR)
    // ET_FlatTensorDataMap
    enum ET_Error executorch_FlatTensorDataMap_load(struct ET_DataLoaderRefMut loader, struct ET_FlatTensorDataMap *out);
    struct ET_NamedDataMapRefMut executorch_FlatTensorDataMap_as_named_data_map_mut(struct ET_FlatTensorDataMap *self);
#endif

    // Tensor
    void executorch_TensorImpl_new(
        struct ET_TensorImpl *self,
        enum ET_ScalarType type,
        size_t dim,
        ET_SizesType *sizes,
        void *data,
        ET_DimOrderType *dim_order,
        ET_StridesType *strides,
        enum ET_TensorShapeDynamism dynamism);
    void executorch_Tensor_new(struct ET_TensorRefMut self, struct ET_TensorImpl *tensor_impl);
    size_t executorch_Tensor_nbytes(struct ET_TensorRef self);
    size_t executorch_Tensor_size(struct ET_TensorRef self, size_t dim);
    size_t executorch_Tensor_dim(struct ET_TensorRef self);
    size_t executorch_Tensor_numel(struct ET_TensorRef self);
    enum ET_ScalarType executorch_Tensor_scalar_type(struct ET_TensorRef self);
    struct ET_Device executorch_Tensor_device(struct ET_TensorRef self);
    size_t executorch_Tensor_element_size(struct ET_TensorRef self);
    struct ET_ArrayRefSizesType executorch_Tensor_sizes(struct ET_TensorRef self);
    struct ET_ArrayRefDimOrderType executorch_Tensor_dim_order(struct ET_TensorRef self);
    struct ET_ArrayRefStridesType executorch_Tensor_strides(struct ET_TensorRef self);
    const void *executorch_Tensor_const_data_ptr(struct ET_TensorRef self);
    void *executorch_Tensor_mutable_data_ptr(struct ET_TensorRef self);
    int64_t executorch_Tensor_coordinate_to_index(struct ET_TensorRef self, struct ET_ArrayRefUsizeType coordinate);
    int64_t executorch_Tensor_coordinate_to_index_unchecked(struct ET_TensorRef self, struct ET_ArrayRefUsizeType coordinate);
    void executorch_Tensor_destructor(struct ET_TensorRefMut self);

    // OptionalTensor
    struct ET_TensorRef executorch_OptionalTensor_get(struct ET_OptionalTensorRef self);

    // ET_TensorLayout
    // enum ET_Error executorch_TensorLayout_create(
    //     struct ET_ArrayRefI32 sizes,
    //     struct ET_ArrayRefU8 dim_order,
    //     enum ET_ScalarType scalar_type,
    //     struct ET_TensorLayout *out);
    struct ET_ArrayRefI32 executorch_TensorLayout_sizes(const struct ET_TensorLayout *self);
    struct ET_ArrayRefU8 executorch_TensorLayout_dim_order(const struct ET_TensorLayout *self);
    enum ET_ScalarType executorch_TensorLayout_scalar_type(const struct ET_TensorLayout *self);
    size_t executorch_TensorLayout_nbytes(const struct ET_TensorLayout *self);

    // EValue
    void executorch_EValue_new_none(struct ET_EValueRefMut self);
    void executorch_EValue_new_from_i64(struct ET_EValueRefMut self, int64_t value);
    void executorch_EValue_new_from_i64_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListI64 *value);
    void executorch_EValue_new_from_f64(struct ET_EValueRefMut self, double value);
    void executorch_EValue_new_from_f64_list(struct ET_EValueRefMut self, const struct ET_ArrayRefF64 *value);
    void executorch_EValue_new_from_bool(struct ET_EValueRefMut self, bool value);
    void executorch_EValue_new_from_bool_list(struct ET_EValueRefMut self, const struct ET_ArrayRefBool *value);
    void executorch_EValue_new_from_string(struct ET_EValueRefMut self, const struct ET_ArrayRefChar *value);
    void executorch_EValue_new_from_tensor(struct ET_EValueRefMut self, struct ET_TensorRef value);
    void executorch_EValue_new_from_tensor_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListTensor *value);
    void executorch_EValue_new_from_optional_tensor_list(struct ET_EValueRefMut self, const struct ET_BoxedEvalueListOptionalTensor *value);
    enum ET_Tag executorch_EValue_tag(struct ET_EValueRef self);
    int64_t executorch_EValue_as_i64(struct ET_EValueRef self);
    struct ET_ArrayRefI64 executorch_EValue_as_i64_list(struct ET_EValueRef self);
    double executorch_EValue_as_f64(struct ET_EValueRef self);
    struct ET_ArrayRefF64 executorch_EValue_as_f64_list(struct ET_EValueRef self);
    bool executorch_EValue_as_bool(struct ET_EValueRef self);
    struct ET_ArrayRefBool executorch_EValue_as_bool_list(struct ET_EValueRef self);
    struct ET_ArrayRefChar executorch_EValue_as_string(struct ET_EValueRef self);
    struct ET_TensorRef executorch_EValue_as_tensor(struct ET_EValueRef self);
    struct ET_ArrayRefTensor executorch_EValue_as_tensor_list(struct ET_EValueRef self);
    struct ET_ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(struct ET_EValueRef self);
    void executorch_EValue_copy(struct ET_EValueRef src, struct ET_EValueRefMut dst);
    void executorch_EValue_destructor(struct ET_EValueRefMut self);
    void executorch_EValue_move(struct ET_EValueRefMut src, struct ET_EValueRefMut dst);

    // ET_Program
    enum ET_ProgramHeaderStatus executorch_Program_check_header(const void *data, size_t size);
    enum ET_Error executorch_Program_load(struct ET_DataLoaderRefMut loader, enum ET_ProgramVerification verification, struct ET_Program *out);
    enum ET_Error executorch_Program_load_method(
        const struct ET_Program *self,
        const char *method_name,
        struct ET_MemoryManager *memory_manager,
        struct ET_EventTracerRefMut event_tracer,
        struct ET_NamedDataMapRef named_data_map,
        const struct ET_LoadBackendOptionsMap *backend_options,
        struct ET_Method *out);
    enum ET_Error executorch_Program_get_method_name(const struct ET_Program *self, size_t method_index, const char **out);
    enum ET_Error executorch_Program_get_named_data_map(const struct ET_Program *self, struct ET_NamedDataMapRef *out);
    enum ET_Error executorch_Program_method_meta(const struct ET_Program *self, const char *method_name, struct ET_MethodMeta *method_meta_out);
    size_t executorch_Program_num_methods(const struct ET_Program *self);
    void executorch_Program_destructor(struct ET_Program *self);

    // ET_MethodMeta
    size_t executorch_Method_inputs_size(const struct ET_Method *self);
    size_t executorch_Method_outputs_size(const struct ET_Method *self);
    enum ET_Error executorch_Method_set_input(struct ET_Method *self, struct ET_EValueRef input_evalue, size_t input_idx);
    struct ET_EValueRef executorch_Method_get_output(const struct ET_Method *self, size_t i);
    enum ET_Error executorch_Method_get_attribute(struct ET_Method *self, struct ET_ArrayRefChar name, struct ET_TensorRefMut out);
    enum ET_Error executorch_Method_execute(struct ET_Method *self);
    void executorch_Method_destructor(struct ET_Method *self);
    const char *executorch_MethodMeta_name(const struct ET_MethodMeta *self);
    size_t executorch_MethodMeta_num_inputs(const struct ET_MethodMeta *self);
    size_t executorch_MethodMeta_num_outputs(const struct ET_MethodMeta *self);
    size_t executorch_MethodMeta_num_memory_planned_buffers(const struct ET_MethodMeta *self);
    enum ET_Error executorch_MethodMeta_input_tag(const struct ET_MethodMeta *self, size_t index, enum ET_Tag *tag_out);
    enum ET_Error executorch_MethodMeta_output_tag(const struct ET_MethodMeta *self, size_t index, enum ET_Tag *tag_out);
    enum ET_Error executorch_MethodMeta_input_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out);
    enum ET_Error executorch_MethodMeta_output_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out);
    size_t executorch_MethodMeta_num_attributes(const struct ET_MethodMeta *self);
    enum ET_Error executorch_MethodMeta_attribute_tensor_meta(const struct ET_MethodMeta *self, size_t index, struct ET_TensorInfo *tensor_info_out);
    enum ET_Error executorch_MethodMeta_memory_planned_buffer_size(const struct ET_MethodMeta *self, size_t index, int64_t *size_out);
    enum ET_Error executorch_MethodMeta_memory_planned_buffer_device(const struct ET_MethodMeta *self, size_t index, struct ET_Device *device_out);
    bool executorch_MethodMeta_uses_backend(const struct ET_MethodMeta *self, const char *backend_name);
    size_t executorch_MethodMeta_num_backends(const struct ET_MethodMeta *self);
    enum ET_Error executorch_MethodMeta_get_backend_name(const struct ET_MethodMeta *self, size_t index, const char **backend_name_out);

    // ET_BackendOption
    enum ET_Error executorch_BackendOption_new_bool(struct ET_ArrayRefChar key, bool value, struct ET_BackendOption *out);
    enum ET_Error executorch_BackendOption_new_int(struct ET_ArrayRefChar key, int value, struct ET_BackendOption *out);
    enum ET_Error executorch_BackendOption_new_str(struct ET_ArrayRefChar key, struct ET_ArrayRefChar value, struct ET_BackendOption *out);
    const char *executorch_BackendOption_key(const struct ET_BackendOption *self);
    bool executorch_BackendOption_is_bool(const struct ET_BackendOption *self);
    bool executorch_BackendOption_is_int(const struct ET_BackendOption *self);
    bool executorch_BackendOption_is_str(const struct ET_BackendOption *self);
    enum ET_Error executorch_BackendOption_as_bool(const struct ET_BackendOption *self, bool *out);
    enum ET_Error executorch_BackendOption_as_int(const struct ET_BackendOption *self, int *out);
    enum ET_Error executorch_BackendOption_as_str(const struct ET_BackendOption *self, const char **out);

    // ET_LoadBackendOptionsMap
    void executorch_LoadBackendOptionsMap_new(struct ET_LoadBackendOptionsMap *out);
    enum ET_Error executorch_LoadBackendOptionsMap_set_options(struct ET_LoadBackendOptionsMap *self, struct ET_ArrayRefChar backend_id, const struct ET_BackendOption *options, size_t n_options);
    size_t executorch_LoadBackendOptionsMap_size(const struct ET_LoadBackendOptionsMap *self);
    enum ET_Error executorch_LoadBackendOptionsMap_entry_at(const struct ET_LoadBackendOptionsMap *self, size_t index, const char **backend_id_out, const struct ET_BackendOption **options_out, size_t *n_options_out);

    // ET_TensorInfo
    struct ET_ArrayRefI32 executorch_TensorInfo_sizes(const struct ET_TensorInfo *self);
    struct ET_ArrayRefU8 executorch_TensorInfo_dim_order(const struct ET_TensorInfo *self);
    enum ET_ScalarType executorch_TensorInfo_scalar_type(const struct ET_TensorInfo *self);
    bool executorch_TensorInfo_is_memory_planned(const struct ET_TensorInfo *self);
    size_t executorch_TensorInfo_nbytes(const struct ET_TensorInfo *self);
    struct ET_ArrayRefChar executorch_TensorInfo_name(const struct ET_TensorInfo *self);

#if defined(EXECUTORCH_RS_ETDUMP)
    // ET_DumpGen
    struct ET_DumpGen executorch_ETDumpGen_new(struct ET_SpanU8 buffer);
    struct ET_ArrayRefU8 executorch_ETDumpGen_get_etdump_data(struct ET_DumpGen *self);
    struct ET_EventTracerRefMut executorch_ETDumpGen_as_event_tracer_mut(struct ET_DumpGen *self);
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
     * a ET_MemoryAllocator wrapper.
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
