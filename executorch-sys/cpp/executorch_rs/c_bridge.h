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
    typedef const void *EValue;
    typedef void *EValueMut;
    struct TensorStorage
    {
        size_t _blob[1];
    };
    typedef const void *Tensor;
    typedef void *TensorMut;
    struct TensorImpl
    {
        size_t _blob[8];
    };
    struct Program
    {
        size_t _blob[11];
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
        size_t _blob[14];
    };

    typedef void *DataLoaderMut;
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
        char _blob1[1];
        size_t _blob2[1];
        bool _blob3[1];
    };
    typedef const void *OptionalTensor;
    typedef void *OptionalTensorMut;

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
        EValueMut data;
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
        Tensor data;
        size_t len;
    };
    struct ArrayRefOptionalTensor
    {
        OptionalTensor data;
        size_t len;
    };
    struct ArrayRefEValue
    {
        EValue data;
        size_t len;
    };
    struct ArrayRefEValuePtr
    {
        const EValue *data;
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
        TensorMut data;
        size_t len;
    };
    struct SpanOptionalTensor
    {
        OptionalTensorMut data;
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
    typedef void *EventTracer;

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
    DataLoaderMut executorch_BufferDataLoader_as_data_loader(struct BufferDataLoader *self);
#if defined(EXECUTORCH_RS_DATA_LOADER)
    enum Error executorch_FileDataLoader_new(const char *file_path, size_t alignment, struct FileDataLoader *out);
    void executorch_FileDataLoader_destructor(struct FileDataLoader *self);
    DataLoaderMut executorch_FileDataLoader_as_data_loader(struct FileDataLoader *self);
    enum Error executorch_MmapDataLoader_new(const char *file_path, enum MmapDataLoaderMlockConfig mlock_config, struct MmapDataLoader *out);
    void executorch_MmapDataLoader_destructor(struct MmapDataLoader *self);
    DataLoaderMut executorch_MmapDataLoader_as_data_loader(struct MmapDataLoader *self);

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
    void executorch_Tensor_new(TensorMut self, struct TensorImpl *tensor_impl);
    size_t executorch_Tensor_nbytes(Tensor self);
    size_t executorch_Tensor_size(Tensor self, size_t dim);
    size_t executorch_Tensor_dim(Tensor self);
    size_t executorch_Tensor_numel(Tensor self);
    enum ScalarType executorch_Tensor_scalar_type(Tensor self);
    size_t executorch_Tensor_element_size(Tensor self);
    struct ArrayRefSizesType executorch_Tensor_sizes(Tensor self);
    struct ArrayRefDimOrderType executorch_Tensor_dim_order(Tensor self);
    struct ArrayRefStridesType executorch_Tensor_strides(Tensor self);
    const void *executorch_Tensor_const_data_ptr(Tensor self);
    void *executorch_Tensor_mutable_data_ptr(Tensor self);
    int64_t executorch_Tensor_coordinate_to_index(Tensor self, struct ArrayRefUsizeType coordinate);
    void executorch_Tensor_destructor(TensorMut self);

    // OptionalTensor
    Tensor executorch_OptionalTensor_get(OptionalTensor self);

    // EValue
    void executorch_EValue_new_none(EValueMut self);
    void executorch_EValue_new_from_i64(EValueMut self, int64_t value);
    void executorch_EValue_new_from_i64_list(EValueMut self, struct BoxedEvalueListI64 value);
    void executorch_EValue_new_from_f64(EValueMut self, double value);
    void executorch_EValue_new_from_f64_list(EValueMut self, struct ArrayRefF64 value);
    void executorch_EValue_new_from_bool(EValueMut self, bool value);
    void executorch_EValue_new_from_bool_list(EValueMut self, struct ArrayRefBool value);
    void executorch_EValue_new_from_string(EValueMut self, struct ArrayRefChar value);
    void executorch_EValue_new_from_tensor(EValueMut self, Tensor value);
    void executorch_EValue_new_from_tensor_list(EValueMut self, struct BoxedEvalueListTensor value);
    void executorch_EValue_new_from_optional_tensor_list(EValueMut self, struct BoxedEvalueListOptionalTensor value);
    enum Tag executorch_EValue_tag(EValue self);
    int64_t executorch_EValue_as_i64(EValue self);
    struct ArrayRefI64 executorch_EValue_as_i64_list(EValue self);
    double executorch_EValue_as_f64(EValue self);
    struct ArrayRefF64 executorch_EValue_as_f64_list(EValue self);
    bool executorch_EValue_as_bool(EValue self);
    struct ArrayRefBool executorch_EValue_as_bool_list(EValue self);
    struct ArrayRefChar executorch_EValue_as_string(EValue self);
    Tensor executorch_EValue_as_tensor(EValue self);
    struct ArrayRefTensor executorch_EValue_as_tensor_list(EValue self);
    struct ArrayRefOptionalTensor executorch_EValue_as_optional_tensor_list(EValue self);
    void executorch_EValue_copy(EValue src, EValueMut dst);
    void executorch_EValue_destructor(EValueMut self);
    void executorch_EValue_move(EValueMut src, EValueMut dst);

    // Program
    enum ProgramHeaderStatus executorch_Program_check_header(const void *data, size_t size);
    enum Error executorch_Program_load(DataLoaderMut loader, enum ProgramVerification verification, struct Program *out);
    enum Error executorch_Program_load_method(const struct Program *self, const char *method_name, struct MemoryManager *memory_manager, EventTracer event_tracer, struct Method *out);
    enum Error executorch_Program_get_method_name(const struct Program *self, size_t method_index, const char **out);
    enum Error executorch_Program_method_meta(const struct Program *self, const char *method_name, struct MethodMeta *method_meta_out);
    size_t executorch_Program_num_methods(const struct Program *self);
    void executorch_Program_destructor(struct Program *self);

    // MethodMeta
    size_t executorch_Method_inputs_size(const struct Method *self);
    size_t executorch_Method_outputs_size(const struct Method *self);
    enum Error executorch_Method_set_input(struct Method *self, EValue input_evalue, size_t input_idx);
    EValue executorch_Method_get_output(const struct Method *self, size_t i);
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

    // TensorInfo
    struct ArrayRefI32 executorch_TensorInfo_sizes(const struct TensorInfo *self);
    struct ArrayRefU8 executorch_TensorInfo_dim_order(const struct TensorInfo *self);
    enum ScalarType executorch_TensorInfo_scalar_type(const struct TensorInfo *self);
    size_t executorch_TensorInfo_nbytes(const struct TensorInfo *self);

#ifdef __cplusplus
} // end of extern "C" block
#endif
