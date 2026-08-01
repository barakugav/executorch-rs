mod c_link {
    #![allow(unused)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(rustdoc::invalid_html_tags)]
    #![allow(rustdoc::broken_intra_doc_links)]
    #![allow(missing_docs)]
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
pub use c_link::*;

impl Copy for ET_Error {}
impl Copy for ET_ScalarType {}
impl Copy for ET_Tag {}
impl Copy for ET_TensorShapeDynamism {}
impl Copy for ET_ProgramHeaderStatus {}
impl Copy for ET_ProgramVerification {}
impl Copy for ET_MmapDataLoaderMlockConfig {}
impl Copy for ET_ModuleLoadMode {}
impl Copy for ET_DeviceType {}

macro_rules! impl_ref_clone_copy {
    ($name:ty) => {
        impl Clone for $name {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl Copy for $name {}
    };
}

// Ref and RefMut
impl_ref_clone_copy!(ET_EValueRef);
impl_ref_clone_copy!(ET_EValueRefMut);
impl_ref_clone_copy!(ET_TensorRef);
impl_ref_clone_copy!(ET_TensorRefMut);
impl_ref_clone_copy!(ET_OptionalTensorRef);
impl_ref_clone_copy!(ET_OptionalTensorRefMut);
impl_ref_clone_copy!(ET_DataLoaderRefMut);
impl_ref_clone_copy!(ET_EventTracerRefMut);

// ArrayRef
impl_ref_clone_copy!(ET_ArrayRefBool);
impl_ref_clone_copy!(ET_ArrayRefChar);
impl_ref_clone_copy!(ET_ArrayRefDimOrderType);
impl_ref_clone_copy!(ET_ArrayRefEValue);
impl_ref_clone_copy!(ET_ArrayRefEValuePtr);
impl_ref_clone_copy!(ET_ArrayRefF64);
impl_ref_clone_copy!(ET_ArrayRefI32);
impl_ref_clone_copy!(ET_ArrayRefI64);
impl_ref_clone_copy!(ET_ArrayRefOptionalTensor);
impl_ref_clone_copy!(ET_ArrayRefSizesType);
impl_ref_clone_copy!(ET_ArrayRefStridesType);
impl_ref_clone_copy!(ET_ArrayRefTensor);
impl_ref_clone_copy!(ET_ArrayRefU8);
impl_ref_clone_copy!(ET_ArrayRefUsizeType);

// Span
impl_ref_clone_copy!(ET_SpanI64);
impl_ref_clone_copy!(ET_SpanOptionalTensor);
impl_ref_clone_copy!(ET_SpanSpanU8);
impl_ref_clone_copy!(ET_SpanTensor);
impl_ref_clone_copy!(ET_SpanU8);

impl_ref_clone_copy!(ET_Device);
