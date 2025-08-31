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

impl Copy for Error {}
impl Copy for ScalarType {}
impl Copy for Tag {}
impl Copy for TensorShapeDynamism {}
impl Copy for ProgramHeaderStatus {}
impl Copy for ProgramVerification {}
impl Copy for MmapDataLoaderMlockConfig {}
impl Copy for ModuleLoadMode {}

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
impl_ref_clone_copy!(EValueRef);
impl_ref_clone_copy!(EValueRefMut);
impl_ref_clone_copy!(TensorRef);
impl_ref_clone_copy!(TensorRefMut);
impl_ref_clone_copy!(OptionalTensorRef);
impl_ref_clone_copy!(OptionalTensorRefMut);
impl_ref_clone_copy!(DataLoaderRefMut);
impl_ref_clone_copy!(EventTracerRefMut);

// ArrayRef
impl_ref_clone_copy!(ArrayRefBool);
impl_ref_clone_copy!(ArrayRefChar);
impl_ref_clone_copy!(ArrayRefDimOrderType);
impl_ref_clone_copy!(ArrayRefEValue);
impl_ref_clone_copy!(ArrayRefEValuePtr);
impl_ref_clone_copy!(ArrayRefF64);
impl_ref_clone_copy!(ArrayRefI32);
impl_ref_clone_copy!(ArrayRefI64);
impl_ref_clone_copy!(ArrayRefOptionalTensor);
impl_ref_clone_copy!(ArrayRefSizesType);
impl_ref_clone_copy!(ArrayRefStridesType);
impl_ref_clone_copy!(ArrayRefTensor);
impl_ref_clone_copy!(ArrayRefU8);
impl_ref_clone_copy!(ArrayRefUsizeType);

// Span
impl_ref_clone_copy!(SpanI64);
impl_ref_clone_copy!(SpanOptionalTensor);
impl_ref_clone_copy!(SpanSpanU8);
impl_ref_clone_copy!(SpanTensor);
impl_ref_clone_copy!(SpanU8);
