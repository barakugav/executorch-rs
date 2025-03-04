mod c_link {
    #![allow(dead_code)]
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

impl Copy for Tag {}
impl Copy for ScalarType {}

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
impl_ref_clone_copy!(EValueRef);
impl_ref_clone_copy!(EValueRefMut);
impl_ref_clone_copy!(TensorRef);
impl_ref_clone_copy!(TensorRefMut);
impl_ref_clone_copy!(OptionalTensorRef);
impl_ref_clone_copy!(OptionalTensorRefMut);
impl_ref_clone_copy!(DataLoaderRefMut);
impl_ref_clone_copy!(EventTracerRefMut);
