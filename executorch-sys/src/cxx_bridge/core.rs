use cxx::{type_id, ExternType};

unsafe impl ExternType for crate::ScalarType {
    type Id = type_id!("ScalarType");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::TensorShapeDynamism {
    type Id = type_id!("TensorShapeDynamism");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::Error {
    type Id = type_id!("Error");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::MethodMeta {
    type Id = type_id!("MethodMeta");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ArrayRefEValue {
    type Id = type_id!("ArrayRefEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::VecEValue {
    type Id = type_id!("VecEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::ProgramVerification {
    type Id = type_id!("ProgramVerification");
    type Kind = cxx::kind::Trivial;
}
