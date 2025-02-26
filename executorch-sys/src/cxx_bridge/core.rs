use cxx::{type_id, ExternType};

unsafe impl ExternType for crate::executorch::runtime::etensor::ScalarType {
    type Id = type_id!("executorch::aten::ScalarType");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch::runtime::TensorShapeDynamism {
    type Id = type_id!("executorch::aten::TensorShapeDynamism");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch_rs::Tensor {
    type Id = type_id!("executorch_rs::Tensor");
    type Kind = cxx::kind::Opaque;
}

unsafe impl ExternType for crate::executorch::runtime::Error {
    type Id = type_id!("executorch::runtime::Error");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch_rs::MethodMeta {
    type Id = type_id!("executorch_rs::MethodMeta");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch_rs::ArrayRefEValue {
    type Id = type_id!("executorch_rs::ArrayRefEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch_rs::VecEValue {
    type Id = type_id!("executorch_rs::VecEValue");
    type Kind = cxx::kind::Trivial;
}

unsafe impl ExternType for crate::executorch::runtime::Program_Verification {
    type Id = type_id!("executorch::runtime::Program::Verification");
    type Kind = cxx::kind::Trivial;
}
