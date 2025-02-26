use cxx::{type_id, ExternType};

#[cxx::bridge]
pub mod ffi {

    unsafe extern "C++" {
        include!("executorch-sys/cpp/executorch_rs/cxx_bridge.hpp");

        #[namespace = "executorch::extension"]
        type Module;

        #[namespace = "executorch::extension::Module"]
        type LoadMode = crate::executorch::extension::Module_LoadMode;

        #[namespace = "executorch::runtime"]
        type Error = crate::executorch::runtime::Error;

        #[namespace = "executorch_rs"]
        type MethodMeta = crate::executorch_rs::MethodMeta;

        #[namespace = "executorch_rs"]
        type ArrayRefEValue = crate::executorch_rs::ArrayRefEValue;
        #[namespace = "executorch_rs"]
        type VecEValue = crate::executorch_rs::VecEValue;

        #[namespace = "executorch::runtime::Program"]
        type Verification = crate::executorch::runtime::Program_Verification;

        #[namespace = "executorch_rs"]
        fn Module_new(
            file_path: &str,
            load_mode: LoadMode,
            // event_tracer: *mut cxx::UniquePtr<crate::executorch::runtime::EventTracer>,
        ) -> UniquePtr<Module>;
        #[namespace = "executorch_rs"]
        fn Module_load(self_: Pin<&mut Module>, verification: Verification) -> Error;
        #[namespace = "executorch_rs"]
        fn Module_method_names(
            self_: Pin<&mut Module>,
            method_names_out: &mut Vec<String>,
        ) -> Error;
        #[namespace = "executorch_rs"]
        fn Module_load_method(self_: Pin<&mut Module>, method_name: &str) -> Error;
        #[namespace = "executorch_rs"]
        fn Module_is_method_loaded(self_: &Module, method_name: &str) -> bool;
        #[namespace = "executorch_rs"]
        unsafe fn Module_method_meta(
            self_: Pin<&mut Module>,
            method_name: &str,
            method_meta_out: *mut MethodMeta,
        ) -> Error;
        #[namespace = "executorch_rs"]
        unsafe fn Module_execute(
            self_: Pin<&mut Module>,
            method_name: &str,
            inputs: ArrayRefEValue,
            outputs: *mut VecEValue,
        ) -> Error;
    }
}

unsafe impl ExternType for crate::executorch::extension::Module_LoadMode {
    type Id = type_id!("executorch::extension::Module::LoadMode");
    type Kind = cxx::kind::Trivial;
}
