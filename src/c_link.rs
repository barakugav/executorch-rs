pub mod executorch_c {
    #![allow(dead_code)]
    #![allow(unused_imports)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
