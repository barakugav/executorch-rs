pub mod executorch_c {
    #![allow(dead_code)]
    #![allow(unused_imports)]
    #![allow(clippy::upper_case_acronyms)]

    include!(concat!(env!("OUT_DIR"), "/executorch_bindings.rs"));
}
