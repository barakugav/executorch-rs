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
