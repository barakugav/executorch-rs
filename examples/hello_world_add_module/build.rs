use std::{collections::HashMap, path::PathBuf};

fn main() {
    println!("cargo::rustc-link-lib=static=portable_kernels");
    println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
        .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
    let libs_dir = envsubst::substitute(
        libs_dir,
        &HashMap::from([(
            String::from("EXECUTORCH_RS_TOP"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
        )]),
    )
    .unwrap();

    println!("cargo::rustc-link-search={}/kernels/portable/", libs_dir);
}
