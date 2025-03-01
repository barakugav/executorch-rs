use std::path::{Path, PathBuf};

// const EXECUTORCH_VERSION: &str = "0.5.0";

fn main() {
    // TODO: verify on runtime we use the correct version of executorch
    // println!(
    //     "cargo:rustc-env=EXECUTORCH_RS_EXECUTORCH_VERSION={}",
    //     EXECUTORCH_VERSION
    // );

    build_c_bridge();
    #[cfg(any(feature = "tensor-ptr", feature = "module"))]
    build_cxx_bridge();
    generate_bindings();
    link_executorch();

    println!(
        "cargo::rerun-if-changed={}",
        include_dir().to_str().unwrap()
    );
    println!(
        "cargo::rerun-if-changed={}",
        third_party_include_dir().to_str().unwrap()
    );
}

fn build_c_bridge() {
    let c_bridge_dir = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("c_bridge");
    let mut builder = cc::Build::new();
    common_cc(&mut builder);
    builder
        .files([c_bridge_dir.join("c_bridge.cpp")])
        .include(include_dir())
        .include(third_party_include_dir());
    builder.compile(&format!(
        "executorch_rs_c_bridge_{}",
        env!("CARGO_PKG_VERSION")
    ));
}

#[cfg(any(feature = "tensor-ptr", feature = "module"))]
fn build_cxx_bridge() {
    let cxx_bridge_dir = Path::new(&env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("cxx_bridge");
    let mut bridges = Vec::new();
    if cfg!(feature = "module") {
        bridges.push("src/cxx_bridge/module.rs");
    }
    if cfg!(feature = "tensor-ptr") {
        bridges.push("src/cxx_bridge/tensor_ptr.rs");
    }
    let mut builder = cxx_build::bridges(bridges);
    common_cc(&mut builder);
    builder
        .files([cxx_bridge_dir.join("cxx_bridge.cpp")])
        .include(include_dir())
        .include(third_party_include_dir());
    builder.compile(&format!(
        "executorch_rs_cxx_bridge_{}",
        env!("CARGO_PKG_VERSION")
    ));
}

fn common_cc(builder: &mut cc::Build) {
    builder.cpp(true).std("c++17").cpp_link_stdlib(None); // linked via link-cplusplus crate
    if !cfg!(feature = "std") {
        // TODO: cpp executorch doesnt support nostd yet
        // builder.flag("-nostdlib");
    }
    for define in cpp_defines() {
        builder.define(define, None);
    }
}

fn generate_bindings() {
    let bindings_h = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("c_bindings.h");
    let mut bindings_h_content = String::from("#pragma once\n");
    for define in cpp_defines() {
        bindings_h_content.push_str(&format!("#define {}\n", define));
    }
    bindings_h_content.push_str("#include \"executorch_rs/c_bridge.h\"\n");
    std::fs::write(&bindings_h, bindings_h_content).expect("Unable to write bindings.h");

    let [_, minor, patch]: [u64; 3] = env!("CARGO_PKG_RUST_VERSION")
        .split('.')
        .map(|v| v.parse::<u64>().expect("Invalid rust version"))
        .collect::<Vec<_>>()
        .try_into()
        .expect("Rust version is not in the format MAJOR.MINOR.PATCH");
    let rust_version = bindgen::RustTarget::stable(minor, patch)
        .map_err(|e| format!("{}", e))
        .expect("Rust version not supported by bindgen");

    let builder = bindgen::Builder::default()
        .rust_target(rust_version)
        .clang_arg(format!("-I{}", include_dir().to_str().unwrap()))
        .use_core()
        .header(bindings_h.as_os_str().to_str().unwrap())
        .allowlist_file(format!(
            "{}/executorch_rs/c_bridge.h",
            include_dir().to_str().unwrap()
        ))
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .no_copy(".*") // TODO: specific some exact types, regex act weird
        .manually_drop_union(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("executorch_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn link_executorch() {
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_LINK");

    let link_enabled = std::env::var("EXECUTORCH_RS_LINK").as_deref() != Ok("0");

    if rustc_version().map(|v| v.minor >= 80).unwrap_or(false) {
        println!("cargo::rustc-check-cfg=cfg(link_cxx)");
    }
    if link_enabled {
        println!("cargo::rustc-cfg=link_cxx");
    }

    if std::env::var("DOCS_RS").is_ok() || !link_enabled {
        // Skip linking to the static library when building documentation
        return;
    }

    let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").ok();
    if libs_dir.is_none() {
        println!("cargo::warning=EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
    }

    if let Some(libs_dir) = &libs_dir {
        println!("cargo::rustc-link-search=native={libs_dir}");
    }
    println!("cargo::rustc-link-lib=static:+whole-archive=executorch");
    println!("cargo::rustc-link-lib=static:+whole-archive=executorch_core");

    if cfg!(feature = "data-loader") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/data_loader/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_data_loader");
    }

    if cfg!(feature = "module") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/module/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_module_static");
    }

    if cfg!(feature = "tensor-ptr") {
        if let Some(libs_dir) = &libs_dir {
            println!("cargo::rustc-link-search=native={libs_dir}/extension/tensor/");
        }
        println!("cargo::rustc-link-lib=static:+whole-archive=extension_tensor");
    }
}

fn include_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR")).join("include")
}

fn third_party_include_dir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR")).join("third-party")
}

fn cpp_defines() -> Vec<&'static str> {
    let mut defines = vec![];
    if cfg!(feature = "data-loader") {
        defines.push("EXECUTORCH_RS_DATA_LOADER");
    }
    if cfg!(feature = "module") {
        defines.push("EXECUTORCH_RS_MODULE");
    }
    if cfg!(feature = "tensor-ptr") {
        defines.push("EXECUTORCH_RS_TENSOR_PTR");
    }
    if cfg!(feature = "std") {
        defines.push("EXECUTORCH_RS_STD");
    }
    defines
}

struct RustVersion {
    #[allow(dead_code)]
    version: String,
    minor: u32,
}

fn rustc_version() -> Option<RustVersion> {
    // Code copied from cxx crate

    let rustc = std::env::var_os("RUSTC")?;
    let output = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .ok()?;
    let version = String::from_utf8(output.stdout).ok()?;
    let mut pieces = version.split('.');
    if pieces.next() != Some("rustc 1") {
        return None;
    }
    let minor = pieces.next()?.parse().ok()?;
    Some(RustVersion { version, minor })
}
