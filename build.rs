fn main() {
    let rustv = rustc_version();
    let check_cfg = rustv.map(|v| v >= 80).unwrap_or(false);

    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(error_in_core)");
    }
    if rustv.map(|v| v >= 81).unwrap_or(false) {
        // core::error::Error
        println!("cargo:rustc-cfg=error_in_core");
    }

    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_DENY_WARNINGS");
    let deny_warnings = std::env::var("EXECUTORCH_RS_DENY_WARNINGS").as_deref() == Ok("1");
    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(deny_warnings)");
    }
    if deny_warnings {
        println!("cargo:rustc-cfg=deny_warnings");
    }

    println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_LINK_TEST_KERNELS");
    if check_cfg {
        println!("cargo:rustc-check-cfg=cfg(tests_with_kernels)");
    }
    if std::env::var("EXECUTORCH_RS_LINK_TEST_KERNELS").as_deref() == Ok("1") {
        println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
        let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR").expect(
            "EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs",
        );

        println!("cargo::rustc-link-search=native={libs_dir}/kernels/portable/");
        println!("cargo::rustc-link-lib=static:+whole-archive=portable_kernels");
        println!("cargo::rustc-link-lib=static:+whole-archive=portable_ops_lib");

        println!("cargo:rustc-cfg=tests_with_kernels");
    }
}

fn rustc_version() -> Option<u32> {
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
    Some(minor)
}
