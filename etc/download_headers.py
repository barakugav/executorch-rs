import argparse
import glob
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT_DIR = Path(__file__).parent.parent.resolve()
HEADERS_DIR = ROOT_DIR / "executorch-sys" / "third-party" / "executorch"


def main():
    parser = argparse.ArgumentParser(description="Download executorch Cpp headers")
    parser.add_argument("version", type=str, help="executorch version")
    args = parser.parse_args()

    if HEADERS_DIR.exists():
        shutil.rmtree(HEADERS_DIR)
    HEADERS_DIR.mkdir(parents=True)

    with TemporaryDirectory() as tmpdir:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                f"v{args.version}",
                "https://github.com/pytorch/executorch.git",
            ],
            cwd=tmpdir,
        )
        tmp_headers_dir = Path(tmpdir) / "executorch"

        includes = [
            "runtime/core/**/*.h",
            "runtime/executor/**/*.h",
            "runtime/platform/**/*.h",
            "extension/data_loader/**/*.h",
            "extension/memory_allocator/**/*.h",
            "extension/module/**/*.h",
            "extension/tensor/tensor_ptr.h",
            "extension/flat_tensor/**/*.h",
            "devtools/etdump/etdump_flatcc.h",
            "devtools/etdump/data_sinks/buffer_data_sink.h",
            "devtools/etdump/data_sinks/data_sink_base.h",
            "LICENSE",
            "version.txt",
        ]
        excludes = [
            "runtime/executor/platform_memory_allocator.h",
            "runtime/platform/compat_unistd.h",
            "runtime/core/exec_aten/util/tensor_shape_to_c_string.h",
            "runtime/core/defines.h",
            "runtime/core/portable_type/c10/c10/util/overflows.h",
            "runtime/core/function_ref.h",
            "extension/data_loader/file_descriptor_data_loader.h",
            "extension/data_loader/mman.h",
            "extension/data_loader/mman_windows.h",
            "extension/module/bundled_module.h",  # TODO
            "extension/flat_tensor/serialize/serialize.h",
            "**/test/**",
            "**/testing_util/**",
        ]

        files = set()
        for include in includes:
            files.update(glob.glob(include, root_dir=tmp_headers_dir, recursive=True))
        for exclude in excludes:
            files.difference_update(
                glob.glob(exclude, root_dir=tmp_headers_dir, recursive=True)
            )
        for f in files:
            dst = HEADERS_DIR / f
            if not dst.parent.exists():
                dst.parent.mkdir(parents=True)
            shutil.copy(tmp_headers_dir / f, HEADERS_DIR / f)


if __name__ == "__main__":
    main()
