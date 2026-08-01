import argparse
import multiprocessing
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

DEV_EXECUTORCH_DIR = (
    Path(__file__).parent.parent.resolve() / "etc" / ".dev-env" / "executorch"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing executorch directory before cloning",
    )
    parser.add_argument(
        "--skip-executorch-python",
        action="store_true",
        help="Skip installing the executorch Python package",
    )
    args = parser.parse_args()

    if args.clean and DEV_EXECUTORCH_DIR.exists():
        shutil.rmtree(DEV_EXECUTORCH_DIR)

    # TODO setup a venv here

    clone_executorch()
    patch_flatcc_werror()

    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    if not args.skip_executorch_python:
        subprocess.check_call(
            [sys.executable, "install_executorch.py", "--use-pt-pinned-commit"],
            cwd=DEV_EXECUTORCH_DIR,
        )
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                DEV_EXECUTORCH_DIR / "requirements-dev.txt",
                "torch==2.12.0",
                "--extra-index-url",
                "https://download.pytorch.org/whl/test/cpu",
            ]
        )
    build_executorch_with_dev_cfg()

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"]
    )


def clone_executorch():
    if not DEV_EXECUTORCH_DIR.exists():
        DEV_EXECUTORCH_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v1.3.1",
                "https://github.com/pytorch/executorch.git",
                ".",
            ],
            cwd=DEV_EXECUTORCH_DIR,
        )

        if platform.system() == "Darwin":
            # Clone coremltools repo
            # Required on apple when EXECUTORCH_BUILD_DEVTOOLS=ON
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "9.0",
                    "https://github.com/apple/coremltools.git",
                ],
                cwd=DEV_EXECUTORCH_DIR / "backends" / "apple" / "coreml" / "scripts",
            )

    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=DEV_EXECUTORCH_DIR
    )
    subprocess.check_call(
        ["git", "submodule", "sync", "--recursive"], cwd=DEV_EXECUTORCH_DIR
    )


def build_executorch_with_dev_cfg():
    cmake_out_dir = DEV_EXECUTORCH_DIR / "cmake-out"
    if not cmake_out_dir.exists():
        cmake_out_dir.mkdir()
    subprocess.check_call(
        [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
            "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
            "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
            "-DEXECUTORCH_ENABLE_LOGGING=ON",
            "-DEXECUTORCH_BUILD_PORTABLE_OPS=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON",
            "-DEXECUTORCH_BUILD_XNNPACK=ON",
            "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON",
            "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON",
            "-DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON",
            "-DEXECUTORCH_BUILD_DEVTOOLS=ON",
            "-DEXECUTORCH_ENABLE_EVENT_TRACER=ON",
            "..",
        ],
        cwd=DEV_EXECUTORCH_DIR / "cmake-out",
    )

    subprocess.check_call(
        ["cmake", "--build", "cmake-out", "-j" + str(multiprocessing.cpu_count() + 1)],
        cwd=DEV_EXECUTORCH_DIR,
    )


def patch_flatcc_werror():
    # flatcc compiles with -Werror by default (its own FLATCC_ALLOW_WERROR option).
    # AppleClang 21+ (on the macos-latest runner) emits new warnings its old code
    # trips, which -Werror turns into hard build failures. Flip the option default
    # to OFF. A source patch is used rather than CFLAGS/-D flags because flatcc is
    # built as an ExternalProject (flatcc_ep) with a fixed CMAKE_ARGS list that the
    # outer cmake's flags/env don't reliably reach; patching its own CMakeLists
    # covers both the ExternalProject and the in-tree flatccrt build.
    flatcc_cmake = DEV_EXECUTORCH_DIR / "third-party" / "flatcc" / "CMakeLists.txt"
    text = flatcc_cmake.read_text()
    needle = 'option (FLATCC_ALLOW_WERROR "allow -Werror to be configured" ON)'
    patched = needle.replace(" ON)", " OFF)")
    if needle in text:
        flatcc_cmake.write_text(text.replace(needle, patched))
    elif patched not in text:
        warnings.warn(f"could not patch FLATCC_ALLOW_WERROR in {flatcc_cmake}")


if __name__ == "__main__":
    main()
