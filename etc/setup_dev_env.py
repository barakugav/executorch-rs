import argparse
import multiprocessing
import platform
import shutil
import subprocess
import sys
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
        help="Remove the existing executorch directory before cloning",
    )
    args = parser.parse_args()

    if args.clean:
        if DEV_EXECUTORCH_DIR.exists():
            shutil.rmtree(DEV_EXECUTORCH_DIR)

    # TODO setup a venv here

    clone_executorch()

    subprocess.check_call([sys.executable, "-m", "ensurepip"])
    if not args.skip_executorch_python:
        subprocess.check_call(
            [sys.executable, "install_requirements.py"], cwd=DEV_EXECUTORCH_DIR
        )
    else:
        deps = [
            "cmake>=3,<4",
            "pyyaml",
            "setuptools>=63",
            "tomli",
            "wheel",
            "zstd",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
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
                "v0.7.0",
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
                    "8.3",
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
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
            "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
            "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
            "-DEXECUTORCH_ENABLE_LOGGING=ON",
            "-DBUILD_EXECUTORCH_PORTABLE_OPS=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON",
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


if __name__ == "__main__":
    main()
