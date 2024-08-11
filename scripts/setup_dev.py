import argparse
import multiprocessing
import shutil
import subprocess
from pathlib import Path

DEV_EXECUTORCH_DIR = (
    Path(__file__).parent.parent.resolve()
    / "executorch-sys"
    / "third-party"
    / "executorch"
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
    if not args.skip_executorch_python:
        subprocess.check_call(["./install_requirements.sh"], cwd=DEV_EXECUTORCH_DIR)
    else:
        deps = [
            "cmake",
            "pyyaml",
            "setuptools>=63",
            "tomli",
            "wheel",
            "zstd",
        ]
        subprocess.check_call([get_pip(), "install", *deps])
    build_executorch_with_dev_cfg()


def clone_executorch():
    if not DEV_EXECUTORCH_DIR.exists():
        DEV_EXECUTORCH_DIR.parent.mkdir(parents=True, exist_ok=True)
        # git clone --depth 1 --branch v0.3.0 https://github.com/pytorch/executorch.git
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.3.0",  # TODO: parse from somewhere
                "https://github.com/pytorch/executorch.git",
            ],
            cwd=DEV_EXECUTORCH_DIR.parent,
        )

    subprocess.check_call(
        ["git", "submodule", "sync", "--recursive"], cwd=DEV_EXECUTORCH_DIR
    )
    subprocess.check_call(
        ["git", "submodule", "update", "--init"], cwd=DEV_EXECUTORCH_DIR
    )


def build_executorch_with_dev_cfg():
    cmake_out_dir = DEV_EXECUTORCH_DIR / "cmake-out"
    if not cmake_out_dir.exists():
        cmake_out_dir.mkdir()
    subprocess.check_call(
        [
            "cmake",
            "-DDEXECUTORCH_SELECT_OPS_LIST=aten::add.out",
            "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
            "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
            "-DBUILD_EXECUTORCH_PORTABLE_OPS=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
            "-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON",
            "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
            "-DEXECUTORCH_ENABLE_LOGGING=ON",
            # TODO check USE_ATEN_LIB=true/false in CI
            "..",
        ],
        cwd=DEV_EXECUTORCH_DIR / "cmake-out",
    )

    subprocess.check_call(
        ["cmake", "--build", "cmake-out", "-j" + str(multiprocessing.cpu_count() + 1)],
        cwd=DEV_EXECUTORCH_DIR,
    )


def get_pip():
    try:
        subprocess.run(['pip3', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 'pip3'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        subprocess.run(['pip', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return 'pip'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError("Neither 'pip3' nor 'pip' is installed on this system.")

if __name__ == "__main__":
    main()
