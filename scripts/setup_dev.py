import subprocess
from pathlib import Path

DEV_EXECUTORCH_DIR = Path(__file__).parent.parent.resolve() / "cpp" / "executorch"


def main():
    clone_executorch()
    subprocess.check_call(["./install_requirements.sh"], cwd=DEV_EXECUTORCH_DIR)
    build_executorch()


def clone_executorch():
    if not DEV_EXECUTORCH_DIR.exists():
        # git clone --depth 1 --branch v0.2.1 https://github.com/pytorch/executorch.git
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.2.1",  # TODO: parse from somewhere
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


def build_executorch():
    if not (DEV_EXECUTORCH_DIR / "cmake-out").exists():
        (DEV_EXECUTORCH_DIR / "cmake-out").mkdir()
        subprocess.check_call(
            [
                "cmake",
                "-DDEXECUTORCH_SELECT_OPS_LIST=aten::add.out",
                "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
                "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
                "-DBUILD_EXECUTORCH_PORTABLE_OPS=ON",
                "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON",
                "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
                "-DEXECUTORCH_ENABLE_LOGGING=ON",
                # TODO check USE_ATEN_LIB=true/false in CI
                "..",
            ],
            cwd=DEV_EXECUTORCH_DIR / "cmake-out",
        )

    subprocess.check_call(
        ["cmake", "--build", "cmake-out", "-j"],
        cwd=DEV_EXECUTORCH_DIR,
    )


if __name__ == "__main__":
    main()
