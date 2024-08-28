import argparse
import glob
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

HEADERS_DIR = Path(__file__).parent.parent / "executorch-sys" / "cpp" / "executorch"


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
            "extension/data_loader/**/*.h",
            "extension/memory_allocator/**/*.h",
            "extension/module/**/*.h",
            "runtime/core/**/*.h",
            "runtime/executor/**/*.h",
            "runtime/platform/**/*.h",
            "LICENSE",
            "version.txt",
        ]
        excludes = [
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
