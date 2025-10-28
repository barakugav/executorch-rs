import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--profile",
    choices=["dev", "release"],
    default="dev",
)
args = parser.parse_args()

dir_excludes = ["models"]
examples_dir = Path(__file__).parent
examples = [d for d in examples_dir.iterdir() if d.is_dir()]
examples = [d for d in examples if d.name not in dir_excludes]
for example in examples:
    match example.name:
        case (
            "hello_world"
            | "no_ndarray"
            | "no_std"
            | "raw_tensor"
            | "data_map"
            | "etdump"
        ):
            extra_args = []
        case "nano-gpt":
            # TODO: https://github.com/pytorch/executorch/issues/15285
            # TODO: remove when we bump cpp lib to 1.0.1
            print("Skipping nanogpt example...")
            continue
            extra_args = [
                *["--model", "nanogpt.pte"],
                *["--tokenizer", "vocab.json"],
                *["--prompt", "hello world"],
                *["--length", "12"],
            ]
        case "llama3":
            extra_args = [
                *["--model", "llama3_2.pte"],
                *["--tokenizer", "vocab.json"],
                *["--prompt", "hello world"],
                *["--temperature", "0.2"],
                *["--length", "12"],
            ]
        case unknown:
            raise Exception(f"Unknown example directory: '{unknown}'")
    print(f"Running example '{example.name}'")
    subprocess.check_call(
        [
            *["cargo", "run"],
            *["--profile", args.profile],
            "-q",
            *["--", *extra_args],
        ],
        cwd=example,
    )
