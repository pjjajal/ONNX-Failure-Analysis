import argparse
import pathlib
import subprocess
import json
import tempfile
import os
import glob
import logging
from typing import Any, Dict

def get_paths(data_path: pathlib.Path,):
    data_path = args.data_path
    with open(data_path, "r") as f:
        paths = f.readlines()

    return list(map(lambda x: pathlib.Path(x), paths))


def test_repo(
    model_path: pathlib.Path, result: Dict[str, Any], 
):
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        try:
            cmd = f"python -m transformers.onnx {tmpdir} --preprocessor=auto --framework=pt --model={model_path}"
            # cmd = f"python -m transformers.onnx {tmpdir} --preprocessor=auto --framework=tf --model={model_path}"
            proccess = subprocess.run(
                cmd, shell=True, check=True, text=True, capture_output=True,
            )
            logging.info(glob.glob(f"{tmpdir}/*"))
            result['files'] = glob.glob(f"{tmpdir}/*")
        except subprocess.CalledProcessError as exp:
            logging.error(f"(conversion failed for {model_path.name}")
            result["error_des"] = {
                "stdout": exp.stdout,
                "stderr": exp.stderr,
            }
            result['files'] = glob.glob(f"{tmpdir}/*")
            logging.info(glob.glob(f"{tmpdir}/*"))
            if glob.glob(f"{tmpdir}/*"):
                return result
            result["error"] = 1
    return result


def main(args):
    paths = get_paths(data_path=args.data_path,)
    results = []
    for i, path in enumerate(paths):
        try:
            logging.info(f"({i}/{len(paths)}): testing {path.parts[5]}")

            result = {"name": path.parts[5], "error": 0, "error_des": {}}
            result = test_repo(path, result,)
            results.append(result)

        except Exception as err:
            logging.error(err)
            result["error"] = 2
            results.append(result)
            continue
    with open(f"{args.output}", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=pathlib.Path)
    parser.add_argument("output", help="specifies the name of the output file")

    args = parser.parse_args()

    logging.basicConfig(
        filename="run.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s",
        datefmt="%I:%M:%S %p",
    )

    main(args)
