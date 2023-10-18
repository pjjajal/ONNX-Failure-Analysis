import argparse
import json
import pickle
import warnings
from pathlib import Path
from pprint import pprint

from nnsmith.materialize import Model, Oracle

# Ignore all warnings
warnings.simplefilter("ignore")


def process_pickle_files(base_directory, method, framework, num_nodes):
    # Construct the full path to the target directory
    directory_path = Path(base_directory) / method / framework / num_nodes

    # Check if the directory exists
    if not directory_path.exists():
        print(
            f"Directory for method '{method}', framework '{framework}', and num_nodes '{num_nodes}' does not exist."
        )
        return

    model_paths = {}
    # Loop through each <model_name> directory inside the target directory
    for model_name_dir in directory_path.iterdir():
        if not model_name_dir.is_dir():
            continue

        # Get a list of all pickle files inside the current <model_name> directory
        if framework == "torch":
            gir_pkl = list(model_name_dir.glob("gir.pkl"))
            oracle_pkl = list(model_name_dir.glob("oracle.pkl"))
            # Check if pickle files exist.
            if not (len(gir_pkl) == 1 and len(oracle_pkl) == 1):
                print(
                    f"Missing pickle files for method '{method}', framework '{framework}', num_nodes '{num_nodes}', and model_name '{model_name_dir.name}'."
                )
                continue
            model_paths[model_name_dir.name] = {
                "gir": gir_pkl[0],
                "oracle": oracle_pkl[0],
            }
        elif framework == "tensorflow":
            tfnet_path = list(model_name_dir.glob("model/tfnet"))
            gir_pkl = list(model_name_dir.glob("model/gir.pkl"))
            oracle_pkl = list(model_name_dir.glob("oracle.pkl"))
            # Check if files exist
            if not (
                len(tfnet_path) == 1 and len(oracle_pkl) == 1 and len(gir_pkl) == 1
            ):
                print(
                    f"Missing files for method '{method}', framework '{framework}', num_nodes '{num_nodes}', and model_name '{model_name_dir.name}'."
                )
                continue
            model_paths[model_name_dir.name] = {
                "gir": gir_pkl[0],
                "tfnet": tfnet_path[0],
                "oracle": oracle_pkl[0],
            }
    # Sort dictionary by key
    model_paths = {key: model_paths[key] for key in sorted(model_paths.keys())}
    return model_paths


def exporter_test(
    framework: str, model_paths: dict, output_file_path: Path, opset: int
):
    if framework == "torch":
        export_torch(model_paths, output_file_path, opset)
    elif framework == "tensorflow":
        export_tf(model_paths, output_file_path, opset)


def export_torch(model_paths: dict, output_file_path: Path, opset: int):
    import torch
    from torch.onnx.verification import find_mismatch

    torch.onnx.disable_log()

    results_data = []
    for model_name, pickle_dict in model_paths.items():
        print(f"Attempting to convert: {model_name}.")

        # Get the paths for pickles and weights
        gir_path: Path = pickle_dict["gir"]
        oracle_path: Path = pickle_dict["oracle"]
        parent_path: Path = gir_path.parent
        weights_path: Path = parent_path / "model.pth"

        # Load the model from pickle
        with gir_path.open("rb") as f:
            gir = pickle.load(f)
        model_type = Model.init("torch", "cpu")
        model = model_type.from_gir(gir)

        # Load weights from weight path.
        if not weights_path.exists():
            print("No weights found.")
        else:
            model.torch_model.load_state_dict(
                torch.load(weights_path),
                strict=False,
            )

        # Load oracle
        oracle = Oracle.load(oracle_path)

        try:
            model_args = tuple(
                [torch.from_numpy(val) for key, val in oracle.input.items()]
            )

            graph_info = find_mismatch(
                model.torch_model,
                model_args,
                opset_version=opset,
                keep_initializers_as_inputs=False,
            )
            print(f"Success: {model_name}")
            results_data.append(
                {
                    "model": model_name,
                    "path": str(parent_path),
                    "result": 0,
                    "error": None,
                    "mismatch": graph_info.has_mismatch(),
                }
            )
        except Exception as e:
            print(f"Failure: {model_name}")
            results_data.append(
                {
                    "model": model_name,
                    "path": str(parent_path),
                    "result": 1,
                    "error": str(e),
                }
            )

    with open(output_file_path, "w") as f:
        json.dump(results_data, f, indent=2)


def export_tf(model_paths: dict, output_file_path: Path, opset: int):
    import tf2onnx
    import tensorflow as tf
    import subprocess
    import onnxruntime as ort
    import numpy as np

    results_data = []
    for model_name, pickle_dict in list(model_paths.items()):
        print(f"Attempting to convert: {model_name}.")
        # Get the paths for pickles and weights
        gir_path: Path = pickle_dict["gir"]
        tfnet_path: Path = pickle_dict["tfnet"]
        oracle_path: Path = pickle_dict["oracle"]
        parent_path: Path = gir_path.parent
        weights_path: Path = parent_path / "model.pth"

        output_model_path = (
            output_file_path.parent / Path(*parent_path.parts[-3:-1]) / "model.onnx"
        )

        # Load the model from pickle
        with gir_path.open("rb") as f:
            gir = pickle.load(f)
        model_type = Model.init("tensorflow", "cpu")
        model = model_type.from_gir(gir)
        model.refine_weights()

        # Load oracle
        oracle = Oracle.load(oracle_path)

        # Convert Model
        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "tf2onnx.convert",
                    "--saved-model",
                    str(tfnet_path),
                    "--output",
                    str(output_model_path),
                    "--opset",
                    f"{opset}",
                ],
                check=True,
                capture_output=True,
            )

            output_names = list(model.output_like.keys())
            sess = ort.InferenceSession(output_model_path)
            mads = []
            for i in range(10):
                inputs = model.random_inputs()
                inputs = {key: val.numpy() for key, val in inputs.items()}
                tf_outputs = model.net.call_by_dict(inputs)
                ort_inputs = {key: np.atleast_1d(val) for key, val in inputs.items()}
                ort_outputs = sess.run(output_names=output_names, input_feed=ort_inputs)
                for out_name, ort_output in zip(output_names, ort_outputs):
                    tf_output = tf_outputs[out_name]
                    mad = np.max(np.absolute(ort_output - tf_output))
                    mads.append(mad)
                    # print(tf_output.numpy(), ort_output)
            max_mad = np.array(mads).max()
            print(f"Success: {model_name}")
            results_data.append(
                {
                    "model": model_name,
                    "path": str(parent_path),
                    "result": 0,
                    "error": None,
                    # "mismatch": graph_info.has_mismatch(),
                    "mad": float(max_mad),
                    "onnx_model_path": str(output_model_path),
                }
            )
        except Exception as err:
            print(f"Failure: {model_name}")
            # print(err)
            if isinstance(err, subprocess.CalledProcessError):
                results_data.append(
                    {
                        "model": model_name,
                        "path": str(parent_path),
                        "result": 1,
                        "error": str(err.stdout),
                    }
                )
            else:
                results_data.append(
                    {
                        "model": model_name,
                        "path": str(parent_path),
                        "result": 1,
                        "error": str(err),
                    }
                )
        except KeyboardInterrupt:
            print(f"Did not complete {model_name}")
            results_data.append(
                {
                    "model": model_name,
                    "path": str(parent_path),
                    "result": 1,
                    "error": "Did not complete.",
                    "hang": 1,
                }
            )
        # result["error_des"] = str(err)
    # pprint(results_data)
    with open(output_file_path, "w") as f:
        json.dump(results_data, f, indent=2)


def main():
    # Set up argument parsing for the CLI
    parser = argparse.ArgumentParser(
        description="Process pickle files based on method, framework, and num_nodes."
    )
    parser.add_argument(
        "--base_directory",
        help="Specify the base directory containing the data.",
        required=True,
    )
    parser.add_argument("--method", help="Specify the method name.", required=True)
    parser.add_argument(
        "--framework", help="Specify the framework name.", required=True
    )
    parser.add_argument(
        "--num_nodes", help="Specify the num_nodes value.", required=True
    )
    parser.add_argument("--opset", help="opset", type=int, required=True)
    parser.add_argument(
        "--output_directory",
        help="Specify the output directory containing the data.",
        required=True,
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to process the pickle files with the specified parameters
    model_paths = process_pickle_files(
        args.base_directory, args.method, args.framework, args.num_nodes
    )

    # base_output_folder = Path("./exporter_test_results") / f"{args.framework}"
    # base_output_folder = Path(f"./exporter_test_results/{args.framework}")
    base_output_folder = Path(f"./{args.output_directory}/{args.framework}")
    if not base_output_folder.exists():
        base_output_folder.mkdir(parents=True)
    output_file_name = (
        f"{args.framework}_{args.method}_opset_{args.opset}_{args.num_nodes}.json"
    )
    output_file_path = base_output_folder / output_file_name

    exporter_test(
        framework=args.framework,
        model_paths=model_paths,
        output_file_path=output_file_path,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
