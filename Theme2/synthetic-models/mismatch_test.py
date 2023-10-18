import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


def setup_file_logger(log_file_path):
    # Create a logger object
    logger = logging.getLogger("mismatch_logger")

    # Check if any handlers are already attached to the logger
    if logger.hasHandlers():
        # If handlers exist, remove them to avoid duplicate logs
        logger.handlers.clear()

    # Set the logging level (options: DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)

    # Create a file handler to write log messages to the file
    file_handler = logging.FileHandler(log_file_path)

    # Set the logging level for the file handler
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter to format the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add the formatter to the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Create a custom stream class to redirect print statements to the logger
    class PrintToLogger:
        def write(self, message):
            if message.strip() != "":
                logger.info(message.strip())

        def flush(self):
            pass

    # Replace the standard output with our custom stream
    sys.stdout = PrintToLogger()

    return logger


def mismatch_test(path, repro_path, opset):
    import pickle
    import warnings

    import torch
    from nnsmith.materialize import Model, Oracle
    from torch.onnx.verification import find_mismatch

    warnings.simplefilter("ignore")

    # Get the paths for pickles and weights
    gir_path: Path = path / "gir.pkl"
    oracle_path: Path = path / "oracle.pkl"
    weights_path: Path = path / "model.pth"

    # Load the model from pickle
    with gir_path.open("rb") as f:
        gir = pickle.load(f)
    model_type = Model.init("torch", "cpu")
    model = model_type.from_gir(gir)

    # Load weights from weight path.
    model.torch_model.load_state_dict(torch.load(weights_path), strict=False)

    # Load oracle
    oracle = Oracle.load(oracle_path)

    model_args = tuple([torch.from_numpy(val) for key, val in oracle.input.items()])

    print(f"Testing: {str(path)}")
    graph_info = find_mismatch(
        model.torch_model,
        model_args,
        opset_version=opset,
        keep_initializers_as_inputs=False,
    )
    graph_info.export_repro(repro_path)
    print(repro_path)


def result_paths(base_directory, method, framework, opset):
    directory: Path = Path(base_directory) / framework
    paths = sorted(directory.glob(f"{framework}_{method}_opset_{opset}_*.json"))
    return paths


def create_dataframe(model_paths):
    dataframes = []
    for path in model_paths:
        dataframes.append(pd.read_json(path))
    all_data = pd.concat(dataframes).fillna(0)
    return all_data


def create_mismatch_test_dir(method, framework, opset, output_directory):
    mismismatch_test_path = Path(f"./{output_directory}/{framework}/{method}/{opset}")
    if not mismismatch_test_path.exists():
        mismismatch_test_path.mkdir(parents=True)
    return mismismatch_test_path


def main():
    # Set up argument parsing for the CLI
    parser = argparse.ArgumentParser(
        description="Process pickle files based on method, framework, and num_nodes."
    )
    parser.add_argument(
        "base_directory", help="Specify the base directory containing the data."
    )
    parser.add_argument("method", help="Specify the method name.")
    parser.add_argument("framework", help="Specify the framework name.")
    parser.add_argument("opset", help="opset", type=int)
    parser.add_argument(
        "output_directory", help="Specify the output directory containing the data."
    )
    parser.add_argument("--correct_graphs", help="Flag to specify if we want to export correct graphs.", action="store_true")

    args = parser.parse_args()

    # Create get model paths
    model_paths = result_paths(
        args.base_directory, args.method, args.framework, args.opset
    )

    # Create dataframe and get models that are mismatched
    all_data = create_dataframe(model_paths)
    if args.correct_graphs:
        print(len(all_data.loc[(all_data["mismatch"] == 0) & (all_data["result"] == 0)]['path']))
        mismatched_paths = [
            Path(i) for i in all_data.loc[(all_data["mismatch"] == 0) & (all_data["result"] == 0)]["path"]
        ]
    else:
        print(len(all_data.loc[all_data["mismatch"] == 1]["path"]))
        mismatched_paths = [
            Path(i) for i in all_data.loc[all_data["mismatch"] == 1]["path"]
        ]


    # Test mismatched models
    mismismatch_test_path = create_mismatch_test_dir(
        args.method, args.framework, args.opset, args.output_directory
    )
    for mismatched_path in mismatched_paths:
        if not args.correct_graphs:
            logger = setup_file_logger(
                mismismatch_test_path / (mismatched_path.name + ".log")
            )
        repro_path = mismismatch_test_path / (mismatched_path.name + "_repro")
        mismatch_test(mismatched_path, repro_path, args.opset)


if __name__ == "__main__":
    main()
