import argparse
from pathlib import Path
from pprint import pprint
from typing import Any

import networkx as nx
import onnx
import pandas as pd

from model_parsing import OnnxAnalyzer, OnnxGraphParser


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
        "mismatch_con_dir", help="Specify the mismatch directory containing the data."
    )
    parser.add_argument(
        "--correct_model_dir",
        help="The directory containing the correctly converted model data",
    )
    parser.add_argument(
        "--test_model_dir",
        help="The directory containing the test models",
    )
    parser.add_argument("--n_jobs", default=3, type=int)
    parser.add_argument("--parallel", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    # Create get model paths
    model_paths = result_paths(
        args.base_directory, args.method, args.framework, args.opset
    )

    # Create dataframe and get models that are mismatched
    all_data = create_dataframe(model_paths)

    ################
    # Path where mismatched graphs are located
    mismatched_data_dir = (
        Path(args.mismatch_con_dir) / args.framework / args.method / str(args.opset)
    )

    # Mismatched paths
    if args.framework == "torch":
        mismatched_paths = [
            Path(i).name for i in all_data.loc[all_data["mismatch"] == 1]["path"]
        ]
    if args.framework == "tensorflow":
        mismatched_paths = [
            Path(i).parent.name
            for i in all_data.loc[(all_data["result"] == 0) & (all_data["mad"] > 1e-7)][
                "path"
            ]
        ]

    mismatched_paths = [
        mismatched_data_dir / str(mismatched_path + "_repro")
        for mismatched_path in mismatched_paths
    ]
    mismatched_onnx_paths = sorted(
        [
            list(mismatched_path.glob("**/*.onnx"))[0]
            for mismatched_path in mismatched_paths
        ]
    )

    print("Mismatched Paths:", len(mismatched_onnx_paths))
    ################

    ################
    if args.correct_model_dir:
        # Path where mismatched graphs are located
        correct_data_dir = (
            Path(args.correct_model_dir)
            / args.framework
            / args.method
            / str(args.opset)
        )

        # Correct paths
        if args.framework == "torch":
            correct_paths = [
                Path(i).name
                for i in all_data.loc[
                    (all_data["mismatch"] == 0) & (all_data["result"] == 0)
                ]["path"]
            ]
        if args.framework == "tensorflow":
            correct_paths = [
                Path(i).parent.name
                for i in all_data.loc[
                    (all_data["result"] == 0) & ~(all_data["mad"] > 1e-7)
                ]["path"]
            ]

        correct_paths = [
            correct_data_dir / str(correct_path + "_repro")
            for correct_path in correct_paths
        ]
        correct_onnx_paths = sorted(
            [list(correct_path.glob("**/*.onnx"))[0] for correct_path in correct_paths]
        )
        print("Correct Paths:", len(correct_onnx_paths))
    ################

    ################
    if args.test_model_dir:
        test_data_dir = (
            Path(args.test_model_dir)
        )
        test_onnx_paths = sorted(
                list(test_data_dir.glob("**/*.onnx"))
            )
        print("Test Model Paths:", len(test_onnx_paths))
    ################

    if args.correct_model_dir:
        analyzer = OnnxAnalyzer((mismatched_onnx_paths, correct_onnx_paths))
        # (
        #     total_mismatched_graph_paths,
        #     total_correct_graph_paths,
        # ) = analyzer.number_of_paths()
        # print(f"Total mismatched sequences: {total_mismatched_graph_paths}, Total correct sequences: {total_correct_graph_paths}")
        # if args.parallel:
        #     analyzer.analyze_parallel(args.n_jobs)
        # else:
        #     analyzer.analyze()
    elif args.test_model_dir:
        analyzer = OnnxAnalyzer((mismatched_onnx_paths, test_onnx_paths))
        # (
        #     total_mismatched_graph_paths,
        #     total_test_graph_paths,
        # ) = analyzer.number_of_paths()
        # print(f"Total mismatched sequences: {total_mismatched_graph_paths}, Total correct sequences: {total_test_graph_paths}")
        # if args.parallel:
        #     analyzer.analyze_parallel(args.n_jobs)
        # else:
        #     analyzer.analyze()
    else:
        analyzer = OnnxAnalyzer(mismatched_onnx_paths)
        # total_mismatched_graph_paths = analyzer.number_of_paths()
        # print(f"Total mismatched sequences: {total_mismatched_graph_paths}")
        # if args.parallel:
        #     analyzer.analyze_parallel(args.n_jobs)
        # else:
        #     analyzer.analyze()


if __name__ == "__main__":
    main()
