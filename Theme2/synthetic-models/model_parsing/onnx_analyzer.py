import itertools
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Set, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

from .onnx_graph_parser import OnnxGraphParser


@dataclass
class OnnxAnalyzer:
    # Mismatched paths
    mismatched_paths: List[Path]
    mismatched_path_hash_dict: Dict[str, str]
    hash_mismatched_path_dict: Dict[str, str]
    parsed_mismatched_graph_dict: Dict[str, OnnxGraphParser]
    # Correct paths
    correct_paths: List[Path]
    correct_path_hash_dict: Dict[str, str]
    hash_correct_path_dict: Dict[str, str]
    parsed_correct_graph_dict: Dict[str, OnnxGraphParser]
    # Token and node maps
    node_type_token_dict: Dict[str, str]
    token_node_type_dict: Dict[str, str]

    def __init__(self, paths: List[Path] | Tuple[List[Path], List[Path]]) -> None:
        self.compare = False
        self.mismatched_paths = paths
        if isinstance(paths, tuple):
            self.mismatched_paths, self.correct_paths = paths
            self.compare = True

        ######## Mismatched Paths ########
        # Create dict mapping path -> hash
        self.mismatched_path_hash_dict = {
            str(path): i for i, path in enumerate(self.mismatched_paths)
        }

        # Create dict mapping hash -> path
        self.hash_mismatched_path_dict = {
            path_hash: path
            for path, path_hash in self.mismatched_path_hash_dict.items()
        }

        # Parse graph, keys are path_hashes this is just nicer to read.
        self.parsed_mismatched_graph_dict = {
            path_hash: OnnxGraphParser.load_from_path(path)
            for path, path_hash in self.mismatched_path_hash_dict.items()
        }

        # Find all the node types.
        self.all_node_types = self._find_all_node_types(
            self.parsed_mismatched_graph_dict
        )
        print("Mismatched Model Nodes", len(self._find_all_node_types(
            self.parsed_mismatched_graph_dict
        )))
        ####################################

        ######## Correct Paths ########
        if self.compare:
            # Create dict mapping path -> hash
            self.correct_path_hash_dict = {
                str(path): i for i, path in enumerate(self.correct_paths)
            }

            # Create dict mapping hash -> path
            self.hash_correct_path_dict = {
                path_hash: path
                for path, path_hash in self.correct_path_hash_dict.items()
            }

            # Parse graph, keys are path_hashes this is just nicer to read.
            self.parsed_correct_graph_dict = {
                path_hash: OnnxGraphParser.load_from_path(path)
                for path, path_hash in self.correct_path_hash_dict.items()
            }

            # Find all the node types.
            self.all_node_types |= self._find_all_node_types(
                self.parsed_correct_graph_dict
            )

            print("Correct Model Nodes", len(self._find_all_node_types(
                self.parsed_correct_graph_dict
            )))
            x = self._find_all_node_types(self.parsed_correct_graph_dict) - self._find_all_node_types(self.parsed_mismatched_graph_dict)
            print("nodes not in mismatched", len(x), x)
            y =  self._find_all_node_types(self.parsed_mismatched_graph_dict) - self._find_all_node_types(self.parsed_correct_graph_dict)
            print("nodes not in test/correct", len(y), y)
            z = self._find_all_node_types(self.parsed_mismatched_graph_dict) & self._find_all_node_types(self.parsed_correct_graph_dict)
            print("intersect", len(z), z)

        ####################################

        # Create create node_type -> token dict
        self.node_type_token_dict = {
            node_type: chr(i) for i, node_type in enumerate(self.all_node_types)
        }

        # Create create token -> node_type dict
        self.token_node_type_dict = {
            token: node_type for node_type, token in self.node_type_token_dict.items()
        }

    def _find_all_node_types(self, graph_dict) -> Set[str]:
        all_node_types = set()
        for _, graph_parser in graph_dict.items():
            all_node_types |= graph_parser.all_nodes_types
        return all_node_types

    def _tokenize_paths(self, onnx_parser: OnnxGraphParser):
        if not hasattr(onnx_parser, "all_simple_paths"):
            onnx_parser.get_all_simple_paths()
        all_simple_paths = onnx_parser.all_simple_paths

        paths_to_strings = [self._tokenize(path) for path in all_simple_paths]
        return paths_to_strings

    def _tokenize(self, path: List) -> str:
        return "".join(list(map(lambda x: self.node_type_token_dict[x], path)))

    def _detokenize(self, tokenized_path: str) -> List[str]:
        return [self.token_node_type_dict[token] for token in tokenized_path]

    def _compare_all_paths(
        self, graph_1_tokenized_paths: List[str], graph_2_tokenized_paths: List[str]
    ) -> Set[str]:
        all_matched_sequences = []
        all_path_pairs = list(
            itertools.product(graph_1_tokenized_paths, graph_2_tokenized_paths)
        )
        for graph_1_path, graph_2_path in all_path_pairs:
            matched_sequences = self._compare_paths(graph_1_path, graph_2_path)
            all_matched_sequences.extend(matched_sequences)
        return set(all_matched_sequences), len(all_path_pairs)

    def _compare_paths(self, path_1: str, path_2: str) -> List[str]:
        matched_sequences = []
        seq_matcher = SequenceMatcher(None, path_1, path_2)
        matching_blocks = seq_matcher.get_matching_blocks()
        for matching_block in matching_blocks:
            a, b, size = matching_block.a, matching_block.b, matching_block.size
            match_1 = path_1[a : a + size]
            if len(match_1) > 1:
                matched_sequences.append(match_1)
        return matched_sequences

    def analyze(self):
        common_op_sequences = {}
        common_op_sequences_decoded = {}
        total_path_pairs = 0

        if self.compare:
            graph_pair_list = list(
                itertools.product(
                    self.parsed_mismatched_graph_dict.items(),
                    self.parsed_correct_graph_dict.items(),
                )
            )
        else:
            graph_pair_list = list(
                itertools.combinations(self.parsed_mismatched_graph_dict.items(), r=2)
            )

        for (graph_1_hash, parsed_graph_1), (
            graph_2_hash,
            parsed_graph_2,
        ) in tqdm(graph_pair_list):
            graph_1_tokenized_paths = self._tokenize_paths(parsed_graph_1)
            graph_2_tokenized_paths = self._tokenize_paths(parsed_graph_2)
            all_matched_sequences, num_path_pairs = self._compare_all_paths(
                graph_1_tokenized_paths, graph_2_tokenized_paths
            )
            all_matched_sequences_decoded = [
                self._detokenize(i) for i in all_matched_sequences
            ]
            common_op_sequences[f"{graph_1_hash},{graph_2_hash}"] = list(
                all_matched_sequences
            )
            common_op_sequences_decoded[
                f"{graph_1_hash},{graph_2_hash}"
            ] = all_matched_sequences_decoded
            total_path_pairs += num_path_pairs

        if not self.compare:
            json_output = {
                "mismatched_path_hash_dict": self.mismatched_path_hash_dict,
                "hash_mismatched_path_dict": self.hash_mismatched_path_dict,
                "node_type_token_dict": self.node_type_token_dict,
                "token_node_type_dict": self.token_node_type_dict,
                "total_path_pairs_analyzed": total_path_pairs,
                "results": common_op_sequences,
                "results_decoded": common_op_sequences_decoded,
            }
            with open("./mismatch_seq_match_results.json", "w") as f:
                json.dump(json_output, f)
        else:
            json_output = {
                "mismatched_path_hash_dict": self.mismatched_path_hash_dict,
                "hash_mismatched_path_dict": self.hash_mismatched_path_dict,
                "correct_path_hash_dict": self.correct_path_hash_dict,
                "hash_correct_path_dict": self.hash_correct_path_dict,
                "node_type_token_dict": self.node_type_token_dict,
                "token_node_type_dict": self.token_node_type_dict,
                "total_path_pairs_analyzed": total_path_pairs,
                "results": common_op_sequences,
                "results_decoded": common_op_sequences_decoded,
            }
            with open("./correct_mismatch_seq_match_results.json", "w") as f:
                json.dump(json_output, f)

    def analyze_parallel(self, n_jobs=3):
        common_op_sequences = {}
        common_op_sequences_decoded = {}
        total_path_pairs = 0

        if self.compare:
            graph_pair_list = list(
                itertools.product(
                    self.parsed_mismatched_graph_dict.items(),
                    self.parsed_correct_graph_dict.items(),
                )
            )
        else:
            graph_pair_list = list(
                itertools.combinations(self.parsed_mismatched_graph_dict.items(), r=2)
            )

        for (graph_1_hash, parsed_graph_1), (
            graph_2_hash,
            parsed_graph_2,
        ) in tqdm(graph_pair_list):
            graph_1_tokenized_paths = self._tokenize_paths(parsed_graph_1)
            graph_2_tokenized_paths = self._tokenize_paths(parsed_graph_2)
            all_matched_sequences, num_path_pairs = compare_all_paths_parallel(
                graph_1_tokenized_paths, graph_2_tokenized_paths, n_jobs=n_jobs
            )
            all_matched_sequences_decoded = [
                self._detokenize(i) for i in all_matched_sequences
            ]
            common_op_sequences[f"{graph_1_hash},{graph_2_hash}"] = list(
                all_matched_sequences
            )
            common_op_sequences_decoded[
                f"{graph_1_hash},{graph_2_hash}"
            ] = all_matched_sequences_decoded
            total_path_pairs += num_path_pairs

        if not self.compare:
            json_output = {
                "mismatched_path_hash_dict": self.mismatched_path_hash_dict,
                "hash_mismatched_path_dict": self.hash_mismatched_path_dict,
                "node_type_token_dict": self.node_type_token_dict,
                "token_node_type_dict": self.token_node_type_dict,
                "total_path_pairs_analyzed": total_path_pairs,
                "results": common_op_sequences,
                "results_decoded": common_op_sequences_decoded,
            }
            with open("./mismatch_seq_match_results.json", "w") as f:
                json.dump(json_output, f)
        else:
            json_output = {
                "mismatched_path_hash_dict": self.mismatched_path_hash_dict,
                "hash_mismatched_path_dict": self.hash_mismatched_path_dict,
                "correct_path_hash_dict": self.correct_path_hash_dict,
                "hash_correct_path_dict": self.hash_correct_path_dict,
                "node_type_token_dict": self.node_type_token_dict,
                "token_node_type_dict": self.token_node_type_dict,
                "total_path_pairs_analyzed": total_path_pairs,
                "results": common_op_sequences,
                "results_decoded": common_op_sequences_decoded,
            }
            with open("./correct_mismatch_seq_match_results.json", "w") as f:
                json.dump(json_output, f)

    def number_of_paths(self):
        total_mismatched_graph_paths = 0
        for graph_hash, parsed_graph in self.parsed_mismatched_graph_dict.items():
            graph_tokenized_paths = self._tokenize_paths(parsed_graph)
            total_mismatched_graph_paths += len(graph_tokenized_paths)
        
        if self.compare:
            total_correct_graph_paths = 0
            for graph_hash, parsed_graph in self.parsed_correct_graph_dict.items():
                graph_tokenized_paths = self._tokenize_paths(parsed_graph)
                total_correct_graph_paths += len(graph_tokenized_paths)
            return total_mismatched_graph_paths, total_correct_graph_paths
        return total_mismatched_graph_paths


# Parallel Code -- This needs to be at the top level so joblib isn't annoying.
# Parallelized version of self._compare_all_paths
def compare_all_paths_parallel(
    graph_1_tokenized_paths: List[str], graph_2_tokenized_paths: List[str], n_jobs=3
) -> Tuple[Set[str], int]:
    all_matched_sequences = []
    all_path_pairs = list(
        itertools.product(graph_1_tokenized_paths, graph_2_tokenized_paths)
    )
    results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
        delayed(compare_paths)(graph_1_path, graph_2_path)
        for graph_1_path, graph_2_path in all_path_pairs
    )
    for result in results:
        all_matched_sequences.extend(result)
    return set(all_matched_sequences), len(all_path_pairs)


# Parallelized version of self._compare_path
def compare_paths(path_1: str, path_2: str) -> List[str]:
    matched_sequences = []
    seq_matcher = SequenceMatcher(None, path_1, path_2)
    matching_blocks = seq_matcher.get_matching_blocks()
    for matching_block in matching_blocks:
        a, b, size = matching_block.a, matching_block.b, matching_block.size
        match_1 = path_1[a : a + size]
        if len(match_1) > 1:
            matched_sequences.append(match_1)
    return matched_sequences


if __name__ == "__main__":
    mismatched_onnx_path = [
        "mismatch_test_new/torch/symbolic-cinit/16/model_15_2262524614_repro/onnx_debug/test_2023_09_02_15_43_33_039052/model.onnx",
        "mismatch_test_new/torch/symbolic-cinit/16/model_1_2432546826_repro/onnx_debug/test_2023_09_02_15_43_45_446317/model.onnx",
        "mismatch_test_new/torch/symbolic-cinit/16/model_5_3093938566_repro/onnx_debug/test_2023_09_02_15_43_37_476944/model.onnx",
    ]

    analyzer = OnnxAnalyzer(mismatched_onnx_path)
    analyzer.analyze_parallel(3)

    # def square(x):
    #     return x**2

    # print(Parallel(5)(delayed(square)(i) for i in range(1000000)))
