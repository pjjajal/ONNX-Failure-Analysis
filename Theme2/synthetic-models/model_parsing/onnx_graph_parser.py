import onnx
from pathlib import Path
import networkx as nx
from typing import Any, Tuple, List, Set
import itertools
from tqdm import tqdm


class OnnxGraphParser:
    def __init__(
        self,
        graph: onnx.GraphProto,
    ) -> None:
        self.onnx_graph = graph
        (
            self.ponnx_graph,
            self.input_node_names,
            self.output_node_names,
        ) = self._parse_graph(graph)

        self.all_nodes_types = self._all_node_types()

    @classmethod
    def load_from_path(cls, model_path: Path):
        model_onnx = onnx.load(model_path)
        return cls(model_onnx.graph)

    def _parse_graph(
        self,
        graph: onnx.GraphProto,
    ) -> Tuple[nx.DiGraph, nx.DiGraph, List[str], List[str]]:
        inputs = [(inp_node.name, {"node_type": "input"}) for inp_node in graph.input]
        outputs = [
            (out_node.name, {"node_type": "output"}) for out_node in graph.output
        ]
        nodes = [
            (
                node.name,
                {
                    "node_type": node.op_type,
                    "inputs": node.input,
                    "outputs": node.output,
                },
            )
            for node in graph.node
        ]

        input_node_names = list(map(lambda x: x[0], inputs))
        output_node_names = list(map(lambda x: x[0], outputs))

        parsed_graph = nx.DiGraph()
        parsed_graph.add_nodes_from(inputs)
        parsed_graph.add_nodes_from(outputs)
        parsed_graph.add_nodes_from(nodes)

        for node_name, node_data in parsed_graph.nodes(data=True):
            # Process non-input and output nodes
            if not node_data["node_type"] in ["input", "output"]:
                # Get the inputs of the current node
                node_inputs = node_data["inputs"]

                # Get the outputs of the current node
                node_outputs = node_data["outputs"]

                # If there are inputs to the node create edges to the current
                # node from the inputs
                if node_inputs:
                    for node_input in node_inputs:
                        # If the input is one of the input nodes
                        if node_input in input_node_names:
                            parsed_graph.add_edge(node_input, node_name)
                        else:
                            # Find which node this input comes from.
                            for con_node, con_node_data in parsed_graph.nodes(
                                data=True
                            ):
                                # Ignore input and output nodes.
                                if not con_node_data["node_type"] in [
                                    "input",
                                    "output",
                                ]:
                                    # Get list of outputs for the current node.
                                    conn_node_outputs = con_node_data["outputs"]
                                    # Iterate outputs and if the output matches the current input
                                    # create an edge.
                                    for conn_output in conn_node_outputs:
                                        if node_input == conn_output:
                                            parsed_graph.add_edge(con_node, node_name)

                # If there are outputs to the node create edges to the current
                # node from the ouptuts
                if node_outputs:
                    for node_output in node_outputs:
                        # If the output is one of the output nodes
                        if node_output in output_node_names:
                            parsed_graph.add_edge(node_name, node_output)
                        else:
                            # Find which node this output goes to.
                            for con_node, con_node_data in parsed_graph.nodes(
                                data=True
                            ):
                                # Ignore input and output nodes.
                                if not con_node_data["node_type"] in [
                                    "input",
                                    "output",
                                ]:
                                    # Get list of inputs for the current node.
                                    conn_node_inputs = con_node_data["inputs"]
                                    # Iterate inputs and if the input matches the current output
                                    # create an edge.
                                    for conn_input in conn_node_inputs:
                                        if node_output == conn_input:
                                            parsed_graph.add_edge(node_name, con_node)

        return (
            parsed_graph,
            input_node_names,
            output_node_names,
        )

    def _all_node_types(self) -> Set[str]:
        all_node_types = [
            node_type
            for node_name, node_type in self.ponnx_graph.nodes(data="node_type")
        ]
        return set(all_node_types)

    def get_all_simple_paths(self):
        simple_paths_dict = {}
        all_simple_paths = []
        for in_node, out_node in itertools.product(
            self.input_node_names, self.output_node_names
        ):
            if nx.has_path(self.ponnx_graph, in_node, out_node):
                simple_paths = list(
                    nx.all_simple_paths(self.ponnx_graph, in_node, out_node)
                )
                simple_paths_dict[(in_node, out_node)] = simple_paths
                all_simple_paths.extend(simple_paths)

        all_simple_paths = [
            list(map(lambda x: self.ponnx_graph.nodes[x]["node_type"], path))
            for path in all_simple_paths
        ]

        self.simple_paths_dict = simple_paths_dict
        self.all_simple_paths = all_simple_paths
        return all_simple_paths


if __name__ == "__main__":
    mismatched_onnx_path = "mismatch_test_new/torch/symbolic-cinit/16/model_15_2262524614_repro/onnx_debug/test_2023_09_02_15_43_33_039052/model.onnx"
    graph_parser = OnnxGraphParser.load_from_path(mismatched_onnx_path)
    all_simple_paths = graph_parser.get_all_simple_paths()
    graph_parser._all_node_types()
