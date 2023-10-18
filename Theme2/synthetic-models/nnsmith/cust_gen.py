import logging
import os
import random
import time
import json
import pydot
import networkx as nx

import hydra
from omegaconf import DictConfig

from nnsmith.nnsmith.abstract.extension import activate_ext
from nnsmith.nnsmith.backends.factory import BackendFactory
from nnsmith.nnsmith.graph_gen import SymbolicGen, model_gen, viz
from nnsmith.nnsmith.logging import MGEN_LOG
from nnsmith.nnsmith.narrow_spec import auto_opset
from nnsmith.nnsmith.util import hijack_patch_requires, mkdir, op_filter
from nnsmith.nnsmith.materialize import Model, TestCase
from models import ModelCust
from models.torch import TorchModelExportable
from models.tensorflow import TFModelCUDAExportable, TFModelCPUExportable

@hydra.main(
    version_base=None, config_path="./nnsmith/nnsmith/config", config_name="main"
)
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]
    root_path = mgen_cfg["save"]
    results = []

    model_cfg = cfg["model"]
    ModelType = ModelCust.init(
        model_cfg["type"], backend_target=cfg["backend"]["target"]
    )
    ModelType.add_seed_setter()

    if cfg["backend"]["type"] is not None:
        factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
        )
    else:
        factory = None

    # Counts ops 
    opset = auto_opset(ModelType, factory, vulops=mgen_cfg["vulops"])
    MGEN_LOG.info(len(opset))
    # for op in opset:
    #     MGEN_LOG.info(op.name())
    opset = op_filter(opset, mgen_cfg["include"], mgen_cfg["exclude"])
    total_op = set(opset)
    
    # total_op = None
    i = 0
    MGEN_LOG.info(mgen_cfg['num_gen'])
    while (i < mgen_cfg['num_gen'] or total_op): 
    # for i in range(mgen_cfg["num_gen"]):
        i += 1
        MGEN_LOG.info(f"({i}/{mgen_cfg['num_gen']}) OPs left: {len(total_op)}")
        seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]
        MGEN_LOG.info(f"Using seed {seed}")
        seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]
        n_nodes = mgen_cfg["max_nodes"]
        mgen_cfg["save"] = root_path + f"/{n_nodes}_{seed}_pt"
        result = {
            "name": mgen_cfg["save"],
            "error": 0,
            "error_des": {},
            "mad": 0,
            "ml1": 0,
            "ml2": 0,
        }

        model_cfg = cfg["model"]
        ModelType = ModelCust.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        ModelType.add_seed_setter()

        if cfg["backend"]["type"] is not None:
            factory = BackendFactory.init(
                cfg["backend"]["type"],
                target=cfg["backend"]["target"],
                optmax=cfg["backend"]["optmax"],
            )
        else:
            factory = None

        # GENERATION
        opset = auto_opset(ModelType, factory, vulops=mgen_cfg["vulops"])
        opset = op_filter(opset, mgen_cfg["include"], mgen_cfg["exclude"])
        # if total_op == None:
        #     total_op = set(opset)
        hijack_patch_requires(mgen_cfg["patch_requires"])
        activate_ext(opset=opset, factory=factory)

        MGEN_LOG.info(len(opset))

        tgen_begin = time.time()
        gen = model_gen(
            opset=opset,
            method=mgen_cfg["method"],
            seed=seed,
            max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
            rank_choices=mgen_cfg["rank_choices"],
            dtype_choices=mgen_cfg["dtype_choices"],
        )
        tgen = time.time() - tgen_begin

        if isinstance(gen, SymbolicGen):
            MGEN_LOG.info(
                f"{len(gen.last_solution)} symbols and {len(gen.solver.assertions())} constraints."
            )

            if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
                MGEN_LOG.debug("solution:" + ", ".join(map(str, gen.last_solution)))

        # MATERIALIZATION
        tmat_begin = time.time()
        ir = gen.make_concrete()
        ir_dot = ir.to_dot()
        dot_graph = pydot.graph_from_dot_data(ir_dot)
        nx_graph = nx.nx_pydot.from_pydot(*dot_graph)
        ops = set(list(map(lambda x: type(x.iexpr.op), ir.insts)))
        # MGEN_LOG.info(len(total_op))
        total_op = total_op - ops

        # MGEN_LOG.info(len(nx_graph.nodes()))
        # MGEN_LOG.info(nx.average_node_connectivity(nx_graph))

        MGEN_LOG.info(
            f"Generated DNN has {ir.n_var()} variables and {ir.n_compute_inst()} operators."
        )

        mkdir(mgen_cfg["save"])
        if cfg["debug"]["viz"]:
            fmt = cfg["debug"]["viz_fmt"].replace(".", "")
            viz(ir, os.path.join(mgen_cfg["save"], f"graph.{fmt}"))
        model = ModelType.from_gir(ir)

        model.refine_weights()  # either random generated or gradient-based.
        # oracle = model.make_oracle()
        tmat = time.time() - tmat_begin

        tsave_begin = time.time()
        # testcase = TestCase(model, oracle)
        # testcase.dump(root_folder=mgen_cfg["save"])
        if isinstance(
            model, (TorchModelExportable, TFModelCPUExportable, TFModelCUDAExportable)
            # model, (TorchModelExportable)
        ): 
            try:
                model_path = os.path.join(mgen_cfg["save"], f"{i}_{mgen_cfg['max_nodes']}_{model_cfg['type']}.onnx") 
                result = model.export_onnx(result, model_path)
            except:
                i -= 1
                continue
            model.test_model(mgen_cfg["save"])
        tsave = time.time() - tsave_begin
        MGEN_LOG.info(
            f"Time:  @Generation: {tgen:.2f}s  @Materialization: {tmat:.2f}s  @Save: {tsave:.2f}s"
        )
        results.append(result)
        with open(
            f"./result_{mgen_cfg['max_nodes']}_{model_cfg['type']}_{mgen_cfg['num_gen']}_.json",
            "a",
        ) as f:
            MGEN_LOG.info(result)
            f.write(json.dumps(result))

    with open(
        f"./result_{mgen_cfg['max_nodes']}_{model_cfg['type']}_{mgen_cfg['num_gen']}_.json",
        "w",
    ) as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    main()
