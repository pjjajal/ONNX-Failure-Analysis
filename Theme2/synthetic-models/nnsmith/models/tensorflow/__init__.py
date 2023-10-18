import pickle
import os
import tempfile
import traceback
import subprocess
import time
import glob
import torch
import tensorflow as tf
import tf2onnx

from typing import Callable, Dict, List, Type
from nnsmith.nnsmith.materialize.tensorflow import TFModelCPU, TFModel
from nnsmith.nnsmith.gir import GraphIR
from nnsmith.nnsmith.logging import MGEN_LOG


class TFModelExportable(TFModel):
    def export_onnx(self, result, opset=14):
        with tempfile.TemporaryDirectory(
            dir=os.getcwd(),
        ) as tmpdir:
            self.dump(tmpdir)
            time.sleep(10)
            try:
                start = time.time()
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "tf2onnx.convert",
                        "--saved-model",
                        f"{tmpdir}/tfnet",
                        "--output",
                        f"{tmpdir}/model.onnx",
                        "--opset",
                        f"{opset}"
                    ],
                    check=True,
                    capture_output=True,
                )
                # tf2onnx.convert.from_keras(
                #     self.net, opset=14, output_path=f"{tmpdir}/output.onnx"
                # )
                end = time.time()
                result["files"] = glob.glob(f"{tmpdir}/*.onnx")
                result["time"] = end - start
            except Exception as err:
                MGEN_LOG.error(err)
                if isinstance(err, subprocess.CalledProcessError):
                    result["error_des"] = err.stdout
                else:
                    result["error_des"] = err
                result["files"] = glob.glob(f"{tmpdir}/*.onnx")
                if glob.glob(f"{tmpdir}/*"):
                    return result
                result["error"] = 1
        return result


class TFModelCPUExportable(TFModelExportable):
    @property
    def device(self) -> tf.device:
        return tf.device(tf.config.list_logical_devices("CPU")[0].name)


class TFModelCUDAExportable(TFModelExportable):
    @property
    def device(self) -> tf.device:
        gpus = tf.config.list_logical_devices("GPU")
        assert gpus, "No GPU available"
        return tf.device(gpus[0].name)
