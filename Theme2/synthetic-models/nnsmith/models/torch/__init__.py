import pickle
import os
import tempfile
import traceback
import time
import glob
import torch
from torch.onnx import verification
import onnxruntime
import numpy as np
import warnings
from onnx.tools import update_model_dims
import onnx

from io import BytesIO
from nnsmith.nnsmith.materialize.torch import TorchModelCPU
from nnsmith.nnsmith.logging import MGEN_LOG


class TorchModelExportable(TorchModelCPU):
    def export_onnx(
        self,
        result,
        model_path,
        opset=14,
    ):
        self.dummy_inputs = [
            torch.ones(size=svar.shape).uniform_(1, 2).to(dtype=svar.dtype.torch())
            for _, svar in self.torch_model.input_like.items()
        ]
        input_names = list(self.input_like.keys())
        path = "testout"
        if not os.path.exists(path):
            os.makedirs(path)

        # with tempfile.TemporaryDirectory(
        #     dir=os.getcwd(),
        # ) as tmpdir:
        try:
            MGEN_LOG.info(list(self.torch_model.output_like.keys()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
                warnings.simplefilter("ignore", category=UserWarning, append=True)
                start = time.time()
                torch.onnx.export(
                    self.torch_model,
                    tuple(self.dummy_inputs),
                    model_path,
                    input_names=list(self.torch_model.input_like.keys()),
                    output_names=list(self.torch_model.output_like.keys()),
                    opset_version=opset,
                )
                end = time.time()
                # result["files"] = glob.glob(f"{tmpdir}/*")
                result["time"] = end - start
                # time.sleep(5)
        except Exception as err:
            err_string = traceback.format_exc()
            result["error_des"] = err_string
            # result["files"] = glob.glob(f"{tmpdir}/*")
            # if glob.glob(f"{tmpdir}/*"):
            # return result
            result["error"] = 1

        return result

    def export_onnx_new(self, result, opset=14):
        dummy_inputs = [
            torch.ones(size=svar.shape).uniform_(1, 2).to(dtype=svar.dtype.torch())
            for _, svar in self.torch_model.input_like.items()
        ]
        input_names = list(self.input_like.keys())
        # path = "testout"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        f = BytesIO()
        # with tempfile.TemporaryDirectory(dir=os.getcwd(),) as tmpdir:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
                warnings.simplefilter("ignore", category=UserWarning, append=True)
                self.torch_model.eval()
                start = time.time()
                torch.onnx.export(
                    self.torch_model,
                    tuple(dummy_inputs),
                    f,
                    # input_names=list(self.torch_model.input_like.keys()),
                    # output_names=list(self.torch_model.output_like.keys()),
                    opset_version=opset,
                    # do_constant_folding=False
                )
                end = time.time()
            # MGEN_LOG.info(f.getvalue())
            # result['files'] = glob.glob(f"{tmpdir}/*")
            result["time"] = end - start
            # time.sleep(5)
        except Exception as err:
            err_string = traceback.format_exc()
            result["error_des"] = err_string
            result["error"] = 1
            return result
            # with open(path + outFile + ".txt", "w") as f:
            # f.write("output  " + str(counter) + " error message:\n" + str(e) + "\n" + traceback.format_exc())
            # self.
        # inputs = self.torch_model.get_random_inps()
        # outs = self.torch_model.forward(inputs)
        # try:
        #     onnx_model = self.get_onnx_model(f)
        #     MGEN_LOG.info(onnx_model)
        #     sess = onnxrt.create_session(onnx_model)
        #     # self.test_model(sess)
        # except:
        #     err_string = traceback.format_exc()
        #     result["error_des"] = err_string
        #     result["error"] = 1
        return result

    def get_onnx_model(self, buffer: BytesIO):
        onnx_model = onnx.load_model_from_string(buffer.getvalue())
        # onnx_model = update_model_dims.update_inputs_outputs_dims(
        #     onnx_model,
        #     {k: v.shape for k, v in self.torch_model.input_like.items()},
        #     {k: v.shape for k, v in self.torch_model.output_like.items()},
        # )
        # onnx.checker.check_model(onnx_model, full_check=True)
        return onnx_model

    def test_model(self, save_path, number_tests=100):

        for i in range(number_tests):
            inputs = self.torch_model.get_random_inps()
            outputs = self.torch_model.forward(**inputs)
            # MGEN_LOG.info(outputs)
            outputs = {key: outputs[key] for key in self.torch_model.output_like.keys()}
            paired_in_out = {
                "inputs": inputs,
                "outputs": outputs,
                "input_shapes": {
                    k: v.shape for k, v in self.torch_model.input_like.items()
                },
                "output_shapes": {
                    k: v.shape for k, v in self.torch_model.output_like.items()
                },
            }
            with open(os.path.join(save_path, f"in_outs_{i}.npy"), "wb") as f:
                np.save(f, paired_in_out)
            # with open(os.path.join(save_path, f"output_{i}.npy"), 'wb') as f:
            #     np.save(f, output)
        # for i, val in output.items():
        # MGEN_LOG.info(val)

    # def verify(self,):
    #     # try:
    #     ver = verification.find_mismatch(
    #         self.torch_model,
    #         tuple(self.dummy_inputs),
    #         # input_names=list(self.torch_model.input_like.keys()),
    #         # output_names=list(self.torch_model.output_like.keys()),
    #         opset_version=14,
    #     )
    #     ver.
    #     MGEN_LOG.info(ver)
        # except AssertionError as err:
        #     MGEN_LOG.error(err)
        # except Exception as err:
        #     pass
        # verification.check_export_model_diff