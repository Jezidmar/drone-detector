import argparse

import onnx
import torch
from onnxruntime.quantization import (
    QuantType,
    matmul_4bits_quantizer,
    quant_utils,
    quantize_dynamic,
)

from model.load_model import load_EAT_model


def export_to_onnx(args):
    """Exports pytorch(.pt) model to .onnx format"""
    img_size = (100, 128)  # 1.02s
    batch_size = 1
    sample_input = torch.rand((batch_size, 1, *img_size))

    state_dict = torch.load(
        args.ckpt_path
    )  # this should be loaded to CUDA:0 by default and but it does not matter since we transfer it to 'rank' device later on.

    model = load_EAT_model(args.ckpt_path, args.config_path)

    model.load_state_dict(state_dict, strict=False)

    torch.onnx.export(
        model,
        sample_input,
        args.save_path,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=20,
    )

    # if args.fp16:
    #     from onnxconverter_common import float16
    #     model = onnx.load(args.save_path)
    #     model_fp16 = float16.convert_float_to_float16(model)
    #     onnx.save(model_fp16, args.save_path.replace('.onnx', '_fp16.onnx'))

    if args.quant_8:
        quantize_dynamic(
            args.save_path,
            args.save_path.replace(".onnx", "_int8.onnx"),
            weight_type=QuantType.QUInt8,
        )
    if args.quant_4:
        model = onnx.load(args.save_path)
        model21 = onnx.version_converter.convert_version(model, 21)
        quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=128,  # 2's exponential and >= 16
            is_symmetric=True,  # if true, quantize to Int4. otherwise, quantize to uint4.
            accuracy_level=4,  # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=(
                "MatMul",
                "Gather",
            ),  # specify which op types to quantize
            quant_axes=(
                ("MatMul", 0),
                ("Gather", 1),
            ),  # specify which axis to quantize for an op
        )
        quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
            model21,
            nodes_to_exclude=None,  # specify a list of nodes to exclude from quantization
            nodes_to_include=None,  # specify a list of nodes to force include from quantization
            algo_config=quant_config,
        )
        quant.process()
        quant.model.save_model_to_file(
            args.save_path.replace(".onnx", "_int4.onnx"), True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to trained model",
    )
    parser.add_argument(
        "--config_path",
        required=True,
        type=str,
        help="Path to model config",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Save path for trained model",
        default=None,
    )
    parser.add_argument("--quant_8", default=False)
    parser.add_argument("--quant_4", default=False)

    args = parser.parse_args()
    export_to_onnx(args)
