# onnxinsight Tool
## This tool is designed for ONNX model analysis.

SPDX-License-Identifier: BSD-3-Clause
Copyright(C) 2023 Marvell.

Required: Python3 (3.7.x & 3.8.x), requirements.txt

Use 'python onnxinsight.py -h' to explore available features
===========================================================

Use 'python onnxinsight.py [--help] [--version] [--all] [--info] [--io] [--op] [--node_csv] [--shape] [--batch] [--simplify] onnx_model'

Given a valid ONNX model as input, the tool performs per-node analysis of the model.

To get more insight, try different arguments.

Required arguments:

    onnx_model  ONNX model file

Optional arguments:

    -h, --help   show this help message and exit

    --version    Show version and exit

    --all        Print all available tables
                 Print out model, inputs/outputs, operators tables

    --info       Print model general info table
                 Model Overview 
                 -----------------------------------
                 | Model Components| Value |
                 -----------------------------------
                 | ir_version      | ****  |
                 -----------------------------------
                 | ****            | ****  |
                 -----------------------------------
                 | model_MACs      | ****  |
                 -----------------------------------
                 | model_params    | ****  |
                 -----------------------------------
                 Disclaimer: MACs (number of multiply and accumulate operations) account for Convolution, ConvTranspose
                     and MatMul layers assuming 4dim input tensors for Convolutions and 2dim for Gemm and 
                     2, 3 and 4 dim input tensors for MatMul. This covers many networks, but may not cover all
                     A detailed layer by layer information is provied with --node_csv
                     Layer specific operator information can be used to verify the MACs and/or to calculate MACs
                     for other operators that perform MAC operations.
           
    --io         Print model inputs/outputs info table
                 Model Inputs/outputs Overview
                ------------------------------------------
                 | ID | Name | Shape | Type | ByteSize|
                 ------------------------------------------
                 | ** | **** | ****  | ***  | *****   |
                 ------------------------------------------

    --op         Print model operators summary table
                 Model Operators Overview
                 --------------------------
                 | Op Name  |   #   |
                 --------------------------
                 | ****     |  ***  |
                 --------------------------
                 | Total    |  **** |
                 --------------------------

    --node_csv   Save model node info to CSV file
                Save each node "name", "type", "inputs", "output", "params", "macs", "attrs" information to CSV file

    --shape      Save shape of intermediate tensor into model
                 Run shape inference and save the model with shape information

    --simplify   Simplify model and save the simplified model

    --batch      Change the batch size (experimental) changes first dimension of inputs and outputs
                 Does not support Reshape operator and other explicitly tensor shape changing operators
                 Feature relies on onnx shape propagation
                 For dynamic batch size models, this feature can be used to set the batch size to a fixed dimension,
                 and it is recomended to run the --simplify step afterwards.
                 An alternative tool to try may be the onnxruntime.tools.make_dynamic_shape_fixed
               
# Examples:
 1. python onnxinsight.py --all model.onnx
 2. python onnxinsight.py --all --node_csv model.onnx
 3. python onnxinsight.py --shape model.onnx

