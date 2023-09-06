# SPDX-License-Identifier: BSD-3-Clause
# Copyright(C) 2023 Marvell.

#!/usr/bin/env python3

try:
    from .analy_func.analy_func import Model
except:
    from analy_func.analy_func import Model

import argparse
import csv
import json

__version__ = '1.0.0'


def parse_cmdline_options():

    parser = argparse.ArgumentParser(description="onnxinsight " + __version__)
    parser.add_argument("onnx_model", help="ONNX model file")
    parser.add_argument("--version", action='version',
                        version=__version__, help="Show version and exit")
    parser.add_argument("--all", action='store_true',
                        help="Print all available tables")
    parser.add_argument("--info", action='store_true',
                        help="Print model general info table")
    parser.add_argument("--io", action='store_true',
                        help="Print model input/output info table")
    parser.add_argument("--op", action='store_true',
                        help="Print model operators summary table")
    parser.add_argument("--node_csv", action='store_true',
                        help="Save model node info to CSV file")
    parser.add_argument("--node_json", action='store_true',
                        help="Save model node info to json file")
    parser.add_argument("--shape", action='store_true',
                        help="Save shape of intermediate tensor into model")
    parser.add_argument("--simplify", action='store_true',
                        help="Simplify model and save the simplified model")
    parser.add_argument("--batch", type=int,
                        help="Set model batch size (experimental")

    args = parser.parse_args()
    return args


def main():
    args = parse_cmdline_options()
    if args.batch == None:
        batch = 0
    else:
        batch = args.batch

    # Load the model
    mod = Model(args.onnx_model)
    print("Analyzing {} ... " .format(mod.name.split("/")[-1]))

    # Check the model
    mod.check(batch)

    if args.all:
        mod.info()
        mod.io()
        mod.op()

    # Changing Batch Size
    if args.batch != None:
        print("Changing Batch Size (experimental)......")
        mod.changebatch(args.batch)

    # Add shape inference info to the model
    if args.shape:
        print("Saving shape info ......")
        mod.shapes()

    # Simplify the model
    if args.simplify:
        print("Simplifying......")
        mod.simplify()

    # Print model gerenal info
    if args.info:
        mod.info()
    # Print model inputs/outputs info
    if args.io:
        mod.io()
    # Print model operators info
    if args.op:
        mod.op()

    # Save model insights into JSON file
    if args.node_json:
        dic_model, _ = mod.infonode()
        print(dic_model)
        json_string = json.dumps(dic_model, indent=4)
        json_name = mod.name[:-5] + '.json'
        json_name = json_name.split("/")[-1]
        print("Saving model info to JSON in {}".format(json_name))
        with open(json_name, 'w') as outfile:
            outfile.write(json_string)

    # Save model insights into CSV file
    if args.node_csv:
        print("")
        print("Preparing CSV file")
        dic_model, _ = mod.infonode()
        csv_columns = ["name", "type", "input0", "in0_1", "in0_2", "in0_3", "in0_4", "in0_5", "in0_6", "in0_7", "in0_8",
                       "input1", "in1_1", "in1_2", "in1_3", "in1_4", "int1_5", "int1_6", "int1_7", "int1_8",
                       "output", "out_1", "out_2", "out_3", "out_4", "out_5", "out_6", "out_7", "out_8", "kernel_h", "kernel_w",
                       "stride_h", "stride_w", "group", "dilation_h", "dilation_w", "pad_0", "pad_1", "pad_2", "pad_3", "params", "macs"]

        csv_name = mod.name[:-5] + '.csv'
        csv_name = csv_name.split("/")[-1]

        for node in dic_model:
            if node["type"] != "Constant":
                node["input0"] = node["inputs"]["name0"]
                for i in range(len(node["inputs"]["shape0"])):
                    node[csv_columns[3+i]] = node["inputs"]["shape0"][i]
                if "name1" in node["inputs"]:
                    node["input1"] = node["inputs"]["name1"]
                    for i in range(len(node["inputs"]["shape1"])):
                        node[csv_columns[12+i]] = node["inputs"]["shape1"][i]
            for i in range(len(node["output"]["shape"])):
                node[csv_columns[21+i]] = node["output"]["shape"][i]
            node["output"] = node["output"]["name"]
            if node["type"] == "Conv" or node["type"] == "ConvTranspose" or node["type"] == "MaxPool" or node["type"] == "AveragePool":
                try:
                    node["kernel_h"] = node["attrs"]["kernel_shape"][0]
                    node["kernel_w"] = node["attrs"]["kernel_shape"][1]
                except:
                    print(
                        "warning: node does not have attributes information set, kernel information set to zero")
                    node["kernel_h"] = 0
                    node["kernel_w"] = 0
                try:
                    node["stride_h"] = node["attrs"]["strides"][0]
                    node["stride_w"] = node["attrs"]["strides"][1]
                except:
                    print(
                        "warning: node does not have attributes information set, stride information set to 1")
                    node["stride_h"] = 1
                    node["stride_w"] = 1
                try:
                    if "dilations" in node["attrs"]:
                        node["dilation_h"] = node["attrs"]["dilations"][0]
                        node["dilation_w"] = node["attrs"]["dilations"][1]
                except:
                    print(
                        "warning: node does not have attributes information set, dilation information set to 1")
                    node["dilation_h"] = 1
                    node["dilation_w"] = 1
                try:
                    if "pads" in node["attrs"]:
                        for i in range(4):
                            node["pad_" + str(i)] = node["attrs"]["pads"][i]
                except:
                    print(
                        "warning: node does not have attributes information set, pads information set to zero")
                    for i in range(4):
                        node["pad_" + str(i)] = 0
            if node["type"] == "Conv" or node["type"] == "ConvTranspose":
                try:
                    node["group"] = node["attrs"]["group"]
                except:
                    print(
                        "warning: node does not have attributes information set, group information set to 1")
                    node["group"] = 1
            node.pop("inputs")
            if "attrs" in node:
                node.pop("attrs")

        print("Saving model information to {}".format(csv_name))
        print("Disclaimer: Calculated macs are based on certain assumptions; see README")
        print("")

        with open(csv_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for node in dic_model:
                writer.writerow(node)


if __name__ == "__main__":
    main()
