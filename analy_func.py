# SPDX-License-Identifier: BSD-3-Clause
# Copyright(C) 2023 Marvell.

import onnx
import numpy as np
import onnxruntime as ort
import argparse
from onnx import numpy_helper
from onnxsim import simplify
from onnxsim import model_info
from rich.table import Table
from rich.text import Text
from rich import print

global_warning_flag_macs  = False

def human_readable_size(num, suffix=""):
    """
    Returns the human readable size

        Parameters:
            num (int): A integer
            suffix (str, optional): A suffix

        Returns:
            human readable number
            e.g. 1000000 -> 1M
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Y{suffix}"

class Model:
    """
    A class to analyze an ONNX model
    ...
    Attributes
    ----------
    name: str
        model name
    model:
        load ONNX model
    shape:
        model tensor shape

    Methods
    -------
    shapes():
        Save shape of intermediate tensors into the model.
    changebatch(batch):
        Changes the batch size (experimental)
    check():
        Check the model.
    simplify():
        simplify the model.
    info():
        print ONNX model metadata table.
    io():
        print model input and output information table.
    op():
        print model operators summary table.
    infoshape():
        extract input and output shape of each node.
    infoparams():
        calculate the params of each node and total params of model.
    infonode():
        construct insight dictionary for each node.
    """

    def __init__(self, model_name):
        """
        Load and construct shape attributes for the Model object.

        Parameters
        ----------
            model_name: str
                ONNX model file path

        Returns
        -------
        """

        self.name = model_name
        self.model = onnx.load(model_name)
        self.shape = onnx.shape_inference.infer_shapes(self.model)

    def shapes(self):
        """
        Save shape of tensors into xxx_shape.onnx file.

        Parameters
        ----------

        Returns
        -------
        """

        name_shape = self.name[:-5]+ '_shape.onnx'
        onnx.save(self.shape, name_shape)
        print("The model with shape info is saved at {}".format(name_shape))

    def changebatch(self, batch):
        """
        Experimental
        Set the model batch if the argument "batch" is passed.
        set the model batch to 1 if the original model batch is unknown
        Save the new batch info into xxx_bx.onnx

        Parameters
        ----------
        batch: int
            batch size

        Returns
        -------
        Model with new batch
        """

        orig_batch = self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        if orig_batch != 0:
            print("The original model batch is {}".format(orig_batch))
            if batch == orig_batch:
                print("The original model batch is {} and equal to the desired batch size".format(orig_batch))
                return self
            else:
                print("Set the new model batch to {}".format(batch))
                for inp in self.model.graph.input:
                    inp.type.tensor_type.shape.dim[0].dim_value = batch
                for output in self.model.graph.output:
                    output.type.tensor_type.shape.dim[0].dim_value = batch
        elif orig_batch == 0:
            print("The original model batch is unknown")
            if batch == None: batch = 1
            print("Set the new model batch to {}".format(batch))
            for inp in self.model.graph.input:
                inp.type.tensor_type.shape.dim[0].dim_value = batch
            for output in self.model.graph.output:
                output.type.tensor_type.shape.dim[0].dim_value = batch

        name_batch = self.name[:-5]+"_b" + str(batch) + ".onnx"
        print("Batch {} ONNX model is saved in {}".format(batch, name_batch))
        onnx.save(self.model, name_batch)

        self.name = name_batch
        self.model = self.model
        try:
            self.shape = onnx.shape_inference.infer_shapes(self.model)
        except:
            print("WARNING: changing of the batch resulted in an invalid model")
            print("Some shape information is inconsistent")
            exit()

        try:
            sess = ort.InferenceSession(self.name)
        except:
            print("WARNING: changing of the batch resulted in an invalid model")
            print("try running --simplify prior to changing of the batch")
            exit()
        return self

    def check(self,batch):
        """
        Check the model if it is valid.

        Parameters
        ----------
        batch: int, optional
            batch size (default is None)

        Returns
        -------
        """
        if batch == 0 and self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 0:
            print("The model has a dynamic batch size")
            print("Try setting the batch size with --batch argument, then use the newly generated model")
            print("After setting batch size, you may also want to try run --simplified, then use the simplified model")
            exit()
        try:
            onnx.checker.check_model(self.model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: {e}")
        else:
            print("The model is valid, based on static ONNX standards validation check")

    def simplify(self):
        """
        Simplify the model and save the simplified model into xxx_simplified.onnx

        Parameters
        ----------
        None

        Returns
        -------
        Simplified model
        """

        model_sim, check  = simplify(self.model)
        model_info.print_simplifying_info(self.model, model_sim)
        assert check, "Simplified ONNX model could not be validated"

        name_sim = self.name[:-5]+ '_simplified.onnx'
        onnx.save(model_sim, name_sim)
        print("Simplified ONNX model is saved in {}".format(name_sim))
        return model_sim

    def info(self):
        """
        Print model metadata table including "ir_version", "producer_name", "producer_version", "domain", "model_version", "doc_string", "Size", "Macs", "Params".

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        #onnx model components
        components = ["ir_version", "producer_name", "producer_version", "domain", "model_version", "doc_string"]
        table = Table(title="Model Overview", show_lines=True)
        table.add_column('Model Components')
        table.add_column('Value')
        for comp in components:
            table.add_row(comp, str(getattr(self.model, comp)))

        op_v = []
        for opset in self.model.opset_import:
            op_v.append(opset.version)
        table.add_row("opset_version", str(op_v))
        table.add_row("model_ByteSize", human_readable_size(self.model.ByteSize(), suffix="B"))
        _, total_macs = self.infonode()
        table.add_row("model_MACs", human_readable_size(total_macs))
        _, total_params = self.infoparam()
        table.add_row("model_params", human_readable_size(total_params))

        print(" ")
        print(table)
        if global_warning_flag_macs:
            print("WARNING: Total MAC count may be incorrect, as tool identified at least one operator for which MACs could not be determined")
            print("         Using the simplified model, generated with --simplify argument, may resolve this")
        print("Disclaimer: model_MACs are based on certain assumptions; see README")
        print("            Layer specific operator information, via --node_csv, can be used to verify MACs")


    def io(self):
        """
        Print model inputs/outputs name, shape, type, size.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        sess = ort.InferenceSession(self.name)
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()

        table = Table(title="Model Inputs/Outputs Overview")
        table.add_column('ID')
        table.add_column('Name')
        table.add_column('Shape')
        table.add_column('Type')
        table.add_column('ByteSize')
        #print inputs
        i = 0
        for inp in inputs:
            dim_input = np.prod(inp.shape)
            if inp.type == "tensor(float16)":
                size_input = dim_input*2
            if inp.type == "tensor(float)":
                size_input = dim_input*4
            else: size_input = dim_input
            table.add_row("input  "+str(i), inp.name, str(inp.shape), str(inp.type), str(size_input))
            i = i + 1
        #print outputs
        i = 0
        for outp in outputs:
            dim_output = np.prod(outp.shape)
            if outp.type == "tensor(float16)":
                size_output = dim_output*2
            if outp.type == "tensor(float)":
                size_output = dim_output*4
            else: size_output = dim_output
            table.add_row("output "+str(i), outp.name, str(outp.shape), str(outp.type), str(size_output))
            i += 1

        print(" ")
        print(table)

    def op(self):
        """
        Print model operators summary table.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        sum_op = []
        for node in self.model.graph.node:
            sum_op.append(node.op_type)

        table = Table(title = "Model Operators Overview", show_lines=True)
        table.add_column("Op Name")
        table.add_column("#")
        for op_type in sorted(set(sum_op)):
            table.add_row(op_type, str(sum_op.count(op_type)))
            #print(op_type + "   " +str(sum_op.count(op_type)))
        table.add_row("Total", str(len(sum_op)), style="green")
        #print("Total" + "   " + str(len(sum_op)))

        print(" ")
        print(table)

    def infoshape(self):
        """
        Construct a shape dictionary for each tensor  {tensor_name0:shape0, tensor_name1:shape1, ...}

        Parameters
        ----------
        None

        Returns
        -------
        Shape dictionary
        """

        info_shape = {}
        attrs = ["input", "value_info", "output", "initializer"]

        for att in attrs:
            for item in getattr(self.shape.graph, att):
                name = item.name
                shape = []
                if att == "initializer": 
                    shape = item.dims
                else:
                    info_dim = item.type.tensor_type.shape.dim
                    for dim in info_dim:
                        shape.append(dim.dim_value)
                info_shape[name]=shape
        
        return info_shape

    def infoparam(self):
        """
        Calculate the params size for each node and the total params size of the model.

        Parameters
        ----------
        None

        Returns
        -------
        Node params dictionary and total size of model params
        """

        info_param = {}
        total_params = 0
        for init in self.model.graph.initializer:
            info_param[init.name]=init.dims
            weight = numpy_helper.to_array(init)
            total_params += np.prod(weight.shape)
        for node in self.model.graph.node:
            if node.op_type == "Constant":
                info_param[node.name] = node.attribute[0].name
                weight = numpy_helper.to_array(node.attribute[0].t)
                total_params += np.prod(weight.shape)

        return info_param, total_params

    def infonode(self):
        """
        Construct insight dictionary for each node including name, type, inputs/outputs name/shape, macs, params, attrs.

        Parameters
        ----------
        None

        Returns
        -------
        The model list of node dictionaries, total macs number.
        """

        global global_warning_flag_macs
        dic_model = []
        total_macs = 0
        for node in self.model.graph.node:
            dic_node = {}
            dic_node["name"] = node.name
            dic_node["type"] = node.op_type
            dic_node["inputs"] = {}

            #each node may contain two types inputs: one is from previous layer, another one is the weight/bias params
            i = 0
            node_param = 0
            info_shape = self.infoshape()
            info_param, _ = self.infoparam()
            for inp in node.input:
                if inp in info_shape:
                    name_inp = "name" + str(i)
                    name_shape = "shape" + str(i)
                    dic_node["inputs"][name_inp] = inp
                    dic_node["inputs"][name_shape] = info_shape[inp]
                    i = i + 1
                if inp in info_param:
                    node_param += np.prod(info_param[inp])
            dic_node["params"] = int(node_param)

            #each node only has one output
            dic_node["output"] = {}
            dic_node["output"]["name"] = node.output[0]
            dic_node["output"]["shape"] = info_shape[node.output[0]]
            #print(dic_node["output"]["shape"])
            if node.op_type == "Constant":
                dic_node["params"] = int(np.prod(info_shape[node.output[0]]))
                
            if node.op_type == "Conv" or node.op_type == "ConvTranspose":
                if len(node.attribute) != 0:
                    dic_node["attrs"] = {}
                    dic_node["attrs"]["group"] = 1
                    dic_node["attrs"]["strides"] = [1,1]
                    dic_node["attrs"]["dilations"] = [1,1]
                    attr_shape = []
                    attr_shape.append(info_shape[node.input[1]][2])
                    attr_shape.append(info_shape[node.input[1]][3])
                    dic_node["attrs"]["kernel_shape"] = attr_shape
                    for att in node.attribute:
                        if att.name == "group":
                             dic_node["attrs"][att.name] = att.i
                        else:
                            attr_shape = []
                            for attr_shape_dim in att.ints:
                                attr_shape.append(attr_shape_dim)
                            dic_node["attrs"][att.name] = attr_shape

            if node.op_type == "MaxPool" or node.op_type == "AveragePool":
                if len(node.attribute) != 0:
                    dic_node["attrs"] = {}
                    dic_node["attrs"]["strides"] = [1,1]
                    dic_node["attrs"]["dilations"] = [1,1]
                    dic_node["attrs"]["pads"] = [0,0,0,0]
                    for att in node.attribute:
                        attr_shape = []
                        for attr_shape_dim in att.ints:
                            attr_shape.append(attr_shape_dim)
                        dic_node["attrs"][att.name] = attr_shape

            if node.op_type == "Gemm":
                dic_node["attrs"] = {}
                dic_node["attrs"]["transA"] = 0
                dic_node["attrs"]["transB"] = 0
                if len(node.attribute) != 0:
                    for att in node.attribute:
                        if att.name == "transA":
                             dic_node["attrs"][att.name] = att.i
                        elif att.name == "transB":
                             dic_node["attrs"][att.name] = att.i

            #Calculate MACs number for Conv/ConvTranspose/Gemm/MatMul ops.
            if node.op_type == "Conv":
                #print(node)
                #print("groups ", dic_node["attrs"]["group"])
                try:
                    node_macs = np.prod(info_shape[node.output[0]])*(info_shape[node.input[0]][1]*np.prod(dic_node["attrs"]["kernel_shape"]))/dic_node["attrs"]["group"]
                except:
                    node_macs = 0
                    global_warning_flag_macs  = True
                    print("Warning, could not determine MACs for this Conv")
                    print(node)

                dic_node["macs"] = int(node_macs)
                total_macs += node_macs
            if node.op_type == "ConvTranspose":
                #print(node)
                #print("groups ", dic_node["attrs"]["group"])
                try:
                    node_macs = np.prod(info_shape[node.input[0]])*(info_shape[node.output[0]][1]*np.prod(dic_node["attrs"]["kernel_shape"]))/dic_node["attrs"]["group"]
                except:
                    node_macs = 0
                    global_warning_flag_macs  = True
                    print("Warning, could not determine MACs for this ConvTranspose")
                    print(node)

                dic_node["macs"] = int(node_macs)
                total_macs += node_macs
            if node.op_type == "Gemm":
                if info_shape[node.input[0]] == [] or info_shape[node.output[0]] == []:
                    print("input1 shape ", info_shape[node.input[0]])
                    print("input2 shape ", info_shape[node.input[1]])
                    print(node)
                    print("calculate macs number failed, please try run --simplified, then use the simplified model")
                    print("WARNING: the total MAC count will not include this operator")
                    global_warning_flag_macs  = True
                else:
                    #the way mac are calcuated, we do not need to consider transA and transB arguments.
                    #print(node)
                    #print("input1 shape ", info_shape[node.input[0]])
                    #print("input2 shape ", info_shape[node.input[1]])
                    #print("output shape ", info_shape[node.output[0]])
                    #print("transA ", dic_node["attrs"]["transA"])
                    #print("transB ", dic_node["attrs"]["transB"])
                    try:
                        node_macs = info_shape[node.input[0]][0]*info_shape[node.input[0]][1]*info_shape[node.output[0]][1]
                        if node_macs == 0:
                            global_warning_flag_macs  = True
                            print("Warning, could not determine MACs for this MatMul")
                            print(node)
                            print("input1 shape ", info_shape[node.input[0]])
                            print("input2 shape ", info_shape[node.input[1]])
                    except:
                        node_macs = 0
                        global_warning_flag_macs  = True
                        print("Warning, could not determine MACs for this Gemm")
                        print(node)

                    dic_node["macs"] = int(node_macs)
                    total_macs += node_macs
            if node.op_type == "MatMul":
                if info_shape[node.input[0]] == [] or info_shape[node.input[1]] == []:
                    print("input1 shape ", info_shape[node.input[0]])
                    print("input2 shape ", info_shape[node.input[1]])
                    print(node)
                    print("calculate macs number failed, please try run --simplified, then use the simplified model")
                    print("WARNING: the total MAC count will not include this operator")
                    global_warning_flag_macs  = True
                else:
                    #print(node)
                    #print("input1 shape ", info_shape[node.input[0]])
                    #print("input1 dim   ", len(info_shape[node.input[0]]))
                    #print("input1 last dim   ", info_shape[node.input[0]][len(info_shape[node.input[0]])-1])
                    #print("input2 shape ", info_shape[node.input[1]])
                    #print("input2 dim   ", len(info_shape[node.input[1]]))
                    #print("input2 second to last dim   ", info_shape[node.input[1]][len(info_shape[node.input[1]])-2])
                    #print("output shape ", info_shape[node.output[0]])
                    if info_shape[node.input[0]][len(info_shape[node.input[0]])-1] != info_shape[node.input[1]][len(info_shape[node.input[1]])-2]:
                        print("input1 shape ", info_shape[node.input[0]])
                        print("input2 shape ", info_shape[node.input[1]])
                        print(node)
                        print("last dimension of first input does not match second to last dimension of second input")
                        print("calculate macs number failed, please try run --simplified, then use the simplified model")
                        print("WARNING: the total MAC count will not include this operator")
                        global_warning_flag_macs  = True
                    else:
                        # 2, 3 and 4 dimensional tensors cases
                        # 2D: [M,K][K,N] = [M,N]
                        # 3D: [I,M,K][I,K,N] = [I,M,N]
                        # 3D: [I,M,K][K,N] = [I,M,N]
                        # 4D: [I,J,M,K][I,J,K,N] = [I,J,M,N]
                        # 4D: [I,J,M,K][J,K,N] = [I,J,M,N]
                        # 4D: [I,J,M,K][K,N] = [I,J,M,N]
                        #node_macs = np.prod(info_shape[node.output[0]])*info_shape[node.input[0]][len(info_shape[node.input[0]])-1]
                        try:
                            node_macs = np.prod(info_shape[node.output[0]])*info_shape[node.input[0]][len(info_shape[node.input[0]])-1]
                            if node_macs == 0:
                                global_warning_flag_macs  = True
                                print("Warning, could not determine MACs for this MatMul")
                                print(node)
                                print("input1 shape ", info_shape[node.input[0]])
                                print("input2 shape ", info_shape[node.input[1]])
                        except:
                            node_macs = 0
                            global_warning_flag_macs  = True
                            print("Warning, could not determine MACs for this MatMul")
                            print(node)

                        dic_node["macs"] = int(node_macs)
                        total_macs += node_macs
            dic_model.append(dic_node)

        return dic_model, total_macs
