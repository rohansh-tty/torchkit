import torch
from torchsummary import summary
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


def torch2onnx(model, input_size=(28,28), batch_size=128, name='.onnx', weights='.pth' ):
    # input_size should be in (CH, H, W) format
    # Load pretrained model weights
    H, W = input_size[1], input_size[2]
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number

    torch_model = model
    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()
    
        
    # Input to the model
    x = torch.randn(batch_size, 1, H, W, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    name,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    return 'SUCCESS'

def savetorchmodel(modelname, path=''):
    torch.save(modelname.state_dict(), path)
    return 'SUCCESS'