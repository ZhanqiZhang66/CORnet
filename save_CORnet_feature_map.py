import numpy as np
#%% Generate features per layer

import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
import torch.utils.model_zoo
import torchvision

import cornet

from PIL import Image
#%%

#%%
# def get_model(model_name, pretrained=False):
#     map_location = 'cpu'
#     model = getattr(cornet, model_name)
#
#     model = model(pretrained=pretrained, map_location=map_location)
#
#
#     model = model.module  # remove DataParallel
#
#     return model
#%%
def get_model(model_name, pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, f'cornet_{model_name.lower()}')

    model = model(pretrained=pretrained, map_location=map_location)

    return model
#%%
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

def test1(modelname,data_path, output_path, layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
    """
    Suitable for small image sets. If you have thousands of images or it is
    taking too long to extract features, consider using
    `torchvision.datasets.ImageFolder`, using `ImageNetVal` as an example.

    Kwargs:
        - layers (choose from: V1, V2, V4, IT, decoder)
        - sublayer (e.g., output, conv1, avgpool)
        - time_step (which time step to use for storing features)
        - imsize (resize image to how many pixels, default: 224)
    """
    model = get_model(modelname, pretrained=True)
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((imsize, imsize)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])
    model.eval()

    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        #_model_feats.append(np.reshape(output, (len(output), -1)))
        _model_feats.append(output)

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model_feats = []
    model_pic_name = []
    with torch.no_grad():
        print("{}, {}, {}, {}".format(modelname, layer, sublayer, time_step))
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {data_path}')
        for fname in tqdm.tqdm(fnames):
            # print(fname)
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            # print(np.shape(_model_feats))
            if modelname == 'S':
                model_feats.append(_model_feats[time_step])
            else:
                model_feats.append(_model_feats)
            #model_pic_name.append(fname)
        model_feats = np.concatenate(model_feats)
        #model_pic_name = np.concatenate(model_pic_name)


    if output_path is not None:
        fname = f'CORnet-{modelname}_{layer}_{sublayer}_{time_step}_w_bn_relu_feats.npy'

        np.save(os.path.join(output_path, fname), model_feats)
    #return model_pic_name


#%%
input_path = r'D:\OneDrive - UC San Diego\GitHub\CORnet\heatmap_pictures'
output_path = r'D:\OneDrive - UC San Diego\GitHub\CORnet\output_images'

#%%
model_names = ['S']
for model_name in model_names:
    #model = get_model(model_name, pretrained=True)
    if model_name == 'S':
        print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        layers_to_test = {'V1': 'conv2',
                          'V1': 'nonlin2',  #'conv2',
                          'V2': 'conv3',  # 'conv3',
                          'V2': 'nonlin3',  #'conv3',
                          'V4': 'conv3',  # 'conv3',
                          'V4': 'nonlin3',  #'conv3',
                          'IT': 'conv3',  # 'conv3'
                          'IT': 'nonlin3'  #'conv3'
                          }
        for key,value in layers_to_test.items():
            layer = key
            sublayer = value
            if key == 'V1':
                sublayers = ['conv2','nonlin2']
                time_step = 0
                for sublayer in sublayers:
                    test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                          imsize=224)

            elif key == 'V2':
                # time_step = 1 #0, 1
                time_steps = [0, 1]
                sublayers = ['conv3', 'nonlin3']
                for time_step in time_steps:
                    for sublayer in sublayers:
                        test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                              imsize=224)
            elif  key == 'V4':
                time_steps = [0, 1, 2, 3]
                sublayers = ['conv3', 'nonlin3']
                for time_step in time_steps:
                    for sublayer in sublayers:
                        test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                              imsize=224)
            else:
                time_steps = [0, 1]
                sublayers = ['conv3', 'nonlin3']
                for time_step in time_steps:
                    for sublayer in sublayers:
                        test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                              imsize=224)

            # test1(model_name,input_path , output_path, layer=layer, sublayer=sublayer, time_step=time_step, imsize=224)
    else:
        print('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ')
        layers_to_test = {'V1': 'conv',
                          'V1': 'nonlin',  # 'conv2',
                          'V2': 'conv',  # 'conv3',
                          'V2': 'nonlin',  # 'conv3',
                          'V4': 'conv',  # 'conv3',
                          'V4': 'nonlin',  # 'conv3',
                          'IT': 'conv',  # 'conv3'
                          'IT': 'nonlin'  # 'conv3'
                          }
        for key, value in layers_to_test.items():
            layer = key
            sublayer = value
            if key == 'V1':
                time_step = 0
                sublayers = ['conv', 'nonlin']
                for sublayer in sublayers:
                    test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                          imsize=224)

            elif key == 'V2':
                time_step = 1
                sublayers = ['conv', 'nonlin']
                for sublayer in sublayers:
                    test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                          imsize=224)
            elif key == 'V4':
                time_step = 3
                sublayers = ['conv', 'nonlin']
                for sublayer in sublayers:
                    test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                          imsize=224)
            else:
                time_step = 1
                sublayers = ['conv', 'nonlin']
                for sublayer in sublayers:
                    test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step,
                          imsize=224)

            test1(model_name, input_path, output_path, layer=layer, sublayer=sublayer, time_step=time_step, imsize=224)

    #np.save(os.path.join(output_path, fname), model_feats)
#%% Example run
sublayers = ['conv2', 'nonlin2']
for sublayer in sublayers:
    test1('S',input_path , output_path, layer='V1', sublayer=sublayer, time_step=1, imsize=224)
#%%
from collections import OrderedDict
import torch
import torchvision
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
def get_layer_names(model):
    layername = []
    conv_cnt = 0
    fc_cnt = 0
    pool_cnt = 0
    do_cnt = 0
    for layer in list(model.features)+list(model.classifier):
        if isinstance(layer, nn.Conv2d):
            conv_cnt += 1
            layername.append("conv%d" % conv_cnt)
        elif isinstance(layer, nn.ReLU):
            name = layername[-1] + "_relu"
            layername.append(name)
        elif isinstance(layer, nn.MaxPool2d):
            pool_cnt += 1
            layername.append("pool%d"%pool_cnt)
        elif isinstance(layer, nn.Linear):
            fc_cnt += 1
            layername.append("fc%d" % fc_cnt)
        elif isinstance(layer, nn.Dropout):
            do_cnt += 1
            layername.append("dropout%d" % do_cnt)
        else:
            layername.append(layer.__repr__())
    return layername

#%
# Readable names for classic CNN layers in torchvision model implementation.
layername_dict ={"alexnet":["conv1", "conv1_relu", "pool1",
                            "conv2", "conv2_relu", "pool2",
                            "conv3", "conv3_relu",
                            "conv4", "conv4_relu",
                            "conv5", "conv5_relu", "pool3",
                            "dropout1", "fc6", "fc6_relu",
                            "dropout2", "fc7", "fc7_relu",
                            "fc8",],
                "vgg16":['conv1', 'conv1_relu',
                         'conv2', 'conv2_relu', 'pool1',
                         'conv3', 'conv3_relu',
                         'conv4', 'conv4_relu', 'pool2',
                         'conv5', 'conv5_relu',
                         'conv6', 'conv6_relu',
                         'conv7', 'conv7_relu', 'pool3',
                         'conv8', 'conv8_relu',
                         'conv9', 'conv9_relu',
                         'conv10', 'conv10_relu', 'pool4',
                         'conv11', 'conv11_relu',
                         'conv12', 'conv12_relu',
                         'conv13', 'conv13_relu', 'pool5',
                         'fc1', 'fc1_relu', 'dropout1',
                         'fc2', 'fc2_relu', 'dropout2',
                         'fc3'],
                 "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1']}


#%% Hooks based methods to get layer and module names
def named_apply(model, name, func, prefix=None):
    # resemble the apply function but suits the functions here.
    cprefix = "" if prefix is None else prefix + "." + name
    for cname, child in model.named_children():
        named_apply(child, cname, func, cprefix)

    func(model, name, "" if prefix is None else prefix)


def get_module_names(model, input_size, device="cpu", ):
    module_names = OrderedDict()
    module_types = OrderedDict()
    module_spec = OrderedDict()
    def register_hook(module, name, prefix):
        # register forward hook and save the handle to the `hooks` for removal.
        def hook(module, input, output):
            # during forward pass, this hook will append the ReceptiveField information to `receptive_field`
            # if a module is called several times, this hook will append several times as well.
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(module_names)
            module_names[str(module_idx)] = prefix + "." + class_name + name
            module_types[str(module_idx)] = class_name
            module_spec[str(module_idx)] = OrderedDict()
            if isinstance(input[0], torch.Tensor):
                module_spec[str(module_idx)]["inshape"] = tuple(input[0].shape[1:])
            else:
                module_spec[str(module_idx)]["inshape"] = (None,)
            if isinstance(output, torch.Tensor):
                module_spec[str(module_idx)]["outshape"] = tuple(output.shape[1:])
            else:
                module_spec[str(module_idx)]["outshape"] = (None,)
        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                # and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(2, *input_size).type(dtype)

    # create properties
    # receptive_field = OrderedDict()
    module_names["0"] = "Image"
    module_types["0"] = "Input"
    module_spec["0"] = OrderedDict()
    module_spec["0"]["inshape"] = input_size
    module_spec["0"]["outshape"] = input_size
    hooks = []

    # register hook recursively at any module in the hierarchy
    # model.apply(register_hook)
    named_apply(model, "", register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>14}  {:>12}   {:>12}   {:>12}   {:>25} ".format("Layer Id", "inshape", "outshape", "Type", "ReadableStr", )
    print(line_new)
    print("==============================================================================")
    for layer in module_names:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:7} {:8} {:>12} {:>12} {:>15}  {:>25}".format(
            "",
            layer,
            str(module_spec[layer]["inshape"]),
            str(module_spec[layer]["outshape"]),
            module_types[layer],
            module_names[layer],
        )
        print(line_new)
    return module_names, module_types, module_spec
#%%
model = get_model('Z', pretrained=True)
get_module_names(model, [3, 715,715], device="cpu" )

