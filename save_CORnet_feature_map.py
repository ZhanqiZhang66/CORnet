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
def get_model(pretrained=False):
    map_location = 'cpu'
    model = getattr(cornet, 'cornet_s')

    model = model(pretrained=pretrained, map_location=map_location)


    model = model.module  # remove DataParallel

    return model
#%%
model = get_model(pretrained=True)

#%%
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

def test1(model, modelname,data_path, output_path, layer='decoder', sublayer='avgpool', time_step=0, imsize=224):
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
    model = get_model(pretrained=True)
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
        model_feats = []
        fnames = sorted(glob.glob(os.path.join(data_path, '*.*')))
        if len(fnames) == 0:
            raise FileNotFoundError(f'No files found in {data_path}')
        for fname in tqdm.tqdm(fnames):
            print(fname)
            try:
                im = Image.open(fname).convert('RGB')
            except:
                raise FileNotFoundError(f'Unable to load {fname}')
            im = transform(im)
            im = im.unsqueeze(0)  # adding extra dimension for batch size of 1
            _model_feats = []
            model(im)
            print(np.shape(_model_feats))
            model_feats.append(_model_feats[time_step])
            #model_pic_name.append(fname)
        model_feats = np.concatenate(model_feats)
        #model_pic_name = np.concatenate(model_pic_name)


    if output_path is not None:
        fname = f'CORnet-{modelname}_{layer}_{sublayer}_{time_step}_feats.npy'

        np.save(os.path.join(output_path, fname), model_feats)
    #return model_pic_name


#%%
input_path = r'D:\OneDrive - Washington University in St. Louis\GitHub\CORnet\heatmap_pictures'
output_path = r'D:\OneDrive - Washington University in St. Louis\GitHub\CORnet\output_images'
layers_to_test = {'V1': 'conv2',
                  'V4': 'conv3',
                  'IT': 'conv3'}

for key,value in layers_to_test.items():
    layer = key
    sublayer = value
    if key == 'V1':
        time_step = 0;
    elif  key == 'V4':
        time_step = 3;
    else:
        time_step = 1;

    test1(model,'S',input_path , output_path, layer=layer, sublayer=sublayer, time_step=time_step, imsize=224)
#np.save(os.path.join(output_path, fname), model_feats)
#%%
input_path = r'D:\OneDrive - Washington University in St. Louis\GitHub\CORnet\heatmap_picture'
test1(model,'S',input_path , output_path, layer='V1', sublayer='conv2', time_step=0, imsize=224)
#%%

v1_nonlinear2im = np.load(r'output_images\CORnet-S_V1_nonlin2_ims.npy')
v1_conv2 = np.load(r'output_images\CORnet-S_V1_conv2_feats.npy')
v1_relu = np.load(r'output_images\CORnet-S_V1_norm2_feats.npy')
v1_nonlinear2 = np.load(r'output_images\CORnet-S_V1_nonlin2_feats.npy')
v1_output = np.load(r'output_images\CORnet-S_V1_output_feats.npy')