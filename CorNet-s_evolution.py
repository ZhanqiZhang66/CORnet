"""
Evolution using specific time step from CorNet-S units.
Binxu
Jan.29th, 2022
"""
import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint
import numpy as np
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
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# sys.path.append("E:\Github_Projects\ActMax-Optimizer-Dev")                 #Binxu local 
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\ActMax-Optimizer-Dev")   #Victoria local
#sys.path.append(r"\data\Victoria\UCSD_projects\ActMax-Optimizer-Dev")       #Victoria remote

from core.GAN_utils import upconvGAN
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid
from core.layer_hook_utils import featureFetcher, get_module_names, get_layer_names
from collections import defaultdict

#%%

class featureFetcher_recurrent:
    """ Light weighted modular feature fetcher, simpler than TorchScorer. """
    def __init__(self, model, input_size=(3, 224, 224), device="cuda", print_module=True):
        self.model = model.to(device)
        module_names, module_types, module_spec = get_module_names(model, input_size, device=device, show=print_module)
        self.module_names = module_names
        self.module_types = module_types
        self.module_spec = module_spec
        self.activations = defaultdict(list)
        self.hooks = {}
        self.device = device

    def record(self, module, submod, key="score", return_input=False, ingraph=False):
        hook_fun = self.get_activation(key, ingraph=ingraph, return_input=return_input)
        hook_h = getattr(getattr(self.model, module), submod).register_forward_hook(hook_fun)
        #register_hook_by_module_names(target_name, hook_fun, self.model, device=self.device)
        self.hooks[key] = hook_h
        return hook_h

    def __del__(self):
        for name, hook in self.hooks.items():
            hook.remove()
        return

    def __getitem__(self, key):
        try:
            return self.activations[key]
        except KeyError:
            raise KeyError

    def get_activation(self, name, ingraph=False, return_input=False): 
        """If returning input, it may return a list or tuple of things """
        if return_input:
            def hook(model, input, output):
                self.activations[name].append(input if ingraph else [inp.detach().cpu() for inp in input])
        else:
            def hook(model, input, output):
                self.activations[name].append(output if ingraph else output.detach().cpu())

        return hook
#%%
"""
Actually, if you use higher version of pytorch, the torch transform could work...
Lower version you need to manually write the preprocessing function. 
"""
# imsize = 224
# normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                              std=[0.229, 0.224, 0.225])
# preprocess_fun = torchvision.transforms.Compose([
#                     torchvision.transforms.Resize((imsize, imsize)),
#                     normalize,
#                 ])
RGBmean = torch.tensor([0.485, 0.456, 0.406]).reshape([1,3,1,1]).cuda()
RGBstd  = torch.tensor([0.229, 0.224, 0.225]).reshape([1,3,1,1]).cuda()
def preprocess_fun(imgtsr, imgsize=224, ):
    """Manually write some version of preprocessing"""
    imgtsr = nn.functional.interpolate(imgtsr, [imgsize,imgsize])
    return (imgtsr - RGBmean) / RGBstd
#%% Prepare model
G = upconvGAN("fc6")
G.eval().cuda().requires_grad_(False)

model = get_model(pretrained=True)
model.eval().requires_grad_(False)
#%% Evolution parameters and Optimzer

area = "IT"
sublayer = "conv3"
time_step = 0
channum = 0
pos = 3, 3

import random
def run_evolution(model, area, sublayer, time_step, channum):
    pos = 3, 3
    print("Evolve from {}_{}_Time {}_ Channel {}".format(area, sublayer, str(time_step), str(channum)))
    fetcher = featureFetcher_recurrent(model, print_module=False)
    h = fetcher.record(area, sublayer, "target")
    optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
    codes = optim.get_init_pop()
    for i in range(100):
        # get score
        fetcher.activations["target"] = []
        with torch.no_grad():
            model(preprocess_fun(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))))
        scores = fetcher["target"][time_step][:, channum, pos[0], pos[1]]
        # optimizer update
        newcodes = optim.step_simple(scores, codes)
        codes = newcodes
        del newcodes
        # print(f"Gen {i:d} {scores.mean():.3f}+-{scores.std():.3f}")
    return fetcher, codes, scores
#%%
from core.montage_utils import ToPILImage, make_grid
time_steps = [0, 1]
area = "IT"
sublayer = "conv3"


C = 512 # np.shape(fetcher["target"][time_step])[1]
channums = random.sample(range(C), 200)
for channum in channums:
    for time_step in time_steps:
        for i in range(3):
            fetcher, codes, scores = run_evolution(model, area, sublayer, time_step, channum)
            pil_image = ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu()))
            # filename = "N:\\Users\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(
            #     area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(), ".2f"))

            filename = "D:\\Ponce-Lab\\Victoria\\Victoria_data\\CORnet_evolution\\{}_{}_time_{}_chan_{}_trial_{}_score{}.png".format(area, sublayer, str(time_step), str(channum), str(i), format(scores.mean(),".2f"))
            pil_image.save(filename)
            del codes, pil_image




#         #%%
# """
# Another seemingly more succint version, more wrapped up
# """
# def get_score_fun(model, area, sublayer, time_step, channum, pos, ):
#     fetcher = featureFetcher_recurrent(model, print_module=False)
#     h = fetcher.record(area, sublayer, "target")
#     def score_fun(imgs):
#         fetcher.activations["target"] = []
#         with torch.no_grad():
#             model(imgs)
#         if fetcher["target"][time_step].ndim == 4:
#             scores = fetcher["target"][time_step][:, channum, pos[0], pos[1]]
#         elif fetcher["target"][time_step].ndim == 2:
#             scores = fetcher["target"][time_step][:, channum, ]
#         else:
#             raise NotImplementedError
#         return scores
#
#     return score_fun, fetcher
#
#
# # area = "IT"
# # sublayer = "conv3"
# # time_step = 0
# # channum = 0
# # pos = 3, 3
# score_fun, fetcher = get_score_fun(model, area="IT", sublayer="conv3", time_step=1, channum=0, pos=(3, 3), )
# optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
# codes = optim.get_init_pop()
# for i in range(75):
#     with torch.no_grad():
#         scores = score_fun(preprocess_fun(G.visualize(torch.tensor(codes,
#                                    dtype=torch.float32, device="cuda")))) #.reshape([-1,4096])
#     newcodes = optim.step_simple(scores, codes)
#     codes = newcodes
#     print(f"Gen {i:d} {scores.mean():.3f}+-{scores.std():.3f}")
#
# # del fetcher, score_fun
# #%%
# from core.montage_utils import ToPILImage, make_grid
# ToPILImage()(make_grid(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda")).cpu())).show()
