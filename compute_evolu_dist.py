from PIL import Image
from collections import Counter
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import os, sys, argparse, time, glob, pickle, subprocess, shlex, io, pprint
from os.path import join
from scipy.spatial.distance import pdist,squareform
#%%
# https://towardsdatascience.com/measuring-similarity-in-two-images-using-python-b72233eb53c6
dataroot = r"N:\Data-Computational\CorNet-recurrent-evol"
# time_steps = [0, 1]
# area = "IT"
#
# outdir = join(dataroot, "%s-output\\"%(area))

#%% across trial
import numpy as np, matplotlib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
labels_cross_trial = ['trial 1', 'trial 2', 'trial 3', 'trial 4', 'trial 5']

for area in ['V4', 'V2']:
    outdir = join(dataroot, "%s-output\\" % (area))
    if area == 'IT' or 'V2':
        time_steps = [0, 1]
    if area == 'V4':
        time_steps = [0, 1, 2, 3]
    for c in range(100):
        for time_step in time_steps:

            mses = np.zeros((5, 5))
            msssims = np.zeros((5, 5))
            uqis = np.zeros((5,5))
            sams = np.zeros((5,5))
            vifps = np.zeros((5,5))
            for i in range(5):
                img_name = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, time_step, i) + '.jpg'
                #grey_img = np.asarray(Image.open(img_name).convert('L'))
                org_img = np.asarray(Image.open(img_name))
                # img = np.asarray(Image.open(img_name)).flatten()
                # D_across_trial.append(img)
                for j in range(5):
                    img_name1 = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, time_step, j) + '.jpg'
                    org_img1 = np.asarray(Image.open(img_name1))
                    mses[i,j] = mse(org_img, org_img1)
                    msssims[i, j] = msssim(org_img, org_img1)
                    uqis[i, j] = uqi(org_img, org_img1)
                    sams[i, j] = sam(org_img, org_img1)
                    vifps[i, j] = vifp(org_img, org_img1)


            # D_across_trial = np.asarray(D_across_trial)
            mses = (mses - np.min(mses))/np.ptp(mses)
            msssims = (msssims - np.min(msssims)) / np.ptp(msssims)
            uqis = (uqis - np.min(uqis)) / np.ptp(uqis)
            sams = (sams - np.min(sams)) / np.ptp(sams)
            vifps = (vifps - np.min(vifps)) / np.ptp(vifps)



            #D_grey_ = np.asarray(D_grey_)

            #pdist_D = 1 / (1 + squareform(pdist(D_across_trial)))
            #pdist_D_grey = 1 / (1 + squareform(pdist(D_grey_)))
            mses = 1 / (1 + mses)
            sams = 1/ (1+sams)

            fig = plt.figure()
            fig.suptitle('{}-Ch{:03d}-T{} Across Trial'.format(area, c, time_step))
            plt.subplot(2, 3, 1)
            mask = np.triu(np.ones_like(mses, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(mses, cmap=cmap, mask=mask, vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f", annot_kws={"fontsize":8})#, xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('MSE')#('Mean Squared Error (MSE) similarity')

            plt.subplot(2, 3, 2)
            sns.heatmap(msssims,  cmap=cmap, mask=mask, vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f", annot_kws={"fontsize":8})#, annot=True)#, xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('MS-SSIM')#'Multi-scale Structural Similarity Index (MS-SSIM)')

            plt.subplot(2, 3, 3)
            sns.heatmap(uqis,cmap=cmap, mask=mask, vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f", annot_kws={"fontsize":8})#, annot=True)#, xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('UQI')#('Universal Quality Image Index (UQI)')

            plt.subplot(2, 3, 4)
            sns.heatmap(sams, cmap=cmap,  mask=mask, vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f", annot_kws={"fontsize":8})#, annot=True)#, xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('SAM')#('Spectral Angle Mapper (SAM)')

            plt.subplot(2, 3, 5)
            sns.heatmap(vifps,cmap=cmap,  mask=mask, vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f", annot_kws={"fontsize":8})#, annot=True)#, xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('VIF')#('Visual Information Fidelity (VIF)')
            plt.tight_layout()

            filename = outdir + '{}-Ch{:03d}-T{}-Across-Trial.png'.format(area, c, time_step)
            print("save" + filename)
            plt.savefig(filename)

#%% Between trial

labels_cross_trial = ['trial 1', 'trial 2', 'trial 3', 'trial 4', 'trial 5']

for area in ['IT', 'V4', 'V2']:
    if area == 'IT' or 'V2':
        time_steps = [0, 1]
    if area == 'V4':
        time_steps = [0, 1, 2, 3]
    for c in range(100):
        c= 1
        for i in range(5):
            l = len(time_steps)
            ds = np.zeros((5, 5))
            for i in range(l - 1):
                img1 = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, time_steps[i], i) + '.jpg'
                img1 = np.asarray(Image.open(img1))
                for j in range(i+1, l):
                    img2 = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, time_steps[j], i) + '.jpg'
                    img2 = np.asarray(Image.open(img2))

                    ds[0, i] = 1/ (1 +mse(img2, img1))
                    ds[1, i] = msssim(img1, img2)
                    ds[2, i] = uqi(img1, img2)
                    ds[3, i] = 1/ (1 + sam(img1, img2))
                    ds[4, i] = vifp(img1, img2)
            ds[0, :] = (ds[0,:] - np.min(ds[0,:]))/np.ptp(ds[0,:])
            ds[1, :] = (ds[1, :] - np.min(ds[1, :])) / np.ptp(ds[1, :])
            ds[2, :] = (ds[2, :] - np.min(ds[2, :])) / np.ptp(ds[2, :])
            ds[3, :] = (ds[3, :] - np.min(ds[3, :])) / np.ptp(ds[3, :])
            ds[4, :] = (ds[4, :] - np.min(ds[4, :])) / np.ptp(ds[4, :])

            fig = plt.figure()
            fig.suptitle('{}-Ch{:03d}-Across-TimeSteps'.format(area, c))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(ds, cmap=cmap,vmin=0, vmax=1, center=0.5,
                        square=True, annot=True, fmt=".1f",
                        annot_kws={"fontsize": 8})  # , xticklabels=labels_cross_trial, yticklabels=labels_cross_trial)
            plt.title('MSE')  # ('Mean Squared Error (MSE) similarity')

#%%
# import numpy as np, matplotlib, matplotlib.pyplot as plt, seaborn as sns
#
# def ImgLabels(N, area, ch, time_step, ax):
#     def offset_image(coord, ax):
#         #print(coord)
#         img_name = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, ch, time_step, coord) + '.jpg'
#         img = Image.open(img_name)
#         im = matplotlib.offsetbox.OffsetImage(img, zoom=0.22)
#         im.image.axes = ax
#
#         for co, xyb in [((0, coord), (-30, -30)),    ((coord, N), (+30, -30))]:
#             ab = matplotlib.offsetbox.AnnotationBbox(im, co,  xybox=xyb,
#                 frameon=False, xycoords='data',  boxcoords="offset points", pad=0)
#             ax.add_artist(ab)
#
#     for i in range(N):
#         offset_image(i, ax)
#
# def HeatMap(N, corr):
#     sns.set_theme(style = "white")
#     #corr = np.random.uniform(-1, 1, size = (N, N))
#     mask = np.triu(np.ones_like(corr, dtype = bool))
#     cmap = sns.diverging_palette(230, 20, as_cmap=True)
#     sns.heatmap(corr, mask=mask, cmap=cmap, vmin=0, vmax=1, center=0.5,
#                 square=True, annot=True, xticklabels=False, yticklabels=False)
#
# N = 5
# f, ax = plt.subplots(figsize=(7,5))
# HeatMap(N, pdist_D)
# ImgLabels(N, area, c, time_step, ax)
# plt.tight_layout(0)
# plt.show()
area = 'IT'
c = 1
i = 0
d_arrar = np.zeros((5, len(time_steps)))
img_name = outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, 0, i) + '.jpg'
img2 = np.asarray(Image.open(img_name))

img_name1= outdir + 'bestimg_{}-output-Ch{:03d}-T{}-run{:02d}'.format(area, c, 1, i) + '.jpg'
img1 = np.asarray(Image.open(img_name1))
print(1 / (1 + mse(img2, img1)))
print(msssim(img1, img2))
print(uqi(img1, img2))
print( 1 / (1 + sam(img1, img2)))
print( vifp(img1, img2))