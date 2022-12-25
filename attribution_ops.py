import os
import pickle
from captum_improve_vitstr import rankedAttributionsBySegm
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from captum.attr._utils.visualization import visualize_image_attr
import torch
import numpy as np

def attr_one_dataset():
    modelName = "vitstr"
    datasetName = "IIIT5k_3000"

    rootDir = f"/data/goo/strattr/attributionData/{modelName}/{datasetName}/"
    attrOutputImgs = f"/data/goo/strattr/attributionDataImgs/{modelName}/{datasetName}/"
    if not os.path.exists(attrOutputImgs):
        os.makedirs(attrOutputImgs)

    minNumber = 1000000
    maxNumber = 0
    # From a folder containing saved attribution pickle files, convert them into attribution images
    for path, subdirs, files in os.walk(rootDir):
        for name in files:
            fullfilename = os.path.join(rootDir, name) # Value
            # fullfilename: /data/goo/strattr/attributionData/trba/CUTE80/66_featablt.pkl
            partfilename = fullfilename[fullfilename.rfind('/')+1:]
            print("fullfilename: ", fullfilename)
            imgNum = int(partfilename.split('_')[0])
            attrImgName = partfilename.replace('.pkl', '.png')
            minNumber = min(minNumber, imgNum)
            maxNumber = max(maxNumber, imgNum)
            with open(fullfilename, 'rb') as f:
                pklData = pickle.load(f)
                attributions = pklData['attribution']
                segmDataNP = pklData['segmData']
                origImgNP = pklData['origImg']
            if np.isnan(attributions).any():
                continue
            attributions = torch.from_numpy(attributions)
            rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
            rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
            rankedAttr = gray2rgb(rankedAttr)
            mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
            mplotfig.savefig(attrOutputImgs + attrImgName)
            mplotfig.clear()
            plt.close(mplotfig)

def attr_all_dataset():
    modelName = "vitstr"

    datasetNameList = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    for datasetName in datasetNameList:
        rootDir = f"/data/goo/strattr/attributionData/{modelName}/{datasetName}/"
        attrOutputImgs = f"/data/goo/strattr/attributionDataImgs/{modelName}/{datasetName}/"
        if not os.path.exists(attrOutputImgs):
            os.makedirs(attrOutputImgs)

        minNumber = 1000000
        maxNumber = 0
        # From a folder containing saved attribution pickle files, convert them into attribution images
        for path, subdirs, files in os.walk(rootDir):
            for name in files:
                fullfilename = os.path.join(rootDir, name) # Value
                # fullfilename: /data/goo/strattr/attributionData/trba/CUTE80/66_featablt.pkl
                partfilename = fullfilename[fullfilename.rfind('/')+1:]
                imgNum = int(partfilename.split('_')[0])
                attrImgName = partfilename.replace('.pkl', '.png')
                minNumber = min(minNumber, imgNum)
                maxNumber = max(maxNumber, imgNum)
                with open(fullfilename, 'rb') as f:
                    pklData = pickle.load(f)
                    attributions = pklData['attribution']
                    segmDataNP = pklData['segmData']
                    origImgNP = pklData['origImg']
                attributions = torch.from_numpy(attributions)
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(attrOutputImgs + attrImgName)
                mplotfig.clear()
                plt.close(mplotfig)

if __name__ == '__main__':
    attr_one_dataset()
    # attr_all_dataset()
