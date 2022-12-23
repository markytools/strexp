import pickle
import copy
import numpy as np
import statistics
import sys
import os
from captum.attr._utils.visualization import visualize_image_attr
import matplotlib.pyplot as plt

### New code (8/3/2022) to acquire average selectivity, infidelity, etc. after running captum test
def acquire_average_auc():
    # pickleFile = "metrics_sensitivity_eval_results_IIIT5k_3000.pkl"
    pickleFile = "shapley_singlechar_ave_vitstr_IC15_1811.pkl"
    acquireSelectivity = True # If True, set to
    acquireInfidelity = False
    acquireSensitivity = False

    with open(pickleFile, 'rb') as f:
        data = pickle.load(f)
    metricDict = {} # Keys: "saliency_acc", "saliency_conf", "saliency_infid", "saliency_sens"
    selectivity_acc_auc_normalized = [] # Normalized because it is divided by the full rectangle
    for imgData in data:
        if acquireSelectivity:
            for keyStr in imgData.keys():
                if "_acc" in keyStr or "_conf" in keyStr: # Accept only selectivity
                    if keyStr not in metricDict:
                        metricDict[keyStr] = []
                    dataList = copy.deepcopy(imgData[keyStr]) # list of 0,1 [1,1,1,0,0,0,0]
                    dataList.insert(0, 1) # Insert 1 at beginning to avoid np.trapz([1]) = 0.0
                    denom = [1] * len(dataList) # Denominator to normalize AUC
                    auc_norm = np.trapz(dataList) / np.trapz(denom)
                    if not np.isnan(auc_norm).any():
                        metricDict[keyStr].append(auc_norm)
        elif acquireInfidelity:
            pass # TODO
        elif acquireSensitivity:
            pass # TODO

    for metricKey in metricDict:
        print("{}: {}".format(metricKey, statistics.mean(metricDict[metricKey])))

###
def sumOfAllAttributions():
    modelName = "trba"
    datasetName = "IC15_1811" # IIIT5k_3000, IC03_867, IC13_857, IC15_1811
    mainRootDir = "/data/goo/strattr/"
    rootDir = f"{mainRootDir}attributionData/{modelName}/{datasetName}/"
    numpyOutputDir = mainRootDir

    if modelName=="vitstr":
        shape = [1, 1, 224, 224]
    elif modelName =="parseq":
        shape = [1, 3, 32, 128]
    elif modelName =="trba":
        shape = [1, 1, 32, 100]
    # pickleFile = f"shapley_singlechar_ave_{modelName}_{datasetName}.pkl"
    # acquireSelectivity = True
    # with open(pickleFile, 'rb') as f:
    #     data = pickle.load(f)
    # metricDict = {} # Keys: "saliency_acc", "saliency_conf", "saliency_infid", "saliency_sens"
    #
    # for imgData in data:
    #     if acquireSelectivity:
    #         for keyStr in imgData.keys():
    #             print("keyStr: ", keyStr)
    #             if "_acc" in keyStr or "_conf" in keyStr: # Accept only selectivity
    #                 if keyStr not in metricDict:
    #                     metricDict[keyStr] = []
    #                 dataList = copy.deepcopy(imgData[keyStr]) # list of 0,1 [1,1,1,0,0,0,0]
    #                 dataList.insert(0, 1) # Insert 1 at beginning to avoid np.trapz([1]) = 0.0
    #                 denom = [1] * len(dataList) # Denominator to normalize AUC
    #                 auc_norm = np.trapz(dataList) / np.trapz(denom)

    totalImgCount = 0
    # From a folder containing saved attribution pickle files, convert them into attribution images
    for path, subdirs, files in os.walk(rootDir):
        for name in files:
            fullfilename = os.path.join(rootDir, name) # Value
            # fullfilename: /data/goo/strattr/attributionData/trba/CUTE80/66_featablt.pkl
            if "_gl." not in fullfilename.split('/')[-1]: # Accept only global+local
                continue
            totalImgCount += 1
    shape[0] = totalImgCount
    main_np = np.memmap(numpyOutputDir+f"aveattr_{modelName}_{datasetName}.dat", dtype='float32', mode='w+', shape=tuple(shape))

    attrIdx = 0
    # From a folder containing saved attribution pickle files, convert them into attribution images
    leftGreaterRightAcc = 0.0
    for path, subdirs, files in os.walk(rootDir):
        for name in files:
            fullfilename = os.path.join(rootDir, name) # Value
            # fullfilename: /data/goo/strattr/attributionData/trba/CUTE80/66_featablt.pkl
            if "_gl." not in fullfilename.split('/')[-1]: # Accept only global+local
                continue
            print("fullfilename: ", fullfilename)
            # imgNum = int(partfilename.split('_')[0])
            # attrImgName = partfilename.replace('.pkl', '.png')
            # minNumber = min(minNumber, imgNum)
            # maxNumber = max(maxNumber, imgNum)
            with open(fullfilename, 'rb') as f:
                pklData = pickle.load(f)
                attributions = pklData['attribution']
                segmDataNP = pklData['segmData']
                origImgNP = pklData['origImg']
            if np.isnan(attributions).any():
                continue
            # attributions[0] = (attributions[0] - attributions[0].min()) / (attributions[0].max() - attributions[0].min())
            main_np[attrIdx] = attributions[0]
            sumLeft = np.sum(attributions[0,:,:,0:attributions.shape[3]//2])
            sumRight = np.sum(attributions[0,:,:,attributions.shape[3]//2:])
            if sumLeft > sumRight:
                leftGreaterRightAcc += 1.0
            attrIdx += 1
    print("leftGreaterRightAcc: ", leftGreaterRightAcc/attrIdx)
    main_np.flush()
    meanAveAttr = np.transpose(np.mean(main_np, axis=0), (1,2,0))
    print("meanAveAttr shape: ", meanAveAttr.shape) # (1, 3, 32, 128)
    meanAveAttr = 2*((meanAveAttr - meanAveAttr.min()) / (meanAveAttr.max() - meanAveAttr.min())) - 1.0
    mplotfig, _ = visualize_image_attr(meanAveAttr, cmap='RdYlGn') # input should be in (H,W,C)
    mplotfig.savefig(numpyOutputDir+f"aveattr_{modelName}_{datasetName}.png")
    mplotfig.clear()
    plt.close(mplotfig)

if __name__ == '__main__':
    # acquire_average_auc()
    sumOfAllAttributions()
