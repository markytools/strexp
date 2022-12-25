import settings
import captum
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils import get_args
from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
import string
import time
import sys
from dataset import hierarchical_dataset, AlignCollate
import validators
from model import Model, STRScore
from PIL import Image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from captum._utils.models.linear_model import SkLearnLinearModel, SkLearnRidge
import random
import os
from skimage.color import gray2rgb
import pickle
from train_shap_corr import getPredAndConf
import re
import copy
import statistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    Deconvolution,
    GuidedGradCam,
    FeatureAblation,
    ShapleyValueSampling,
    Lime,
    KernelShap
)

from captum.metrics import (
    infidelity,
    sensitivity_max
)

### Returns the mean for each segmentation having shape as the same as the input
### This function can only one attribution image at a time
def averageSegmentsOut(attr, segments):
    averagedInput = torch.clone(attr)
    sortedDict = {}
    for x in np.unique(segments):
        segmentMean = torch.mean(attr[segments == x][:])
        sortedDict[x] = float(segmentMean.detach().cpu().numpy())
        averagedInput[segments == x] = segmentMean
    return averagedInput, sortedDict

### Output and save segmentations only for one dataset only
def outputSegmOnly(opt):
    ### targetDataset - one dataset only, SVTP-645, CUTE80-288images
    targetDataset = "CUTE80" # ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    segmRootDir = "/home/uclpc1/Documents/STR/datasets/segmentations/224X224/{}/".format(targetDataset)

    if not os.path.exists(segmRootDir):
        os.makedirs(segmRootDir)

    opt.eval = True
    ### Only IIIT5k_3000
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = [targetDataset]
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = [targetDataset]

    ### Taken from LIME
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=random.randint(0, 1000))

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        for i, (image_tensors, labels) in enumerate(evaluation_loader):
            imgDataDict = {}
            img_numpy = image_tensors.cpu().detach().numpy()[0] ### Need to set batch size to 1 only
            if img_numpy.shape[0] == 1:
                img_numpy = gray2rgb(img_numpy[0])
            # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
            segmOutput = segmentation_fn(img_numpy)
            imgDataDict['segdata'] = segmOutput
            imgDataDict['label'] = labels[0]
            outputPickleFile = segmRootDir + "{}.pkl".format(i)
            with open(outputPickleFile, 'wb') as f:
                pickle.dump(imgDataDict, f)

def acquireSelectivityHit(origImg, attributions, segmentations, model, converter, labels, scoring):
    # print("segmentations unique len: ", np.unique(segmentations))
    aveSegmentations, sortedDict = averageSegmentsOut(attributions[0,0], segmentations)
    sortedKeys = [k for k, v in sorted(sortedDict.items(), key=lambda item: item[1])]
    sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
    # print("sortedDict: ", sortedDict) # {0: -5.51e-06, 1: -1.469e-05, 2: -3.06e-05,...}
    # print("aveSegmentations unique len: ", np.unique(aveSegmentations))
    # print("aveSegmentations device: ", aveSegmentations.device) # cuda:0
    # print("aveSegmentations shape: ", aveSegmentations.shape) # (224,224)
    # print("aveSegmentations: ", aveSegmentations)

    n_correct = []
    confidenceList = [] # First index is one feature removed, second index two features removed, and so on...
    clonedImg = torch.clone(origImg)
    gt = str(labels)
    for totalSegToHide in range(0, len(sortedKeys)):
        ### Acquire LIME prediction result
        currentSegmentToHide = sortedKeys[totalSegToHide]
        clonedImg[0,0][segmentations == currentSegmentToHide] = 0.0
        pred, confScore = getPredAndConf(opt, model, scoring, clonedImg, converter, np.array([gt]))
        # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
        if opt.sensitive and opt.data_filtering_off:
            pred = pred.lower()
            gt = gt.lower()
            alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
        if pred == gt:
            n_correct.append(1)
        else:
            n_correct.append(0)
        confScore = confScore[0][0]*100
        confidenceList.append(confScore)
    return n_correct, confidenceList

### Once you have the selectivity_eval_results.pkl file,
def acquire_selectivity_auc(opt, pkl_filename=None):
    if pkl_filename is None:
        pkl_filename = "/home/goo/str/str_vit_dataexplain_lambda/metrics_sensitivity_eval_results_CUTE80.pkl" # VITSTR
    accKeys = []

    with open(pkl_filename, 'rb') as f:
        selectivity_data = pickle.load(f)

    for resDictIdx, resDict in enumerate(selectivity_data):
        keylistAcc = []
        keylistConf = []
        metricsKeys = resDict.keys()
        for keyStr in resDict.keys():
            if "_acc" in keyStr: keylistAcc.append(keyStr)
            if "_conf" in keyStr: keylistConf.append(keyStr)
        # Need to check if network correctly predicted the image
        for metrics_accStr in keylistAcc:
            if 1 not in resDict[metrics_accStr]: print("resDictIdx")

## gtClassNum - set to gtClassNum=0 for standard implemention, or specific class idx for local explanation
def acquireAttribution(opt, super_model, input, segmTensor, gtClassNum, lowestAccKey, device):
    channels = 1
    if opt.rgb:
        channels = 3

    ### Perform attribution
    if "intgrad_" in lowestAccKey:
        ig = IntegratedGradients(super_model)
        attributions = ig.attribute(input, target=gtClassNum)
    elif "gradshap_" in lowestAccKey:
        gs = GradientShap(super_model)
        baseline_dist = torch.zeros((1, channels, opt.imgH, opt.imgW))
        baseline_dist = baseline_dist.to(device)
        attributions = gs.attribute(input, baselines=baseline_dist, target=gtClassNum)
    elif "deeplift_" in lowestAccKey:
        dl = DeepLift(super_model)
        attributions = dl.attribute(input, target=gtClassNum)
    elif "saliency_" in lowestAccKey:
        saliency = Saliency(super_model)
        attributions = saliency.attribute(input, target=gtClassNum)
    elif "inpxgrad_" in lowestAccKey:
        input_x_gradient = InputXGradient(super_model)
        attributions = input_x_gradient.attribute(input, target=gtClassNum)
    elif "guidedbp_" in lowestAccKey:
        gbp = GuidedBackprop(super_model)
        attributions = gbp.attribute(input, target=gtClassNum)
    elif "deconv_" in lowestAccKey:
        deconv = Deconvolution(super_model)
        attributions = deconv.attribute(input, target=gtClassNum)
    elif "featablt_" in lowestAccKey:
        ablator = FeatureAblation(super_model)
        attributions = ablator.attribute(input, target=gtClassNum, feature_mask=segmTensor)
    elif "shapley_" in lowestAccKey:
        svs = ShapleyValueSampling(super_model)
        attributions = svs.attribute(input, target=gtClassNum, feature_mask=segmTensor)
    elif "lime_" in lowestAccKey:
        interpretable_model = SkLearnRidge(alpha=1, fit_intercept=True) ### This is the default used by LIME
        lime = Lime(super_model, interpretable_model=interpretable_model)
        attributions = lime.attribute(input, target=gtClassNum, feature_mask=segmTensor)
    elif "kernelshap_" in lowestAccKey:
        ks = KernelShap(super_model)
        attributions = ks.attribute(input, target=gtClassNum, feature_mask=segmTensor)
    else:
        assert False
    return attributions

### In addition to acquire_average_auc(), this function also returns the best selectivity_acc attr-based method
### pklFile - you need to pass pkl file here
def acquire_bestacc_attr(opt, pickleFile):
    # pickleFile = "metrics_sensitivity_eval_results_IIIT5k_3000.pkl"
    # pickleFile = "/home/goo/str/str_vit_dataexplain_lambda/shapley_singlechar_ave_matrn_SVT.pkl"
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
                if ("_acc" in keyStr or "_conf" in keyStr) and not ("_local_" in keyStr or "_global_local_" in keyStr): # Accept only selectivity
                    if keyStr not in metricDict:
                        metricDict[keyStr] = []
                    dataList = copy.deepcopy(imgData[keyStr]) # list of 0,1 [1,1,1,0,0,0,0]
                    dataList.insert(0, 1) # Insert 1 at beginning to avoid np.trapz([1]) = 0.0
                    denom = [1] * len(dataList) # Denominator to normalize AUC
                    auc_norm = np.trapz(dataList) / np.trapz(denom)
                    metricDict[keyStr].append(auc_norm)
        elif acquireInfidelity:
            pass # TODO
        elif acquireSensitivity:
            pass # TODO

    lowestAccKey = ""
    lowestAcc = 10000000
    for metricKey in metricDict:
        if "_acc" in metricKey: # Used for selectivity accuracy only
            statisticVal = statistics.mean(metricDict[metricKey])
            if statisticVal < lowestAcc:
                lowestAcc = statisticVal
                lowestAccKey = metricKey
        # print("{}: {}".format(metricKey, statisticVal))

    assert lowestAccKey!=""
    return lowestAccKey

def saveAttrData(filename, attribution, segmData, origImg):
    pklData = {}
    pklData['attribution'] = torch.clone(attribution).detach().cpu().numpy()
    pklData['segmData'] = segmData
    pklData['origImg'] = origImg
    with open(filename, 'wb') as f:
        pickle.dump(pklData, f)

### New code (8/3/2022) to acquire average selectivity, infidelity, etc. after running captum test
def acquire_average_auc(opt):
    # pickleFile = "metrics_sensitivity_eval_results_IIIT5k_3000.pkl"
    pickleFile = "/home/goo/str/str_vit_dataexplain_lambda/shapley_singlechar_ave_vitstr_IC03_860.pkl"
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
                    metricDict[keyStr].append(auc_norm)
        elif acquireInfidelity:
            pass # TODO
        elif acquireSensitivity:
            pass # TODO

    for metricKey in metricDict:
        print("{}: {}".format(metricKey, statistics.mean(metricDict[metricKey])))

### Use this acquire list
def acquireListOfAveAUC(opt):
    acquireSelectivity = True
    acquireInfidelity = False
    acquireSensitivity = False
    totalChars = 10
    collectedMetricDict = {}
    for charNum in range(0, totalChars):
        pickleFile = f"/home/goo/str/str_vit_dataexplain_lambda/singlechar{charNum}_results_{totalChars}chardataset.pkl"
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
                        metricDict[keyStr].append(auc_norm)
        for metricKey in metricDict:
            selec_auc_normalize = statistics.mean(metricDict[metricKey])
            if metricKey not in collectedMetricDict:
                collectedMetricDict[metricKey] = []
            collectedMetricDict[metricKey].append(selec_auc_normalize)
    for collectedMetricDictKey in collectedMetricDict:
        print("{}: {}".format(collectedMetricDictKey, collectedMetricDict[collectedMetricDictKey]))
    for charNum in range(0, totalChars):
        selectivityAcrossCharsLs = []
        for collectedMetricDictKey in collectedMetricDict:
            if "_acc" in collectedMetricDictKey:
                selectivityAcrossCharsLs.append(collectedMetricDict[collectedMetricDictKey][charNum])
        print("accuracy -- {}: {}".format(charNum, statistics.mean(selectivityAcrossCharsLs)))
    for charNum in range(0, totalChars):
        selectivityAcrossCharsLs = []
        for collectedMetricDictKey in collectedMetricDict:
            if "_conf" in collectedMetricDictKey:
                selectivityAcrossCharsLs.append(collectedMetricDict[collectedMetricDictKey][charNum])
        print("confidence -- {}: {}".format(charNum, statistics.mean(selectivityAcrossCharsLs)))

if __name__ == '__main__':
    # deleteInf()
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    main(opt)
