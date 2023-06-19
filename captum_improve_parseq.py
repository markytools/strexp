import settings
import captum
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
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
from captum_test import acquire_average_auc, acquireListOfAveAUC, acquire_bestacc_attr, acquireAttribution, saveAttrData
import copy
from captum_improve_vitstr import rankedAttributionsBySegm
from matplotlib import pyplot as plt
from captum.attr._utils.visualization import visualize_image_attr

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
        pkl_filename = "metrics_sensitivity_eval_results_CUTE80.pkl" # VITSTR
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

### This acquires the attributes of the STR network on individual character levels,
### then averages them.
def acquireSingleCharAttrAve(opt):
    ### targetDataset - one dataset only, CUTE80 has 288 samples
    # 'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    targetDataset = settings.TARGET_DATASET
    segmRootDir = "{}/32X128/{}/".format(settings.SEGM_DIR, targetDataset)
    outputSelectivityPkl = "strexp_ave_{}_{}.pkl".format(settings.MODEL, targetDataset)
    outputDir = "./attributionImgs/{}/{}/".format(settings.MODEL, targetDataset)
    attrOutputDir = "./attributionData/{}/{}/".format(settings.MODEL, targetDataset)
    ### Set only one below to True to have enough GPU
    acquireSelectivity = True
    acquireInfidelity = False
    acquireSensitivity = False ### GPU error
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(attrOutputDir):
        os.makedirs(attrOutputDir)

    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)
    model = model.to(device)
    model_obj = model
    converter = TokenLabelConverter(opt)

    modelCopy = copy.deepcopy(model)

    """ evaluation """
    scoring_singlechar = STRScore(opt=opt, converter=converter, device=device, enableSingleCharAttrAve=True, model=modelCopy)
    super_pixel_model_singlechar = torch.nn.Sequential(
        # super_pixler,
        # numpy2torch_converter,
        modelCopy,
        scoring_singlechar
    ).to(device)
    modelCopy.eval()
    scoring_singlechar.eval()
    super_pixel_model_singlechar.eval()

    # Single Char Attribution Averaging
    # enableSingleCharAttrAve - set to True
    scoring = STRScore(opt=opt, converter=converter, device=device, model=model)
    super_pixel_model = torch.nn.Sequential(
        # super_pixler,
        # numpy2torch_converter,
        model,
        scoring
    ).to(device)
    model.eval()
    scoring.eval()
    super_pixel_model.eval()

    if opt.blackbg:
        shapImgLs = np.zeros(shape=(1, 1, 224, 224)).astype(np.float32)
        trainList = np.array(shapImgLs)
        background = torch.from_numpy(trainList).to(device)

    opt.eval = True

    ### Only IIIT5k_3000
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = [targetDataset] ### One dataset only
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = [targetDataset]

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    selectivity_eval_results = []

    testImgCount = 0
    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0

    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=random.randint(0, 1000))

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, segmRootDir=segmRootDir)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        testImgCount = 0

    for i, (orig_img_tensors, segAndLabels) in enumerate(evaluation_loader):
        results_dict = {}
        aveAttr = []
        aveAttr_charContrib = []
        segmData, labels = segAndLabels[0]
        target = converter.encode([labels])

        # labels: RONALDO
        segmDataNP = segmData["segdata"]
        segmTensor = torch.from_numpy(segmDataNP).unsqueeze(0).unsqueeze(0)
        # print("segmTensor min: ", segmTensor.min()) # 0 starting segmentation
        segmTensor = segmTensor.to(device)
        img1 = orig_img_tensors.to(device)
        img1.requires_grad = True
        bgImg = torch.zeros(img1.shape).to(device)

        ### Single char averaging
        if settings.MODEL == 'vitstr':
            charOffset = 1
        elif settings.MODEL == 'parseq':
            charOffset = 0
            img1 = transforms.Normalize(0.5, 0.5)(img1) # Between -1 to 1

        # preds = model(img1, seqlen=converter.batch_max_length)
        input = img1
        origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
        origImgNP = gray2rgb(origImgNP)

        ### BASELINE Evaluations

        ### Integrated Gradients
        ig = IntegratedGradients(super_pixel_model)
        attributions = ig.attribute(input, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_intgrad.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_intgrad.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["intgrad_acc"] = n_correct
            results_dict["intgrad_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["intgrad_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(ig.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["intgrad_sens"] = sens

        ### Gradient SHAP using zero-background
        gs = GradientShap(super_pixel_model)
        # We define a distribution of baselines and draw `n_samples` from that
        # distribution in order to estimate the expectations of gradients across all baselines
        baseline_dist = torch.zeros((1, 3, opt.imgH, opt.imgW))
        baseline_dist = baseline_dist.to(device)
        attributions = gs.attribute(input, baselines=baseline_dist, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_gradshap.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_gradshap.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["gradshap_acc"] = n_correct
            results_dict["gradshap_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["gradshap_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(gs.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["gradshap_sens"] = sens

        ### DeepLift using zero-background
        dl = DeepLift(super_pixel_model)
        attributions = dl.attribute(input, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_deeplift.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_deeplift.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["deeplift_acc"] = n_correct
            results_dict["deeplift_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["deeplift_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(dl.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["deeplift_sens"] = sens

        ### Saliency
        saliency = Saliency(super_pixel_model)
        attributions = saliency.attribute(input, target=0) ### target=class0
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_saliency.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_saliency.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["saliency_acc"] = n_correct
            results_dict["saliency_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["saliency_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(saliency.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["saliency_sens"] = sens

        ### InputXGradient
        input_x_gradient = InputXGradient(super_pixel_model)
        attributions = input_x_gradient.attribute(input, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_inpxgrad.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_inpxgrad.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["inpxgrad_acc"] = n_correct
            results_dict["inpxgrad_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["inpxgrad_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(input_x_gradient.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["inpxgrad_sens"] = sens

        ### GuidedBackprop
        gbp = GuidedBackprop(super_pixel_model)
        attributions = gbp.attribute(input, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_guidedbp.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_guidedbp.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["guidedbp_acc"] = n_correct
            results_dict["guidedbp_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["guidedbp_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(gbp.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["guidedbp_sens"] = sens

        ### Deconvolution
        deconv = Deconvolution(super_pixel_model)
        attributions = deconv.attribute(input, target=0)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_deconv.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_deconv.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["deconv_acc"] = n_correct
            results_dict["deconv_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["deconv_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(deconv.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["deconv_sens"] = sens

        ### Feature ablator
        ablator = FeatureAblation(super_pixel_model)
        attributions = ablator.attribute(input, target=0, feature_mask=segmTensor)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_featablt.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_featablt.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["featablt_acc"] = n_correct
            results_dict["featablt_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["featablt_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(ablator.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["featablt_sens"] = sens

        ### Shapley Value Sampling
        svs = ShapleyValueSampling(super_pixel_model)
        # attr = svs.attribute(input, target=0, n_samples=200) ### Individual pixels, too long to calculate
        attributions = svs.attribute(input, target=0, feature_mask=segmTensor)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_shapley.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_shapley.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["shapley_acc"] = n_correct
            results_dict["shapley_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["shapley_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["shapley_sens"] = sens

        ## LIME
        interpretable_model = SkLearnRidge(alpha=1, fit_intercept=True) ### This is the default used by LIME
        lime = Lime(super_pixel_model, interpretable_model=interpretable_model)
        attributions = lime.attribute(input, target=0, feature_mask=segmTensor)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_lime.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_lime.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["lime_acc"] = n_correct
            results_dict["lime_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["lime_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(lime.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["lime_sens"] = sens

        ### KernelSHAP
        ks = KernelShap(super_pixel_model)
        attributions = ks.attribute(input, target=0, feature_mask=segmTensor)
        rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_kernelshap.png'.format(i))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_kernelshap.pkl', attributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
            results_dict["kernelshap_acc"] = n_correct
            results_dict["kernelshap_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions).detach().cpu().numpy())
            results_dict["kernelshap_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(ks.attribute, img1, target=0).detach().cpu().numpy())
            results_dict["kernelshap_sens"] = sens

        selectivity_eval_results.append(results_dict)

        with open(outputSelectivityPkl, 'wb') as f:
            pickle.dump(selectivity_eval_results, f)

        testImgCount += 1
        print("testImgCount: ", testImgCount)

    bestAttributionKeyStr = acquire_bestacc_attr(opt, outputSelectivityPkl)
    bestAttrName = bestAttributionKeyStr.split('_')[0]

    testImgCount = 0
    for i, (orig_img_tensors, segAndLabels) in enumerate(evaluation_loader):
        results_dict = {}
        aveAttr = []
        aveAttr_charContrib = []
        segmData, labels = segAndLabels[0]
        target = converter.encode([labels])

        # labels: RONALDO
        segmDataNP = segmData["segdata"]
        segmTensor = torch.from_numpy(segmDataNP).unsqueeze(0).unsqueeze(0)
        # print("segmTensor min: ", segmTensor.min()) # 0 starting segmentation
        segmTensor = segmTensor.to(device)
        # print("segmTensor shape: ", segmTensor.shape)
        # img1 = np.asarray(imgPIL.convert('L'))
        # sys.exit()
        # img1 = img1 / 255.0
        # img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        img1 = orig_img_tensors.to(device)
        img1.requires_grad = True
        bgImg = torch.zeros(img1.shape).to(device)

        ### Single char averaging
        if settings.MODEL == 'vitstr':
            charOffset = 1
        elif settings.MODEL == 'parseq':
            target = target[:, 1:] # First position [GO] not used in parseq too.
             # 0 index is [GO] char, not used in parseq, only the [EOS] which is in 1 index
            target[target > 0] -= 1
            charOffset = 0
            img1 = transforms.Normalize(0.5, 0.5)(img1) # Between -1 to 1

        # preds = model(img1, seqlen=converter.batch_max_length)
        input = img1
        origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
        origImgNP = gray2rgb(origImgNP)

        ### Captum test
        collectedAttributions = []
        for charIdx in range(0, len(labels)):
            scoring_singlechar.setSingleCharOutput(charIdx + charOffset)
            gtClassNum = target[0][charIdx + charOffset]

            # Best
            attributions = acquireAttribution(opt, super_pixel_model_singlechar, \
            input, segmTensor, gtClassNum, bestAttributionKeyStr, device)
            collectedAttributions.append(attributions)
        aveAttributions = torch.mean(torch.cat(collectedAttributions,dim=0), dim=0).unsqueeze(0)
        rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_{}_l.png'.format(i, bestAttrName))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_{bestAttrName}_l.pkl', aveAttributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, converter, labels, scoring_singlechar)
            results_dict[f"{bestAttrName}_local_acc"] = n_correct
            results_dict[f"{bestAttrName}_local_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model_singlechar, perturb_fn, img1, aveAttributions).detach().cpu().numpy())
            results_dict[f"{bestAttrName}_local_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
            results_dict[f"{bestAttrName}_local_sens"] = sens

        ### Best single
        attributions = acquireAttribution(opt, super_pixel_model, \
        input, segmTensor, 0, bestAttributionKeyStr, device)
        collectedAttributions.append(attributions)

        ### Global + Local context
        aveAttributions = torch.mean(torch.cat(collectedAttributions,dim=0), dim=0).unsqueeze(0)
        rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
        rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
        rankedAttr = gray2rgb(rankedAttr)
        mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
        mplotfig.savefig(outputDir + '{}_{}_gl.png'.format(i, bestAttrName))
        mplotfig.clear()
        plt.close(mplotfig)
        saveAttrData(attrOutputDir + f'{i}_{bestAttrName}_gl.pkl', aveAttributions, segmDataNP, origImgNP)
        if acquireSelectivity:
            n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, converter, labels, scoring_singlechar)
            results_dict[f"{bestAttrName}_global_local_acc"] = n_correct
            results_dict[f"{bestAttrName}_global_local_conf"] = confidenceList
        if acquireInfidelity:
            infid = float(infidelity(super_pixel_model_singlechar, perturb_fn, img1, aveAttributions).detach().cpu().numpy())
            results_dict[f"{bestAttrName}_global_local_infid"] = infid
        if acquireSensitivity:
            sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
            results_dict[f"{bestAttrName}_global_local_sens"] = sens

        selectivity_eval_results.append(results_dict)

        with open(outputSelectivityPkl, 'wb') as f:
            pickle.dump(selectivity_eval_results, f)

        testImgCount += 1
        print("testImgCount GlobLoc: ", testImgCount)

if __name__ == '__main__':
    # deleteInf()
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # combineBestDataXAI(opt)
    # acquire_average_auc(opt)
    # acquireListOfAveAUC(opt)
    acquireSingleCharAttrAve(opt)
