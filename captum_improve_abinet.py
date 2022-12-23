import os
import time
import string
import argparse
import re
import sys
import random
import pickle
import logging
from fastai.distributed import *
from fastai.vision import *

import settings
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage.color import gray2rgb
from nltk.metrics.distance import edit_distance
import cv2
import pickle
import copy

# from dataset import hierarchical_dataset, AlignCollate
# from model import Model, SuperPixler, CastNumpy, STRScore
# import hiddenlayer as hl
from callbacks import DumpPrediction, IterationCallback, TextAccuracy, TopKTextAccuracy
from dataset_abinet import ImageDataset, CustomImageDataset, TextDataset
from losses import MultiLosses
import matplotlib.pyplot as plt
import random
from utils_abinet import Config, Logger, MyDataParallel, MyConcatDataset, CharsetMapper
from utils import SRNConverter
from model_abinet import STRScore
from lime.wrappers.scikit_image import SegmentationAlgorithm
from captum._utils.models.linear_model import SkLearnLinearModel, SkLearnRidge
from captum_test import acquire_average_auc, saveAttrData, acquire_bestacc_attr, acquireAttribution

device = torch.device('cpu')

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

from captum.attr._utils.visualization import visualize_image_attr

### Acquire pixelwise attributions and replace them with ranked numbers averaged
### across segmentation with the largest contribution having the largest number
### and the smallest set to 1, which is the minimum number.
### attr - original attribution
### segm - image segmentations
def rankedAttributionsBySegm(attr, segm):
    aveSegmentations, sortedDict = averageSegmentsOut(attr[0,0], segm)
    totalSegm = len(sortedDict.keys()) # total segmentations
    sortedKeys = [k for k, v in sorted(sortedDict.items(), key=lambda item: item[1])]
    sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
    currentRank = totalSegm
    rankedSegmImg = torch.clone(attr)
    for totalSegToHide in range(0, len(sortedKeys)):
        currentSegmentToHide = sortedKeys[totalSegToHide]
        rankedSegmImg[0,0][segm == currentSegmentToHide] = currentRank
        currentRank -= 1
    return rankedSegmImg

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

def acquireSelectivityHit(origImg, attributions, segmentations, model, charset, labels, scoring):
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
    gt = labels
    for totalSegToHide in range(0, len(sortedKeys)):
        ### Acquire LIME prediction result
        currentSegmentToHide = sortedKeys[totalSegToHide]
        clonedImg[0,0][segmentations == currentSegmentToHide] = 0.0
        modelOut = model(clonedImg) ### Returns a tuple of dictionaries
        confScore = scoring(modelOut).cpu().detach().numpy()
        pred, _, __ = postprocess(modelOut[0], charset, config.model_eval)
        pred = pred[0] # outputs a list, so query [0]
        if pred.lower() == gt.lower(): ### not lowercase gt labels, pred only predicts lowercase
            n_correct.append(1)
        else:
            n_correct.append(0)
        confScore = confScore[0][0]*100
        confidenceList.append(confScore)
    return n_correct, confidenceList

def _set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')

def get_model(config):
    import importlib
    names = config.model_name.split('.')
    module_name, class_name = '.'.join(names[:-1]), names[-1]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls(config)
    logging.info(model)
    model = model.eval()
    return model

def load(model, file, device=None, strict=True):
    if device is None: device = 'cpu'
    elif isinstance(device, int): device = torch.device('cuda', device)
    assert os.path.isfile(file)
    state = torch.load(file, map_location=device)
    if set(state.keys()) == {'model', 'opt'}:
        state = state['model']
    model.load_state_dict(state, strict=strict)
    return model

def _get_dataset(ds_type, paths, is_training, config, **kwargs):
    kwargs.update({
        'img_h': config.dataset_image_height,
        'img_w': config.dataset_image_width,
        'max_length': config.dataset_max_length,
        'case_sensitive': config.dataset_case_sensitive,
        'charset_path': config.dataset_charset_path,
        'data_aug': config.dataset_data_aug,
        'deteriorate_ratio': config.dataset_deteriorate_ratio,
        'is_training': is_training,
        'multiscales': config.dataset_multiscales,
        'one_hot_y': config.dataset_one_hot_y,
    })
    datasets = [ds_type(p, **kwargs) for p in paths]
    if len(datasets) > 1: return MyConcatDataset(datasets)
    else: return datasets[0]

def _get_databaunch(config):
    # An awkward way to reduce loadding data time during test
    if config.global_phase == 'test': config.dataset_train_roots = config.dataset_test_roots
    train_ds = _get_dataset(ImageDataset, config.dataset_train_roots, True, config)
    valid_ds = _get_dataset(ImageDataset, config.dataset_test_roots, False, config)
    data = ImageDataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        bs=config.dataset_train_batch_size,
        val_bs=config.dataset_test_batch_size,
        num_workers=config.dataset_num_workers,
        pin_memory=config.dataset_pin_memory).normalize(imagenet_stats)
    ar_tfm = lambda x: ((x[0], x[1]), x[1])  # auto-regression only for dtd
    data.add_tfm(ar_tfm)

    logging.info(f'{len(data.train_ds)} training items found.')
    if not data.empty_val:
        logging.info(f'{len(data.valid_ds)} valid items found.')

    return data

def postprocess(output, charset, model_eval):
    def _get_output(last_output, model_eval):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                if res['name'] == model_eval: return res
        return last_output

    def _decode(logit):
        """ Greed decode """
        out = F.softmax(logit, dim=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = charset.get_text(o.argmax(dim=1), padding=False, trim=False)
            text = text.split(charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(dim=1)[0])
            pt_lengths.append(min(len(text) + 1, charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    output = _get_output(output, model_eval)
    # print("output type: ", type(output))
    logits, pt_lengths = output['logits'], output['pt_lengths']
    pt_text, pt_scores, pt_lengths_ = _decode(logits)

    return pt_text, pt_scores, pt_lengths_

def main(config):
    height = config.imgH
    width = config.imgW
    # custom_segm_dataroot = "/media/markytools/OrigDocs/markytools/Documents/MSEE"\
    # "Thesis/STR/datasets/data_lmdb_segmentations/{}X{}/{}/".format(height, width, datasetName)
    # 'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    targetDataset = "IIIT5k_3000" # Change also the configs/train_abinet.yaml test.roots test folder
    segmRootDir = "/home/goo/str/datasets/segmentations/{}X{}/{}/".format(height, width, targetDataset)
    outputSelectivityPkl = "shapley_singlechar_ave_{}_{}.pkl".format(settings.MODEL, targetDataset)
    outputDir = "./attributionImgs/{}/{}/".format(settings.MODEL, targetDataset)
    attrOutputDir = "/data/goo/strattr/attributionData/{}/{}/".format(settings.MODEL, targetDataset)
    resumePkl = "" # Use to resume when session destroyed. Set to "" to disable
    acquireSelectivity = True
    acquireInfidelity = False
    acquireSensitivity = False
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(attrOutputDir):
        os.makedirs(attrOutputDir)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    config.character = "abcdefghijklmnopqrstuvwxyz1234567890$#" # See charset_36.txt
    converter = SRNConverter(config.character, 36)

    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)

    """ evaluation """
    modelCopy = copy.deepcopy(model)
    scoring_singlechar = STRScore(config=config, charsetMapper=charset, postprocessFunc=postprocess, device=device, enableSingleCharAttrAve=True)
    super_pixel_model_singlechar = torch.nn.Sequential(
        modelCopy,
        scoring_singlechar
    ).to(device)
    modelCopy.eval()
    scoring_singlechar.eval()
    super_pixel_model_singlechar.eval()

    scoring = STRScore(config=config, charsetMapper=charset, postprocessFunc=postprocess, device=device)
    ### SuperModel
    super_pixel_model = torch.nn.Sequential(
    model,
    scoring
    ).to(device)
    model.eval()
    scoring.eval()
    super_pixel_model.eval()

    selectivity_eval_results = []

    if config.blackbg:
        shapImgLs = np.zeros(shape=(1, 3, 32, 128)).astype(np.float32)
        trainList = np.array(shapImgLs)
        background = torch.from_numpy(trainList).to(device)

    # define a perturbation function for the input (used for calculating infidelity)
    def perturb_fn(modelInputs):
        noise = torch.tensor(np.random.normal(0, 0.003, modelInputs.shape)).float()
        noise = noise.to(device)
        return noise, modelInputs - noise

    strict = ifnone(config.model_strict, True)
    ### Dataset not shuffled because it is not a dataloader, just a dataset
    valid_ds = _get_dataset(CustomImageDataset, config.dataset_test_roots, False, config)
    # print("valid_ds: ", len(valid_ds[0]))
    testImgCount = 0
    if resumePkl != "":
        with open(resumePkl, 'rb') as filePkl:
            selectivity_eval_results = pickle.load(filePkl)
        testImgCount = selectivity_eval_results[-1]["testImgCount"] # ResumeCount
    try:
        for i, (orig_img_tensors, labels, labels_tensor) in enumerate(valid_ds):
            if i <= testImgCount:
                continue
            orig_img_tensors = orig_img_tensors.unsqueeze(0)
            # print("orig_img_tensors: ", orig_img_tensors.shape) # (3, 32, 128)
            # img_rgb *= 255.0
            # img_rgb = img_rgb.astype('int')
            # print("img_rgb max: ", img_rgb.max()) ### 255
            # img_rgb = np.asarray(orig_img_tensors)
            # segmentations = segmentation_fn(img_rgb)
            # print("segmentations shape: ", segmentations.shape) # (224, 224)
            # print("segmentations min: ", segmentations.min()) 0
            # print("Unique: ", len(np.unique(segmentations))) # (70)
            results_dict = {}
            with open(segmRootDir + "{}.pkl".format(i), 'rb') as f:
                pklData = pickle.load(f)
            # segmData, labels = segAndLabels[0]
            segmDataNP = pklData["segdata"]
            labels = labels.lower() # For fair evaluation for all
            assert pklData['label'] == labels
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
            # preds = model(img1, seqlen=converter.batch_max_length)
            input = img1
            origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
            origImgNP = gray2rgb(origImgNP)

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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["intgrad_acc"] = n_correct
                results_dict["intgrad_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["intgrad_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(ig.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["intgrad_sens"] = sens

            ### Gradient SHAP using zero-background
            gs = GradientShap(super_pixel_model)
            # We define a distribution of baselines and draw `n_samples` from that
            # distribution in order to estimate the expectations of gradients across all baselines
            baseline_dist = torch.zeros((1, 3, height, width))
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["gradshap_acc"] = n_correct
                results_dict["gradshap_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["deeplift_acc"] = n_correct
                results_dict["deeplift_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["saliency_acc"] = n_correct
                results_dict["saliency_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["inpxgrad_acc"] = n_correct
                results_dict["inpxgrad_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["guidedbp_acc"] = n_correct
                results_dict["guidedbp_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["deconv_acc"] = n_correct
                results_dict["deconv_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["featablt_acc"] = n_correct
                results_dict["featablt_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["shapley_acc"] = n_correct
                results_dict["shapley_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["lime_acc"] = n_correct
                results_dict["lime_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, charset, labels, scoring)
                results_dict["kernelshap_acc"] = n_correct
                results_dict["kernelshap_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["kernelshap_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(ks.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["kernelshap_sens"] = sens

            # Other data
            results_dict["testImgCount"] = testImgCount # 0 to N-1
            selectivity_eval_results.append(results_dict)

            with open(outputSelectivityPkl, 'wb') as f:
                pickle.dump(selectivity_eval_results, f)

            testImgCount += 1
            print("testImgCount: ", testImgCount)
    except:
        print("An exception occurred1")

    del valid_ds
    valid_ds = _get_dataset(CustomImageDataset, config.dataset_test_roots, False, config)
    bestAttributionKeyStr = acquire_bestacc_attr(config, outputSelectivityPkl)
    bestAttrName = bestAttributionKeyStr.split('_')[0]

    testImgCount = 0
    try:
        for i, (orig_img_tensors, labels, labels_tensor) in enumerate(valid_ds):
            orig_img_tensors = orig_img_tensors.unsqueeze(0)
            # print("orig_img_tensors: ", orig_img_tensors.shape) # (3, 32, 128)
            # img_rgb *= 255.0
            # img_rgb = img_rgb.astype('int')
            # print("img_rgb max: ", img_rgb.max()) ### 255
            # img_rgb = np.asarray(orig_img_tensors)
            # segmentations = segmentation_fn(img_rgb)
            # print("segmentations shape: ", segmentations.shape) # (224, 224)
            # print("segmentations min: ", segmentations.min()) 0
            # print("Unique: ", len(np.unique(segmentations))) # (70)
            results_dict = {}
            with open(segmRootDir + "{}.pkl".format(i), 'rb') as f:
                pklData = pickle.load(f)
            # segmData, labels = segAndLabels[0]
            segmDataNP = pklData["segdata"]
            labels = labels.lower() # For fair evaluation for all
            assert pklData['label'] == labels
            # labels = "lama0"
            target = converter.encode([labels], len(config.character))
            target = target[0] + 1 # Idx predicted by ABINET is 1 to N chars, not 0 to N-1
            target[target > 36] = 0 # Remove EOS predictions, set endpoint chars to 0
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
            # preds = model(img1, seqlen=converter.batch_max_length)
            input = img1
            origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
            origImgNP = gray2rgb(origImgNP)

            charOffset = 0
            ### Local explanations only
            collectedAttributions = []
            for charIdx in range(0, len(labels)):
                scoring_singlechar.setSingleCharOutput(charIdx + charOffset)
                # print("charIdx + charOffset: ", charIdx + charOffset)
                # print("target[0]: ", target[0])
                gtClassNum = target[0][charIdx + charOffset]

                ### Best local
                attributions = acquireAttribution(config, super_pixel_model_singlechar, \
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
                n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, charset, labels, scoring_singlechar)
                results_dict[f"{bestAttrName}_local_acc"] = n_correct
                results_dict[f"{bestAttrName}_local_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model_singlechar, perturb_fn, img1, aveAttributions, normalize=True).detach().cpu().numpy())
                results_dict[f"{bestAttrName}_local_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
                results_dict[f"{bestAttrName}_local_sens"] = sens

            ### Best global
            attributions = acquireAttribution(config, super_pixel_model, \
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
                n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, charset, labels, scoring_singlechar)
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
    except:
        print("An exception occurred2")

### Use to check if the model predicted the image or not. Output a pickle file with the image index.
def modelDatasetPredOnly(opt):
    # 'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    datasetName = "IIIT5k_3000"
    outputSelectivityPkl = "metrics_predictonly_eval_results_{}.pkl".format(datasetName)
    charset = CharsetMapper(filename=config.dataset_charset_path,
                            max_length=config.dataset_max_length + 1)
    model = get_model(config).to(device)
    model = load(model, config.model_checkpoint, device=device)
    model.eval()
    strict = ifnone(config.model_strict, True)
    ### Dataset not shuffled because it is not a dataloader, just a dataset
    valid_ds = _get_dataset(CustomImageDataset, config.dataset_test_roots, False, config)
    # print("valid_ds: ", len(valid_ds[0]))
    testImgCount = 0
    predOutput = []
    for i, (orig_img_tensors, labels, labels_tensor) in enumerate(valid_ds):
        orig_img_tensors = orig_img_tensors.unsqueeze(0).to(device)
        modelOut = model(orig_img_tensors) ### Returns a tuple of dictionaries
        pred, _, __ = postprocess(modelOut[0], charset, config.model_eval)
        pred = pred[0] # outputs a list, so query [0]
        if pred.lower() == labels.lower(): predOutput.append(1)
        else: predOutput.append(0)
        with open(outputSelectivityPkl, 'wb') as f:
            pickle.dump(predOutput, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--phase', type=str, default=None, choices=['train', 'test'])
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_root', type=str, default=None)
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--scorer', type=str, default='mean', help='See STRScore: cumprod | mean')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--image_only', action='store_true', default=None)
    parser.add_argument('--blackbg', action='store_true', default=None)
    parser.add_argument('--model_strict', action='store_false', default=None)
    parser.add_argument('--model_eval', type=str, default=None,
                        choices=['alignment', 'vision', 'language'])
    args = parser.parse_args()
    config = Config(args.config)
    if args.name is not None: config.global_name = args.name
    if args.phase is not None: config.global_phase = args.phase
    if args.test_root is not None: config.dataset_test_roots = [args.test_root]
    if args.scorer is not None: config.scorer = args.scorer
    if args.blackbg is not None: config.blackbg = args.blackbg
    if args.rgb is not None: config.rgb = args.rgb
    if args.imgH is not None: config.imgH = args.imgH
    if args.imgW is not None: config.imgW = args.imgW
    if args.checkpoint is not None: config.model_checkpoint = args.checkpoint
    if args.debug is not None: config.global_debug = args.debug
    if args.image_only is not None: config.global_image_only = args.image_only
    if args.model_eval is not None: config.model_eval = args.model_eval
    if args.model_strict is not None: config.model_strict = args.model_strict

    Logger.init(config.global_workdir, config.global_name, config.global_phase)
    Logger.enable_file()
    _set_random_seed(config.global_seed)
    logging.info(config)

    # acquire_average_auc(config)
    main(config)
    # modelDatasetPredOnly(config)
