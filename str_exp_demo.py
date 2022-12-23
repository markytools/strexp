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
from captum_test import acquire_average_auc, saveAttrData
from netdissect import nethook
import copy
from skimage.color import gray2rgb
from matplotlib import pyplot as plt
from torchvision import transforms

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

### Create segmentation formats for the synthetic str dataset
def outputSegmOnly_synth(opt):
    ### targetDataset - one dataset only, SVTP-645, CUTE80-288images
    targetDataset = "synthstr_char10" # ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    targetHeight = 224 # for trba (32,100), vitstr (224, 224)
    targetWidth = 224
    segmRootDir = "/media/markytools/OrigDocs/markytools/Documents/MSEEThesis/STR/datasets/segmentations/{}X{}/{}/".format(targetHeight, targetWidth, targetDataset)

    if not os.path.exists(segmRootDir):
        os.makedirs(segmRootDir)

    opt.eval = True
    ### Only IIIT5k_3000
    # eval_data_list = [targetDataset]
    target_output_orig = opt.outputOrigDir

    ### Taken from LIME
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=random.randint(0, 1000))
    # for eval_data in eval_data_list:
    eval_data_path = opt.eval_data
    AlignCollate_evaluation = AlignCollate(imgH=targetHeight, imgW=targetWidth, keep_ratio_with_pad=opt.PAD)
    eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, targetDir=target_output_orig)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=1,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image_tensors = ((image_tensors + 1.0) / 2.0) * 255.0
        imgDataDict = {}
        img_numpy = image_tensors.cpu().detach().numpy()[0] ### Need to set batch size to 1 only
        if img_numpy.shape[0] == 1:
            img_numpy = gray2rgb(img_numpy[0])
        # print("img_numpy shape: ", img_numpy.shape) # (32,100,3)
        segmOutput = segmentation_fn(img_numpy)
        # print("segmOutput unique: ", len(np.unique(segmOutput)))
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
        # pkl_filename = "/home/markytools/Documents/MSEEThesis/STR/deep-text-recognition-benchmark-deepshap/selectivity_eval_results.pkl" # TRBA
        pkl_filename = "/home/goo/str/str_vit_dataexplain_lambda/metrics_sensitivity_eval_results_CUTE80.pkl" # VITSTR
        # pkl_filename = "/home/markytools/Documents/MSEEThesis/STR/ABINet/selectivity_eval_results.pkl" # ABINET
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

# Single directory STRExp explanations output demo
def sampleDemo(opt):
    targetDataset = "SVTP"
    demoImgDir = "demo_image/"
    outputDir = "/data/goo/demo_image_output/"

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=random.randint(0, 1000))

    """ model configuration """
    if opt.Transformer:
        converter = TokenLabelConverter(opt)
    elif 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model_obj = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model_obj).to(device)

    modelCopy = copy.deepcopy(model)

    """ evaluation """
    scoring_singlechar = STRScore(opt=opt, converter=converter, device=device, enableSingleCharAttrAve=True)
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
    scoring = STRScore(opt=opt, converter=converter, device=device)
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
    for path, subdirs, files in os.walk(demoImgDir):
        for name in files:
            nameNoExt = name.split('.')[0]
            labels = nameNoExt
            fullfilename = os.path.join(demoImgDir, name) # Value
            # fullfilename: /data/goo/strattr/attributionData/trba/CUTE80/66_featablt.pkl
            pilImg = Image.open(fullfilename)

            if settings.MODEL=="vitstr":
                pilImg = pilImg.resize((224, 224))

            orig_img_tensors = transforms.ToTensor()(pilImg)
            orig_img_tensors = torch.mean(orig_img_tensors, dim=0).unsqueeze(0).unsqueeze(0)
            image_tensors = ((torch.clone(orig_img_tensors) + 1.0) / 2.0) * 255.0
            imgDataDict = {}
            img_numpy = image_tensors.cpu().detach().numpy()[0] ### Need to set batch size to 1 only
            if img_numpy.shape[0] == 1:
                img_numpy = gray2rgb(img_numpy[0])
            # print("img_numpy shape: ", img_numpy.shape) # (32,100,3)
            segmOutput = segmentation_fn(img_numpy)
            # print("orig_img_tensors shape: ", orig_img_tensors.shape) # (3, 224, 224)
            # print("orig_img_tensors max: ", orig_img_tensors.max()) # 0.6824 (1)
            # print("orig_img_tensors min: ", orig_img_tensors.min()) # 0.0235 (0)
            # sys.exit()

            results_dict = {}
            aveAttr = []
            aveAttr_charContrib = []
            # segmData, labels = segAndLabels[0]
            target = converter.encode([labels])

            # labels: RONALDO
            segmDataNP = segmOutput
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
            charOffset = 1

            # preds = model(img1, seqlen=converter.batch_max_length)
            input = img1
            origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
            origImgNP = gray2rgb(origImgNP)

            ### Local explanations only
            collectedAttributions = []
            for charIdx in range(0, len(labels)):
                scoring_singlechar.setSingleCharOutput(charIdx + charOffset)
                gtClassNum = target[0][charIdx + charOffset]

                ### Shapley Value Sampling
                svs = ShapleyValueSampling(super_pixel_model_singlechar)
                # attr = svs.attribute(input, target=0, n_samples=200) ### Individual pixels, too long to calculate
                attributions = svs.attribute(input, target=gtClassNum, feature_mask=segmTensor)
                collectedAttributions.append(attributions)
            aveAttributions = torch.mean(torch.cat(collectedAttributions,dim=0), dim=0).unsqueeze(0)
            if not torch.isnan(aveAttributions).any():
                rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_shapley_l.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Shapley Value Sampling
            svs = ShapleyValueSampling(super_pixel_model)
            # attr = svs.attribute(input, target=0, n_samples=200) ### Individual pixels, too long to calculate
            attributions = svs.attribute(input, target=0, feature_mask=segmTensor)
            if not torch.isnan(attributions).any():
                collectedAttributions.append(attributions)
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_shapley.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Global + Local context
            aveAttributions = torch.mean(torch.cat(collectedAttributions,dim=0), dim=0).unsqueeze(0)
            if not torch.isnan(aveAttributions).any():
                rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_shapley_gl.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### BASELINE Evaluations

            ### Integrated Gradients
            ig = IntegratedGradients(super_pixel_model)
            attributions = ig.attribute(input, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_intgrad.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Gradient SHAP using zero-background
            gs = GradientShap(super_pixel_model)
            # We define a distribution of baselines and draw `n_samples` from that
            # distribution in order to estimate the expectations of gradients across all baselines
            baseline_dist = torch.zeros((1, 1, 224, 224))
            baseline_dist = baseline_dist.to(device)
            attributions = gs.attribute(input, baselines=baseline_dist, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_gradshap.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### DeepLift using zero-background
            dl = DeepLift(super_pixel_model)
            attributions = dl.attribute(input, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_deeplift.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Saliency
            saliency = Saliency(super_pixel_model)
            attributions = saliency.attribute(input, target=0) ### target=class0
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_saliency.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### InputXGradient
            input_x_gradient = InputXGradient(super_pixel_model)
            attributions = input_x_gradient.attribute(input, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_inpxgrad.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### GuidedBackprop
            gbp = GuidedBackprop(super_pixel_model)
            attributions = gbp.attribute(input, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_guidedbp.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Deconvolution
            deconv = Deconvolution(super_pixel_model)
            attributions = deconv.attribute(input, target=0)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_deconv.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### Feature ablator
            ablator = FeatureAblation(super_pixel_model)
            attributions = ablator.attribute(input, target=0, feature_mask=segmTensor)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_featablt.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ## LIME
            interpretable_model = SkLearnRidge(alpha=1, fit_intercept=True) ### This is the default used by LIME
            lime = Lime(super_pixel_model, interpretable_model=interpretable_model)
            attributions = lime.attribute(input, target=0, feature_mask=segmTensor)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_lime.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

            ### KernelSHAP
            ks = KernelShap(super_pixel_model)
            attributions = ks.attribute(input, target=0, feature_mask=segmTensor)
            if not torch.isnan(attributions).any():
                rankedAttr = rankedAttributionsBySegm(attributions, segmDataNP)
                rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
                rankedAttr = gray2rgb(rankedAttr)
                mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map', cmap='RdYlGn')
                mplotfig.savefig(outputDir + '{}_kernelshap.png'.format(nameNoExt))
                mplotfig.clear()
                plt.close(mplotfig)

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
    # acquireSingleCharAttrAve(opt)
    sampleDemo(opt)
