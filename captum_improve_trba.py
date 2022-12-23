import os
import time
import string
import argparse
import re
import sys
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from skimage.color import gray2rgb
from nltk.metrics.distance import edit_distance
import cv2

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset_trba import hierarchical_dataset, AlignCollate
from model_trba import Model, SuperPixler, CastNumpy, STRScore
# import hiddenlayer as hl
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt
import random
from captum._utils.models.linear_model import SkLearnLinearModel, SkLearnRidge
import statistics
import settings
import sys
import copy
from captum_test import acquire_average_auc, saveAttrData
from captum_improve import rankedAttributionsBySegm
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

def getPredAndConf(opt, model, scoring, image, converter, labels):
    batch_size = image.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
    if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred)

        confScore = scoring(preds)
        confScore = confScore.detach().cpu().numpy()

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        # Select max probabilty (greedy decoding) then decode index to character
        if opt.baiduCTC:
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
        else:
            _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    else:
        preds = model(image, text_for_pred, is_train=False)

        confScore = scoring(preds)
        confScore = confScore.detach().cpu().numpy()

        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        target = text_for_loss[:, 1:]  # without [GO] Symbol
        # cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        ### Remove all chars after '[s]'
        preds_str = preds_str[0]
        preds_str = preds_str[:preds_str.find('[s]')]
        # pred = pred[:pred_EOS]
    return preds_str, confScore

### Output and save segmentations only for one dataset only
def outputSegmOnly(opt):
    ### targetDataset - one dataset only, SVTP-645, CUTE80-288images
    targetDataset = "CUTE80" # ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    targetHeight = 32
    targetWidth = 100
    segmRootDir = "/home/uclpc1/Documents/STR/datasets/segmen"\
    "tations/{}X{}/{}/".format(targetHeight, targetWidth, targetDataset)

    if not os.path.exists(segmRootDir):
        os.makedirs(segmRootDir)

    opt.eval = True
    ### Only IIIT5k_3000
    eval_data_list = [targetDataset]
    target_output_orig = opt.outputOrigDir

    ### Taken from LIME
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=random.randint(0, 1000))
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
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
    gt = str(labels[0])
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

def main(opt):
    # 'IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857', 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80'
    datasetName = "SVTP"
    custom_segm_dataroot = "/home/goo/str/datasets/segmentations/{}X{}/{}/".format(opt.imgH, opt.imgW, datasetName)
    outputSelectivityPkl = "shapley_singlechar_ave_{}_{}.pkl".format(settings.MODEL, datasetName)
    outputDir = "./attributionImgs/{}/{}/".format(settings.MODEL, datasetName)
    attrOutputDir = "/data/goo/strattr/attributionData/{}/{}/".format(settings.MODEL, datasetName)
    acquireSelectivity = True
    acquireInfidelity = False
    acquireSensitivity = False ### GPU error
    imgHeight = 32
    imgWidth = 100
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    if not os.path.exists(attrOutputDir):
        os.makedirs(attrOutputDir)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model_obj = Model(opt, device)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model_obj).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])

    modelCopy = copy.deepcopy(model)
    scoring_singlechar = STRScore(opt=opt, converter=converter, device=device, enableSingleCharAttrAve=True)
    super_pixel_model_singlechar = torch.nn.Sequential(
        # super_pixler,
        # numpy2torch_converter,
        modelCopy,
        scoring_singlechar
    ).to(device)
    modelCopy.train()
    scoring_singlechar.train()
    super_pixel_model_singlechar.train()

    scoring = STRScore(opt=opt, converter=converter, device=device)
    super_pixel_model = torch.nn.Sequential(
        model,
        scoring
    )
    model.train()
    scoring.train()
    super_pixel_model.train()

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """Output shap values"""
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    # eval_data_list = ['IIIT5k_3000', 'IC03_860', 'IC03_867', 'IC15_1811']
    target_output_orig = opt.outputOrigDir
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    # eval_data_list = ['IIIT5k_3000']
    eval_data_list = [datasetName]
    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')

    selectivity_eval_results = []
    imageData = []
    targetText = "all"
    middleMaskThreshold = 5
    testImgCount = 0
    imgResultDir = str(opt.Transformation) + "-" + str(opt.FeatureExtraction) + "-" + str(opt.SequenceModeling) + "-" + str(opt.Prediction) + "-" + str(opt.scorer)

    # define a perturbation function for the input (used for calculating infidelity)
    def perturb_fn(modelInputs):
        noise = torch.tensor(np.random.normal(0, 0.003, modelInputs.shape)).float()
        noise = noise.to(device)
        return noise, modelInputs - noise

    if opt.blackbg:
        shapImgLs = np.zeros(shape=(1, 1, 32, 100)).astype(np.float32)
        trainList = np.array(shapImgLs)
        background = torch.from_numpy(trainList).to(device)
    if imgResultDir != "":
        if not os.path.exists(imgResultDir):
            os.makedirs(imgResultDir)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, targetDir=target_output_orig)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        # image_tensors, labels = next(iter(evaluation_loader)) ### Iterate one batch only
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            # img_rgb *= 255.0
            # img_rgb = img_rgb.astype('int')
            # print("img_rgb max: ", img_rgb.max()) ### 255
            # img_rgb = np.asarray(orig_img_tensors)
            # segmentations = segmentation_fn(img_rgb)
            # print("segmentations shape: ", segmentations.shape) # (224, 224)
            # print("segmentations min: ", segmentations.min()) 0
            # print("Unique: ", len(np.unique(segmentations))) # (70)
            # print("target: ", target) tensor([[ 0, 29, 26, 25, 12
            results_dict = {}
            pklFilename = custom_segm_dataroot + "{}.pkl".format(i)
            with open(pklFilename, 'rb') as f:
                pklData = pickle.load(f)
            segmDataNP = pklData["segdata"]
            # print("segmDataNP unique: ", len(np.unique(segmDataNP)))
            assert pklData["label"] == labels[0]
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
            target = converter.encode(labels)
            target = target[0][:, 1:]
            charOffset = 0
            input = img1
            origImgNP = torch.clone(orig_img_tensors).detach().cpu().numpy()[0][0] # (1, 1, 224, 224)
            origImgNP = gray2rgb(origImgNP)

            # preds = model(input)
            # preds_prob = F.softmax(preds, dim=2)
            # preds_max_prob, preds_max_idx = preds_prob.max(dim=2)
            # print("preds_max_idx: ", preds_max_idx) tensor([[14, 26, 25, 12

            ### Captum test
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
            rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
            rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
            rankedAttr = gray2rgb(rankedAttr)
            mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
            mplotfig.savefig(outputDir + '{}_shapley_l.png'.format(i))
            mplotfig.clear()
            plt.close(mplotfig)
            saveAttrData(attrOutputDir + f'{i}_shapley_l.pkl', aveAttributions, segmDataNP, origImgNP)
            if acquireSelectivity:
                n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, converter, labels, scoring_singlechar)
                results_dict["shapley_local_acc"] = n_correct
                results_dict["shapley_local_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model_singlechar, perturb_fn, img1, aveAttributions).detach().cpu().numpy())
                results_dict["shapley_local_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["shapley_local_sens"] = sens

            ### Shapley Value Sampling
            svs = ShapleyValueSampling(super_pixel_model)
            # attr = svs.attribute(input, target=0, n_samples=200) ### Individual pixels, too long to calculate
            attributions = svs.attribute(input, target=0, feature_mask=segmTensor)
            collectedAttributions.append(attributions)
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
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["shapley_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["shapley_sens"] = sens

            ### Global + Local context
            aveAttributions = torch.mean(torch.cat(collectedAttributions,dim=0), dim=0).unsqueeze(0)
            rankedAttr = rankedAttributionsBySegm(aveAttributions, segmDataNP)
            rankedAttr = rankedAttr.detach().cpu().numpy()[0][0]
            rankedAttr = gray2rgb(rankedAttr)
            mplotfig, _ = visualize_image_attr(rankedAttr, origImgNP, method='blended_heat_map')
            mplotfig.savefig(outputDir + '{}_shapley_gl.png'.format(i))
            mplotfig.clear()
            plt.close(mplotfig)
            saveAttrData(attrOutputDir + f'{i}_shapley_gl.pkl', aveAttributions, segmDataNP, origImgNP)
            if acquireSelectivity:
                n_correct, confidenceList = acquireSelectivityHit(img1, aveAttributions, segmDataNP, modelCopy, converter, labels, scoring_singlechar)
                results_dict["shapley_global_local_acc"] = n_correct
                results_dict["shapley_global_local_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model_singlechar, perturb_fn, img1, aveAttributions).detach().cpu().numpy())
                results_dict["shapley_global_local_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(svs.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["shapley_global_local_sens"] = sens


            # Baselines
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
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["intgrad_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(ig.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["intgrad_sens"] = sens

            ### Gradient SHAP using zero-background
            gs = GradientShap(super_pixel_model)
            # We define a distribution of baselines and draw `n_samples` from that
            # distribution in order to estimate the expectations of gradients across all baselines
            baseline_dist = torch.zeros((1, 1, imgHeight, imgWidth))
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
                results_dict["inpxgrad_acc"] = n_correct
                results_dict["inpxgrad_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["inpxgrad_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(input_x_gradient.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["inpxgrad_sens"] = sens

            ## GuidedBackprop
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
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["guidedbp_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(gbp.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["guidedbp_sens"] = sens
            #
            # ## Deconvolution
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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
                results_dict["featablt_acc"] = n_correct
                results_dict["featablt_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["featablt_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(ablator.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["featablt_sens"] = sens

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
                n_correct, confidenceList = acquireSelectivityHit(img1, attributions, segmDataNP, model, converter, labels, scoring)
                results_dict["kernelshap_acc"] = n_correct
                results_dict["kernelshap_conf"] = confidenceList
            if acquireInfidelity:
                infid = float(infidelity(super_pixel_model, perturb_fn, img1, attributions, normalize=True).detach().cpu().numpy())
                results_dict["kernelshap_infid"] = infid
            if acquireSensitivity:
                sens = float(sensitivity_max(ks.attribute, img1, target=0).detach().cpu().numpy())
                results_dict["kernelshap_sens"] = sens

            selectivity_eval_results.append(results_dict)

            with open(outputSelectivityPkl, 'wb') as f:
                pickle.dump(selectivity_eval_results, f)

            testImgCount += 1
            print("testImgCount: ", testImgCount)

def outputOrigImagesOnly(opt):
    datasetName = "CUTE80" # ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    opt.outputOrigDir = "./datasetOrigImgs/{}/".format(datasetName)
    opt.output_orig = True
    opt.corruption_num = 0
    opt.apply_corruptions = False
    opt.min_imgnum = 0
    opt.max_imgnum = 1000

    target_output_orig = opt.outputOrigDir
    if not os.path.exists(target_output_orig):
        os.makedirs(target_output_orig)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model_obj = Model(opt, device)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model_obj).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    scoring = STRScore(opt=opt, converter=converter, device=device)
    ###

    super_pixel_model = torch.nn.Sequential(
        model,
        scoring
    )
    model.train()
    scoring.train()
    super_pixel_model.train()
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """Output shap values"""
    """ evaluation with 10 benchmark evaluation datasets """
    # The evaluation datasets, dataset order is same with Table 1 in our paper.
    # eval_data_list = ['IIIT5k_3000', 'IC03_860', 'IC03_867', 'IC15_1811']
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    # eval_data_list = ['IIIT5k_3000']
    eval_data_list = [datasetName]
    # # To easily compute the total accuracy of our paper.
    # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867',
    #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')

    selectivity_eval_results = []
    imageData = []
    targetText = "all"
    middleMaskThreshold = 5
    testImgCount = 0
    imgResultDir = str(opt.Transformation) + "-" + str(opt.FeatureExtraction) + "-" + str(opt.SequenceModeling) + "-" + str(opt.Prediction) + "-" + str(opt.scorer)

    if opt.blackbg:
        shapImgLs = np.zeros(shape=(1, 1, 32, 100)).astype(np.float32)
        trainList = np.array(shapImgLs)
        background = torch.from_numpy(trainList).to(device)
    if imgResultDir != "":
        if not os.path.exists(imgResultDir):
            os.makedirs(imgResultDir)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, targetDir=target_output_orig)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        # image_tensors, labels = next(iter(evaluation_loader)) ### Iterate one batch only
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            testImgCount += 1
            print("testImgCount: ", testImgCount)

### Use to check if the model predicted the image or not. Output a pickle file with the image index.
def modelDatasetPredOnly(opt):
    ### targetDataset - one dataset only, CUTE80 has 288 samples
    targetDataset = "CUTE80" # ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    outputSelectivityPkl = "metrics_predictonly_results_{}.pkl".format(targetDataset)
    start_time = time.time()

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model_obj = Model(opt, device)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model_obj).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    scoring = STRScore(opt=opt, converter=converter, device=device)
    ###

    super_pixel_model = torch.nn.Sequential(
        model,
        scoring
    )
    model.train()
    scoring.train()
    super_pixel_model.train()

    if opt.blackbg:
        shapImgLs = np.zeros(shape=(1, 1, 224, 224)).astype(np.float32)
        trainList = np.array(shapImgLs)
        background = torch.from_numpy(trainList).to(device)

    opt.eval = True
    eval_data_list = [targetDataset]

    testImgCount = 0
    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    target_output_orig = opt.outputOrigDir
    predOutput = []
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt, targetDir=target_output_orig)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        testImgCount = 0
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            image = orig_img_tensors.to(device)
            batch_size = 1
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                confScore = scoring(preds)
                confScore = confScore.detach().cpu().numpy()

                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                # Select max probabilty (greedy decoding) then decode index to character
                if opt.baiduCTC:
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                else:
                    _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)[0]
            else:
                preds = model(image, text_for_pred, is_train=False)

                confScore = scoring(preds)
                confScore = confScore.detach().cpu().numpy()

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                # cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

                ### Remove all chars after '[s]'
                preds_str = preds_str[0]
                preds_str = preds_str[:preds_str.find('[s]')]
            # print("preds_str: ", preds_str) # lowercased prediction
            # print("labels: ", labels[0]) # gt already in lowercased
            if preds_str==labels[0]: predOutput.append(1)
            else: predOutput.append(0)

            with open(outputSelectivityPkl, 'wb') as f:
                pickle.dump(predOutput, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--superHeight', type=int, default=5, help='the height of the superpixel')
    parser.add_argument('--superWidth', type=int, default=2, help='the width of the superpixel')
    parser.add_argument('--min_imgnum', type=int, default=0, help='set this to skip for loop index of specific image number')
    parser.add_argument('--max_imgnum', type=int, default=2, help='set this to skip for loop index of specific image number')
    parser.add_argument('--severity', type=int, default=1, help='severity level if apply corruptions')
    parser.add_argument('--scorer', type=str, default='cumprod', help='See STRScore: cumprod | mean')
    parser.add_argument('--corruption_num', type=int, default=0, help='corruption to apply')
    parser.add_argument('--confidence_mode', type=int, default=0, help='0-sum of argmax; 1-edit distance')
    parser.add_argument('--outputOrigDir', type=str, default="output_orig/", help='output directory to save original \
    images. This will be automatically created. Needs --output_orig too.')
    parser.add_argument('--output_orig', action='store_true', help='if true, output first original rgb  image of each batch')
    parser.add_argument('--compare_corrupt', action='store_true', help='set to true to output results across corruptions')
    parser.add_argument('--is_shap', action='store_true', help='no need to call in command line')
    parser.add_argument('--blackbg', action='store_true', help='if True, background color for covering features will be black(0)')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--apply_corruptions', action='store_true', help='apply corruptions to images')
    parser.add_argument('--output_feat_maps', action='store_true', help='toggle this to output images of featmaps')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # acquire_average_auc(opt)
    main(opt)
    # outputOrigImagesOnly(opt)
