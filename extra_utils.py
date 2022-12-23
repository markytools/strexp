import os
import time
import sys
import string
import argparse
import re
import validators

import itertools
import statistics
import heapq as hq
import pickle
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from numpy import linspace
from nltk.metrics.distance import edit_distance
import warnings
from sklearn.utils import check_random_state
import seaborn as sns
warnings.filterwarnings("ignore")

from train_shap_corr import getPredAndConf
from scipy import stats
from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate
from model import Model, STRScore
from utils import get_args

from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import gray2rgb
from lime import lime_image
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

from lib.gradients import VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def deleteInf():
    for i in range(0, 8271):
        filename = "/media/markytools/NewVol2TB/STR/STRVITmodels/infoutdir_gpu0/influence_results_tmp_0_400000_last-i_"+str(i)+".json"
        if os.path.exists(filename):
            os.remove(filename)

def evaluate_selectivity_shap(opt): #, calculate_infer_time=False):
    ### Extra variables
    shapImgLs = []
    costSamples = []
    shapValueLs = []
    maxTestSamples = 10000 ### So that DeepSHAP will not give CPU error
    middleMaskThreshold = 0.0005
    # maxImgPrintout = 50 ### Maximum number for each hit and nothit prediction shap printout
    testImgIdx = 1### 1 to len(testList)
    maxTrainCollect = 1
    maxImgBG = 15
    imgOutWidth = 224
    imgOutHeight = 224
    gridPixelsX = 28 ### There is a 14X14 pixel amount to be covered for each rectangular grid
    gridPixelsY = 28
    explainabilityDirs = "./explainability/SHAP/"
    heapSHAPOutdir = "./metrics/selectivitySHAP/"
    currentFeaturesToRemove = 1 ### Increment this value every iteration after acquiring accuracy
    ### E.G. numFeaturesAccDict[1] = 0.78 means that 1 feature removed has whole test dataset accuracy of 0.78
    numFeaturesAccDict = {}
    numFeaturesConfDict = {} ### Same as accuracy dictionary above, except here you store the avearge confidence
    # maxHeapList = []

    start_time = time.time()

    if not os.path.exists(explainabilityDirs):
        os.makedirs(explainabilityDirs)

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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        print("opt.saved_model: ", opt.saved_model)
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
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

    opt.eval = True

    # if opt.fast_acc:
    # # # To easily compute the total accuracy of our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    # else:
    #     # The evaluation datasets, dataset order is same with Table 1 in our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    ### Only IIIT5k_3000
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IIIT5k_3000']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000']

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    testImgCount = 0
    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
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
            img_numpy = image_tensors.cpu().detach().numpy()[0]
            labelsStr = str(labels[0]).lower()
            shapImgLs.append(img_numpy)
            testImgCount += 1
    random.shuffle(shapImgLs)
    shapImgLs = shapImgLs[0:maxImgBG]
    if opt.blackbg:
        shapImgLs = np.zeros(shape=(1, 1, 224, 224)).astype(np.float32)

    # Acquire SHAP values and map its corresponding image pixel coordinates
    # testImgCount = 0
    # list_accuracy = []
    # total_forward_time = 0
    # total_evaluation_data_number = 0
    # total_correct_number = 0
    # log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    # dashed_line = '-' * 80
    # print(dashed_line)
    # log.write(dashed_line + '\n')
    # for eval_data in eval_data_list:
    #     eval_data_path = os.path.join(opt.eval_data, eval_data)
    #     AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
    #     eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
    #     evaluation_loader = torch.utils.data.DataLoader(
    #         eval_data, batch_size=1,
    #         shuffle=False,
    #         num_workers=int(opt.workers),
    #         collate_fn=AlignCollate_evaluation, pin_memory=True)
    #     for i, (image_tensors, labels) in enumerate(evaluation_loader):
    #         labelStrVal = str(labels[0])
    #         model_obj.setGTText(labelStrVal)
    #
    #         ### Acquire shap values
    #         img_numpy = image_tensors.cpu().detach().numpy()
    #         singleImg = np.array([img_numpy[0]])
    #         singleImg = torch.from_numpy(singleImg).to(device)
    #         trainList = np.array(shapImgLs)
    #         background = torch.from_numpy(trainList).to(device)
    #         # print("background shape: ", background.shape) # 15, 1, 224, 224
    #         # print("background shape:", background.shape)
    #         # print("background type: ", background.dtype)
    #         e = shap.DeepExplainer(super_pixel_model, background)
    #         shap_values = e.shap_values(singleImg) ### (1,1,224,224)
    #
    #         shapLocalDict = {}
    #         for xi in range(0, imgOutWidth):
    #             for yi in range(0, imgOutHeight):
    #                 pixelShapVal = shap_values[0,0,xi,yi]
    #                 if -pixelShapVal not in shapLocalDict:
    #                     shapLocalDict[-pixelShapVal] = []
    #                 ### Note here pixelShapVal is negative for heapq to pop maximum shap
    #                 shapLocalDict[-pixelShapVal].append((xi,yi))
    #         shapLocalDict = list(shapLocalDict.items()) ### Convert to list
    #         hq.heapify(shapLocalDict)
    #         emptyNP = np.copy(trainList)
    #         for sortedSHAPIdx, sortedSHAPVal in enumerate(shapLocalDict):
    #             coordsList = sortedSHAPVal[1]
    #             for coordXY in coordsList:
    #                 emptyNP[0,0,coordXY[0],coordXY[1]] = sortedSHAPIdx+1 ### Starting from 1-N
    #         with open(heapSHAPOutdir+"heapSHAP"+str(testImgCount)+".pkl", 'wb') as f:
    #             pickle.dump(emptyNP, f)
    #         # maxHeapList.append(shapLocalDict)
    #         testImgCount += 1

    ## Hide highest shap valued pixels with background pixel
    maxFeaturesToRemove = 50000
    breakOuterLoop = False

    print("Reached last loop --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    averagesLs = []
    aveConfidencesLs = []
    featRemoveNumLs = []
    for featRemoveNum in range(1, maxFeaturesToRemove):
        featRemoveNum=featRemoveNum*16
        testImgCount = 0
        n_correct = 0
        confidenceLs = []
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
                gt = str(labels[0])
                model_obj.setGTText(gt)
                # print("image_tensors shape: ", image_tensors.shape) # 1, 1, 224, 224
                image_tensors_hidden = torch.clone(image_tensors)
                trainList = np.array(shapImgLs)
                background = torch.from_numpy(trainList).to(device)
                backgroundMean = torch.mean(background, dim=0).unsqueeze(0)
                with open(heapSHAPOutdir+"heapSHAP"+str(testImgCount)+".pkl", 'rb') as f:
                    shapRankedNP = pickle.load(f)
                # print("backgroundMean shape: ", backgroundMean.shape)
                # print("image_tensors_hidden shape: ", image_tensors_hidden.shape)
                # print("type shapRankedNP: ", type(shapRankedNP))
                # print("shapRankedNP shape: ", shapRankedNP.shape)
                image_tensors_hidden = image_tensors_hidden.to(device)
                shapRankedNP = torch.from_numpy(shapRankedNP).to(device)
                image_tensors_hidden = torch.where(shapRankedNP<=featRemoveNum, backgroundMean, image_tensors_hidden)
                # for featToHideIdx in range(0, featRemoveNum):
                #     if featToHideIdx >= len(imgHeapqLs):
                #         breakOuterLoop = True
                #         break
                #     pixelList = imgHeapqLs[featToHideIdx][1]
                #     if len(pixelList)<=0: assert(False) ### Abnormal event
                #     for pixelDataXY in pixelList:
                #         image_tensors_hidden[0,0,pixelDataXY[0],pixelDataXY[1]] = backgroundMean[0,pixelDataXY[0],pixelDataXY[1]]
                if breakOuterLoop: break
                imgD = image_tensors_hidden.cpu().detach().numpy()[0]
                singleImg = np.array([imgD])
                singleImg = torch.from_numpy(singleImg).to(device)
                pred, confScore = getPredAndConf(opt, model, scoring, singleImg, converter, np.array([gt]))
                # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
                if pred == gt:
                    n_correct += 1
                confidenceLs.append(confScore[0][0]*100)
                testImgCount += 1
            if breakOuterLoop: break
        if breakOuterLoop: break
        print("featRemoveNum: ", featRemoveNum)
        featRemoveNumLs.append(featRemoveNum)
        aveConfidencesLs.append(statistics.mean(confidenceLs))
        averagesLs.append(n_correct / float(testImgCount))
        if (featRemoveNum//16) % 1000 == 0:
            with open("featremovenum.pkl", 'wb') as f:
                pickle.dump(featRemoveNumLs, f)
            with open("aveconflist.pkl", 'wb') as f:
                pickle.dump(aveConfidencesLs, f)
            with open("avelist.pkl", 'wb') as f:
                pickle.dump(averagesLs, f)
    print("FINISHED IN --- %s seconds ---" % (time.time() - start_time))

def produceAttrMethodScores(opt, model, scoring, converter, gt, output_values, orig_img_tensors, segments):
    segmScoreDict = {}
    for segmNum in np.unique(segments):
        segmScore = np.sum(output_values[0,0][segments == segmNum])
        segmScoreDict[segmNum] = segmScore
    sortedOutputKeys = [k for k, v in sorted(segmScoreDict.items(), key=lambda item: item[1])]
    sortedOutputKeys = sortedOutputKeys[::-1] ### A list that should contain largest to smallest score

    ### First index is one feature removed, second index two features removed, and so on...
    n_correct = []
    confidenceList = [] # First index is one feature removed, second index two features removed, and so on...
    inputClonedImg = torch.clone(orig_img_tensors)
    for totalSegToHide in range(0, len(sortedOutputKeys)):
        ### Acquire VanGrad prediction result
        currentSegmentToHide = sortedOutputKeys[totalSegToHide]
        inputClonedImg[0,0][segments == currentSegmentToHide] = 0.0
        pred, confScore = getPredAndConf(opt, model, scoring, inputClonedImg, converter, np.array([gt]))
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
        confidenceList.append(confScore[0][0]*100)

    return n_correct, confidenceList, sortedOutputKeys

def evaluate_selectivity_attr_methods(opt): #, calculate_infer_time=False):
    ### Extra variables
    shapImgLs = []
    costSamples = []
    maxTestSamples = 10000 ### So that DeepSHAP will not give CPU error
    middleMaskThreshold = 0.0005
    # maxImgPrintout = 50 ### Maximum number for each hit and nothit prediction shap printout
    testImgIdx = 1### 1 to len(testList)
    maxTrainCollect = 1
    maxImgBG = 15
    imgOutWidth = 224
    imgOutHeight = 224
    gridPixelsX = 28 ### There is a 14X14 pixel amount to be covered for each rectangular grid
    gridPixelsY = 28
    explainabilityDirs = "./explainability/SHAP/"
    heapSHAPOutdir = "./metrics/selectivitySHAP/"
    selectivityOutput = "/home/markytools/Documents/MSEEThesis/STR/str_vit/deep-text-recognition-benchmark/explainability/VITSTR/selectivity/"
    custom_segm_dataroot = "/home/markytools/Documents/MSEEThesis/STR/str_vit/deep-text-recognition-benchmark/segmdata/224X224/"
    currentFeaturesToRemove = 1 ### Increment this value every iteration after acquiring accuracy
    ### E.G. numFeaturesAccDict[1] = 0.78 means that 1 feature removed has whole test dataset accuracy of 0.78
    numFeaturesAccDict = {}
    numFeaturesConfDict = {} ### Same as accuracy dictionary above, except here you store the avearge confidence
    # maxHeapList = []

    start_time = time.time()

    if not os.path.exists(explainabilityDirs):
        os.makedirs(explainabilityDirs)

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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        print("opt.saved_model: ", opt.saved_model)
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
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

    # if opt.fast_acc:
    # # # To easily compute the total accuracy of our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    # else:
    #     # The evaluation datasets, dataset order is same with Table 1 in our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    ### Only IIIT5k_3000
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IIIT5k_3000']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000']

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
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            # print("orig_img_tensors shape: ", orig_img_tensors.shape) # (1, 1, 224, 224)
            with open(custom_segm_dataroot+"vitstr_lime_segm_{}.pkl".format(testImgCount), 'rb') as f:
                customSegmData = pickle.load(f)
            gt = str(labels[0])
            model_obj.setGTText(gt)
            # Acquire trainable imagenumpy
            image_tensors = torch.clone(orig_img_tensors)
            detachedImg = image_tensors.cpu().detach().numpy()[0]
            inputImg = np.array([detachedImg])
            inputImg = torch.from_numpy(inputImg).to(device)
            # print("inputImg shape: ", inputImg.shape) # (1,1,224,224)
            img_numpy = detachedImg.squeeze(0).astype('float64')
            img_numpy = gray2rgb(img_numpy)
            # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img_numpy, inputImg,
                                                     super_pixel_model, # classification function
                                                     top_labels=1,
                                                     hide_color=0,
                                                     num_samples=100,
                                                     batch_size=1,
                                                     squaredSegm=-2,
                                                     loadedSegmData=customSegmData["lime_segm"]) # number of images that will be sent to classification function
            ind =  explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            # dict_heatmap --> {0: -0.2424142, 1: 0.92214}, where 0, 1 is the segm number and the values are the LIME scores
            # Need to acquire segment keys. Here, we reverse because we will hiding the largest to smallest value
            sortedKeys = [k for k, v in sorted(dict_heatmap.items(), key=lambda item: item[1])]
            sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
            segments = explanation.segments ### (224,224)
            totalUniqueSegm = len(np.unique(segments))

            ### Acquire vanilla gradients
            super_pixel_model.train()
            super_pixel_model.zero_grad()
            vanGradImg = torch.clone(orig_img_tensors)
            vanGradImg = Variable(vanGradImg.to(device), requires_grad=True)
            vanilla_grad = VanillaGrad(pretrained_model=super_pixel_model, cuda=True)
            vanilla_saliency = vanilla_grad(vanGradImg, index=0)
            # save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
            # print('Saved vanilla gradient image')
            vangrad_values = vanilla_saliency[np.newaxis, ...]
            # vangrad_values shape:  (1,1,224,224)

            ### Acquire guided backprop output
            super_pixel_model.zero_grad()
            guidedBackpropImg = torch.clone(orig_img_tensors)
            guidedBackpropImg = Variable(guidedBackpropImg.to(device), requires_grad=True)
            guided_grad = GuidedBackpropGrad(pretrained_model=super_pixel_model, cuda=True)
            guided_saliency = guided_grad(guidedBackpropImg, index=0)
            # save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
            # print('Saved vanilla gradient image')
            guidedprop_values = guided_saliency[np.newaxis, ...]
            # vangrad_values shape:  (1,1,224,224)

            ### Acquire smoothgrad output
            super_pixel_model.zero_grad()
            smoothGradImg = torch.clone(orig_img_tensors)
            smoothGradImg = Variable(smoothGradImg.to(device), requires_grad=True)
            smooth_grad = SmoothGrad(
                pretrained_model=super_pixel_model,
                cuda=True,
                n_samples=10,
                magnitude=True)
            smooth_saliency = smooth_grad(smoothGradImg, index=0)
            # save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
            # print('Saved vanilla gradient image')
            smoothgrad_values = smooth_saliency[np.newaxis, ...]
            # vangrad_values shape:  (1,1,224,224)

            super_pixel_model.eval()

            # define a function that depends on a binary mask representing if an image region is hidden
            def mask_image(zs, segmentation, image, background=None):
                if background is None:
                    background = image.mean((0,1))
                out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
                for i in range(zs.shape[0]):
                    out[i,:,:,:] = image
                    for j in range(zs.shape[1]):
                        if zs[i,j] == 0:
                            out[i][:,segmentation == j] = background
                out = torch.from_numpy(out).type(torch.FloatTensor)
                return out
            def f(z):
                shap_orig_tensor = torch.clone(orig_img_tensors)
                maskedImg = mask_image(z, segments, shap_orig_tensor[0], 255)
                modelOutput = super_pixel_model(maskedImg)
                modelOutput = modelOutput.cpu().detach().numpy()
                return modelOutput

            ### Acquire SHAP from explanation too
            e = shap.KernelExplainer(f, np.zeros((1,totalUniqueSegm)))
            shap_values = e.shap_values(np.ones((1,totalUniqueSegm)), nsamples=100)
            batchOneData = shap_values[0]
            dict_heatmap_shap = {}
            for segIdx, shapValNum in enumerate(batchOneData[0]):
                dict_heatmap_shap[segIdx] = shapValNum
            sortedSHAPKeys = [k for k, v in sorted(dict_heatmap_shap.items(), key=lambda item: item[1])]
            sortedSHAPKeys = sortedSHAPKeys[::-1] ### A list that should contain largest to smallest score

            # print("shap_values len: ", len(shap_values)) # BatchSize
            # print("shap_values[0] shape: ", shap_values[0].shape) ### (1,totalUniqueSegm)
            # print("explanation.segments shape: ", explanation.segments.shape) (224,224)

            ### First index is one feature removed, second index two features removed, and so on...
            ### 1 - correct, 0 - incorrect
            n_correct_lime = []
            confidenceList_lime = [] # First index is one feature removed, second index two features removed, and so on...
            limeImg = torch.clone(orig_img_tensors)
            for totalSegToHide in range(0, len(sortedKeys)):
                ### Acquire LIME prediction result
                currentSegmentToHide = sortedKeys[totalSegToHide]
                limeImg[0,0][segments == currentSegmentToHide] = 0.0
                pred, confScore = getPredAndConf(opt, model, scoring, limeImg, converter, np.array([gt]))
                # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
                if pred == gt:
                    n_correct_lime.append(1)
                else:
                    n_correct_lime.append(0)
                confidenceList_lime.append(confScore[0][0]*100)

            ### First index is one feature removed, second index two features removed, and so on...
            ### 1 - correct, 0 - incorrect
            n_correct_shap = []
            confidenceList_shap = [] # First index is one feature removed, second index two features removed, and so on...
            shapImg = torch.clone(orig_img_tensors)
            for totalSegToHide in range(0, len(sortedSHAPKeys)):
                ### Acquire SHAP prediction result
                currentSegmentToHide = sortedSHAPKeys[totalSegToHide]
                shapImg[0,0][segments == currentSegmentToHide] = 0.0
                pred, confScore = getPredAndConf(opt, model, scoring, shapImg, converter, np.array([gt]))
                # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)
                if pred == gt:
                    n_correct_shap.append(1)
                else:
                    n_correct_shap.append(0)
                confidenceList_shap.append(confScore[0][0]*100)

            # n_correct_shap, confidenceList_shap, sortedSHAPKeys = produceAttrMethodScores(\
            # opt, model, scoring, converter, gt, shap_values, orig_img_tensors, segments)
            n_correct_vangrad, confidenceList_vangrad, sortedVanGradKeys = produceAttrMethodScores(\
            opt, model, scoring, converter, gt, vangrad_values, orig_img_tensors, segments)
            n_correct_guidedbp, confidenceList_guidedbp, sortedGuidedBPKeys = produceAttrMethodScores(\
            opt, model, scoring, converter, gt, guidedprop_values, orig_img_tensors, segments)
            n_correct_smoothgrad, confidenceList_smoothgrad, sortedSmoothGradKeys = produceAttrMethodScores(\
            opt, model, scoring, converter, gt, smoothgrad_values, orig_img_tensors, segments)

            results_dict = {}
            results_dict["lime_acc"] = n_correct_lime
            results_dict["lime_conf"] = confidenceList_lime
            results_dict["lime_segmrank"] = sortedKeys
            results_dict["shap_acc"] = n_correct_shap
            results_dict["shap_conf"] = confidenceList_shap
            results_dict["shap_segmrank"] = sortedSHAPKeys
            results_dict["vangrad_acc"] = n_correct_vangrad
            results_dict["vangrad_conf"] = confidenceList_vangrad
            results_dict["vangrad_segmrank"] = sortedVanGradKeys
            results_dict["guidedbp_acc"] = n_correct_guidedbp
            results_dict["guidedbp_conf"] = confidenceList_guidedbp
            results_dict["guidedbp_segmrank"] = sortedGuidedBPKeys
            results_dict["smoothgrad_acc"] = n_correct_smoothgrad
            results_dict["smoothgrad_conf"] = confidenceList_smoothgrad
            results_dict["smoothgrad_segmrank"] = sortedSmoothGradKeys
            selectivity_eval_results.append(results_dict)

            with open("selectivity_eval_results.pkl", 'wb') as f:
                pickle.dump(selectivity_eval_results, f)

            testImgCount += 1

### Once you have the selectivity_eval_results.pkl file,
def acquire_selectivity_auc(opt, pkl_filename=None):
    if pkl_filename is None:
        # pkl_filename = "/home/markytools/Documents/MSEEThesis/STR/deep-text-recognition-benchmark-deepshap/selectivity_eval_results.pkl" # TRBA
        pkl_filename = "/home/markytools/Documents/MSEEThesis/STR/str_vit/deep-text-recognition-benchmark/selectivity_eval_results.pkl" # VITSTR
        # pkl_filename = "/home/markytools/Documents/MSEEThesis/STR/ABINet/selectivity_eval_results.pkl" # ABINET

    with open(pkl_filename, 'rb') as f:
        selectivity_data = pickle.load(f)

    smallestRemovedFeats = 100000
    for resDict in selectivity_data:
        n_correct_lime = resDict["lime_acc"]
        confidenceList_lime = resDict["lime_conf"]
        n_correct_shap = resDict["shap_acc"]
        confidenceList_shap = resDict["shap_conf"]
        n_correct_vangrad = resDict["vangrad_acc"]
        confidenceList_vangrad = resDict["vangrad_conf"]
        n_correct_guidedbp = resDict["guidedbp_acc"]
        confidenceList_guidedbp = resDict["guidedbp_conf"]
        n_correct_smoothgrad = resDict["smoothgrad_acc"]
        confidenceList_smoothgrad = resDict["smoothgrad_conf"]
        minFeatRemovedTotal = min(len(n_correct_lime), len(confidenceList_lime))
        if minFeatRemovedTotal < smallestRemovedFeats:
            smallestRemovedFeats = minFeatRemovedTotal
    totalImages = len(selectivity_data)
    lime_accList = []
    lime_confList = []
    shap_accList = []
    shap_confList = []
    vangrad_accList = []
    vangrad_confList = []
    guidedbp_accList = []
    guidedbp_confList = []
    smoothgrad_accList = []
    smoothgrad_confList = []
    for numFeatRemoved in range(0, smallestRemovedFeats): ### Total features removed
        lime_total_hit = 0
        lime_confToAve = []
        shap_total_hit = 0
        shap_confToAve = []
        vangrad_total_hit = 0
        vangrad_confToAve = []
        guidedbp_total_hit = 0
        guidedbp_confToAve = []
        smoothgrad_total_hit = 0
        smoothgrad_confToAve = []
        for resDict in selectivity_data: ### For every image
            lime_hit = resDict["lime_acc"][numFeatRemoved]
            lime_conf = resDict["lime_conf"][numFeatRemoved]
            shap_hit = resDict["shap_acc"][numFeatRemoved]
            shap_conf = resDict["shap_conf"][numFeatRemoved]
            vangrad_hit = resDict["vangrad_acc"][numFeatRemoved]
            vangrad_conf = resDict["vangrad_conf"][numFeatRemoved]
            guidedbp_hit = resDict["guidedbp_acc"][numFeatRemoved]
            guidedbp_conf = resDict["guidedbp_conf"][numFeatRemoved]
            smoothgrad_hit = resDict["smoothgrad_acc"][numFeatRemoved]
            smoothgrad_conf = resDict["smoothgrad_conf"][numFeatRemoved]
            lime_total_hit += lime_hit
            lime_confToAve.append(lime_conf)
            shap_total_hit += shap_hit
            shap_confToAve.append(shap_conf)
            vangrad_total_hit += vangrad_hit
            vangrad_confToAve.append(vangrad_conf)
            guidedbp_total_hit += guidedbp_hit
            guidedbp_confToAve.append(guidedbp_conf)
            smoothgrad_total_hit += smoothgrad_hit
            smoothgrad_confToAve.append(smoothgrad_conf)
        lime_accList.append(lime_total_hit/float(totalImages))
        lime_confList.append(statistics.mean(lime_confToAve))
        shap_accList.append(shap_total_hit/float(totalImages))
        shap_confList.append(statistics.mean(shap_confToAve))
        vangrad_accList.append(vangrad_total_hit/float(totalImages))
        vangrad_confList.append(statistics.mean(vangrad_confToAve))
        guidedbp_accList.append(guidedbp_total_hit/float(totalImages))
        guidedbp_confList.append(statistics.mean(guidedbp_confToAve))
        smoothgrad_accList.append(smoothgrad_total_hit/float(totalImages))
        smoothgrad_confList.append(statistics.mean(smoothgrad_confToAve))
    print("lime_accList AUC: ", np.trapz(lime_accList))
    print("lime_confList AUC: ", np.trapz(lime_confList))
    print("shap_accList AUC: ", np.trapz(shap_accList))
    print("shap_confList AUC: ", np.trapz(shap_confList))
    print("vangrad_accList AUC: ", np.trapz(vangrad_accList))
    print("vangrad_confList AUC: ", np.trapz(vangrad_confList))
    print("guidedbp_accList AUC: ", np.trapz(guidedbp_accList))
    print("guidedbp_confList AUC: ", np.trapz(guidedbp_confList))
    print("smoothgrad_accList AUC: ", np.trapz(smoothgrad_accList))
    print("smoothgrad_confList AUC: ", np.trapz(smoothgrad_confList))
    return lime_accList, lime_confList, shap_accList, shap_confList, vangrad_accList, vangrad_confList,\
    guidedbp_accList, guidedbp_confList, smoothgrad_accList, smoothgrad_confList

def acquireSelectivityPlots(opt):
    strNetworks = [
    "TRBA",
    "VITSTR"
    ]
    pklFilenames = [
    "/home/markytools/Documents/MSEEThesis/STR/deep-text-recognition-benchmark-deepshap/selectivity_eval_results.pkl",
    "/home/markytools/Documents/MSEEThesis/STR/str_vit/deep-text-recognition-benchmark/selectivity_eval_results.pkl"
    ]
    explainability_methods = [
    "Vanilla Gradients",
    "LIME",
    "SHAP"
    ]
    metrics = [
    "Accuracy",
    "Confidence"
    ]
    colorList = []
    fig, axs = plt.subplots(2, 2)
    palette = itertools.cycle(sns.color_palette())
    for i in range(0, len(explainability_methods)):
        c = next(palette)
        colorList.append(c)
        print("c: ", c)
    fig, axs = plt.subplots(2, 2)
    for strArchIdx, pklFile in enumerate(pklFilenames):
        lime_accList, lime_confList, shap_accList, shap_confList, vangrad_accList, vangrad_confList = \
        acquire_selectivity_auc(opt, pklFile)
        minFeaturesRemoved = len(lime_accList)
        exp_methods_accdata = [vangrad_accList, lime_accList, shap_accList]
        exp_methods_confdata = [vangrad_confList, lime_confList, shap_confList]
        for expidx, expName in enumerate(exp_methods_accdata): ### Plot accuracy
            axs[strArchIdx, 0].plot(expName, color=colorList[expidx])
        for expidx, expName in enumerate(exp_methods_confdata): ### Plot confidence
            axs[strArchIdx, 1].plot(expName, color=colorList[expidx])
    # lgd = fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('selectivityAllPlot.png')
    plt.clf()
    # color_cycle = plt.rcParams['axes.prop_cycle']()
    # palette = itertools.cycle(sns.color_palette())
    # c = next(palette)
    #
    # # I need some curves to plot
    #
    # x = linspace(0, 1, 51)
    # f1 = x*(1-x)   ; lab1 = 'x - x x'
    # f2 = 0.25-f1   ; lab2 = '1/4 - x + x x'
    # f3 = x*x*(1-x) ; lab3 = 'x x - x x x'
    # f4 = 0.25-f3   ; lab4 = '1/4 - x x + x x x'
    #
    # # let's plot our curves (note the use of color cycle, otherwise the curves colors in
    # # the two subplots will be repeated and a single legend becomes difficult to read)
    # # fig, (a13, a14, a24, a25) = plt.subplots(4)
    # fig, axs = plt.subplots(2, 2)
    #
    # axs[0, 0].plot(x, f1, label=lab1, color=c)
    # c = next(palette)
    # axs[0, 1].plot(x, f3, label=lab3, color=c)
    # axs[1, 0].plot(x, f2, label=lab2, color=c)
    # axs[1, 1].plot(x, f4, label=lab4, color=c)
    #
    # # so far so good, now the trick
    #
    # lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #
    # # finally we invoke the legend (that you probably would like to customize...)
    #
    # lgd = fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('selectivityAllPlot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plt.clf()

def evaluate_continuity_attr_methods(opt): #, calculate_infer_time=False):
    ### Extra variables
    shapImgLs = []
    costSamples = []
    shapValueLs = []
    maxTestSamples = 10000 ### So that DeepSHAP will not give CPU error
    middleMaskThreshold = 0.0005
    # maxImgPrintout = 50 ### Maximum number for each hit and nothit prediction shap printout
    testImgIdx = 1### 1 to len(testList)
    maxTrainCollect = 1
    maxImgBG = 15
    imgOutWidth = 224
    imgOutHeight = 224
    gridPixelsX = 28 ### There is a 14X14 pixel amount to be covered for each rectangular grid
    gridPixelsY = 28
    explainabilityDirs = "./explainability/SHAP/"
    heapSHAPOutdir = "./metrics/selectivitySHAP/"
    currentFeaturesToRemove = 1 ### Increment this value every iteration after acquiring accuracy
    ### E.G. numFeaturesAccDict[1] = 0.78 means that 1 feature removed has whole test dataset accuracy of 0.78
    numFeaturesAccDict = {}
    numFeaturesConfDict = {} ### Same as accuracy dictionary above, except here you store the avearge confidence
    # maxHeapList = []

    start_time = time.time()

    if not os.path.exists(explainabilityDirs):
        os.makedirs(explainabilityDirs)

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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        print("opt.saved_model: ", opt.saved_model)
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
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

    # if opt.fast_acc:
    # # # To easily compute the total accuracy of our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    # else:
    #     # The evaluation datasets, dataset order is same with Table 1 in our paper.
    #     eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                       'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    # test set length:
    # SVT: 647
    # IC03_860: 860
    # IC13_857: 857
    # SVTP: 645
    # IC15_2077: 2077
    # CUTE80: 288

    ## Only IIIT5k_3000
    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = ['CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['CUTE80']

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    selectivity_eval_results = []
    pixelGap = 6.72 ### adjusted for TRBA 100width with 3 pixels per translation
    testImgCount = 0
    halfIntervals=16 ### Translate to the left side 16 times, and to the right side 16 times
    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')

    ### delete this later
    # maxBatchSize = 10
    # start_time = time.time()
    ###

    ### Continuity variables
    r1ExpScoresVangrad = []
    r1ExpScoresLIME = []
    r1ExpScoresSHAP = []
    r2ExpScoresVangrad = []
    r2ExpScoresLIME = []
    r2ExpScoresSHAP = []
    r3ExpScoresVangrad = []
    r3ExpScoresLIME = []
    r3ExpScoresSHAP = []
    r4ExpScoresVangrad = []
    r4ExpScoresLIME = []
    r4ExpScoresSHAP = []
    fExpScoresVangrad = []
    fExpScoresLIME = []
    fExpScoresSHAP = []

    for transNum in range(0, halfIntervals+halfIntervals+1):
        r1ExpScoresVangrad.append([])
        r1ExpScoresLIME.append([])
        r1ExpScoresSHAP.append([])
        r2ExpScoresVangrad.append([])
        r2ExpScoresLIME.append([])
        r2ExpScoresSHAP.append([])
        r3ExpScoresVangrad.append([])
        r3ExpScoresLIME.append([])
        r3ExpScoresSHAP.append([])
        r4ExpScoresVangrad.append([])
        r4ExpScoresLIME.append([])
        r4ExpScoresSHAP.append([])
        fExpScoresVangrad.append([])
        fExpScoresLIME.append([])
        fExpScoresSHAP.append([])

    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            # orig_img_tensors.shape # (1, 1, 224height, 224width)
            gt = str(labels[0])
            model_obj.setGTText(gt)
            ### total of 16+16+1
            translationNum = 0

            # Acquire trainable imagenumpy
            image_tensors = torch.clone(orig_img_tensors)

            ### Acquire non-translated continuity data
            detachedImg = image_tensors.cpu().detach().numpy()[0]
            inputImg = np.array([detachedImg])
            inputImg = torch.from_numpy(inputImg).to(device)
            # print("inputImg shape: ", inputImg.shape) # (1,1,224,224)
            img_numpy = detachedImg.squeeze(0).astype('float64')
            img_numpy = gray2rgb(img_numpy)
            # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img_numpy, inputImg,
                                                     super_pixel_model, # classification function
                                                     top_labels=1,
                                                     hide_color=0,
                                                     num_samples=500,
                                                     batch_size=1,
                                                     squaredSegm=4) # number of images that will be sent to classification function
            ind =  explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            # Need to acquire segment keys. Here, we reverse because we will hiding the largest to smallest value
            sortedKeys = [k for k, v in sorted(dict_heatmap.items(), key=lambda item: item[1])]
            sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
            segments = explanation.segments ### (224,224)

            ### Acquire vanilla gradients
            model.train()
            vanGradImg = torch.clone(image_tensors)
            vanGradImg = Variable(vanGradImg.to(device), requires_grad=True)
            target = converter.encode(labels)
            preds = model(vanGradImg, text=target, seqlen=converter.batch_max_length)
            one_hot = np.zeros(preds.shape, dtype=np.float32)
            one_hot = Variable(torch.from_numpy(one_hot).to(device), requires_grad=False)
            for idx, val in enumerate(target[0]):### Important for batchsize to be one
                one_hot[0,idx,val] = 1.0
            one_hot = torch.sum(one_hot * preds)
            model.zero_grad()
            one_hot.backward(retain_graph=True)
            vangrad_values = vanGradImg.grad.data.cpu().numpy()
            # vangrad_values shape:  (1,1,224,224)
            model.eval()

            ### Acquire SHAP from explanation too
            e = shap.DeepExplainer(super_pixel_model, background)
            shap_values = e.shap_values(inputImg) ### (1,1,224,224)
            # print("explanation.segments shape: ", explanation.segments.shape) (224,224)

            ### Add all scores for continuity, center(excluded) to right(included)
            limeContScores = [dict_heatmap[0], dict_heatmap[1], dict_heatmap[2], dict_heatmap[3]]
            r1ExpScoresLIME[translationNum].append(float(limeContScores[0]))
            r2ExpScoresLIME[translationNum].append(float(limeContScores[1]))
            r3ExpScoresLIME[translationNum].append(float(limeContScores[2]))
            r4ExpScoresLIME[translationNum].append(float(limeContScores[3]))
            fExpScoresLIME[translationNum].append(float(sum(limeContScores)))

            vangradContScores = [np.sum(vangrad_values[0,0][segments == 0]), np.sum(vangrad_values[0,0][segments == 1]),\
            np.sum(vangrad_values[0,0][segments == 2]), np.sum(vangrad_values[0,0][segments == 3])]
            r1ExpScoresVangrad[translationNum].append(float(vangradContScores[0]))
            r2ExpScoresVangrad[translationNum].append(float(vangradContScores[1]))
            r3ExpScoresVangrad[translationNum].append(float(vangradContScores[2]))
            r4ExpScoresVangrad[translationNum].append(float(vangradContScores[3]))
            fExpScoresVangrad[translationNum].append(float(sum(vangradContScores)))

            shapContScores = [np.sum(shap_values[0,0][segments == 0]), np.sum(shap_values[0,0][segments == 1]),\
            np.sum(shap_values[0,0][segments == 2]), np.sum(shap_values[0,0][segments == 3])]
            r1ExpScoresSHAP[translationNum].append(float(shapContScores[0]))
            r2ExpScoresSHAP[translationNum].append(float(shapContScores[1]))
            r3ExpScoresSHAP[translationNum].append(float(shapContScores[2]))
            r4ExpScoresSHAP[translationNum].append(float(shapContScores[3]))
            fExpScoresSHAP[translationNum].append(float(sum(shapContScores)))

            ### Left Translation, image will move slowly from center to the left (does not include normal)
            for xleftOff in range(1, halfIntervals+1):
                translationNum += 1

                blankImg = torch.zeros(size=image_tensors.shape, dtype=image_tensors.dtype)
                ### 1 to 16 (zero is not included, which should be separate)
                xOffsetLeft = int(pixelGap*xleftOff)
                sourceStartX = 0 + xOffsetLeft
                sourceEndX = image_tensors.shape[3]
                targetStartX = 0
                targetEndX = image_tensors.shape[3] - xOffsetLeft
                blankImg[:,:,:,targetStartX:targetEndX] = image_tensors[:,:,:,sourceStartX:sourceEndX]

                limeImg = torch.clone(blankImg)
                detachedImg = limeImg.cpu().detach().numpy()[0]
                inputImg = np.array([detachedImg])
                inputImg = torch.from_numpy(inputImg).to(device)
                # print("inputImg shape: ", inputImg.shape) # (1,1,224,224)
                img_numpy = detachedImg.squeeze(0).astype('float64')
                img_numpy = gray2rgb(img_numpy)
                # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(img_numpy, inputImg,
                                                         super_pixel_model, # classification function
                                                         top_labels=1,
                                                         hide_color=0,
                                                         num_samples=100,
                                                         batch_size=1,
                                                         squaredSegm=4) # number of images that will be sent to classification function
                ind =  explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                # Need to acquire segment keys. Here, we reverse because we will hiding the largest to smallest value
                sortedKeys = [k for k, v in sorted(dict_heatmap.items(), key=lambda item: item[1])]
                sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
                segments = explanation.segments ### (224,224)

                ### Acquire vanilla gradients
                model.train()
                vanGradImg = torch.clone(blankImg)
                vanGradImg = Variable(vanGradImg.to(device), requires_grad=True)
                target = converter.encode(labels)
                preds = model(vanGradImg, text=target, seqlen=converter.batch_max_length)
                one_hot = np.zeros(preds.shape, dtype=np.float32)
                one_hot = Variable(torch.from_numpy(one_hot).to(device), requires_grad=False)
                for idx, val in enumerate(target[0]):### Important for batchsize to be one
                    one_hot[0,idx,val] = 1.0
                one_hot = torch.sum(one_hot * preds)
                model.zero_grad()
                one_hot.backward(retain_graph=True)
                vangrad_values = vanGradImg.grad.data.cpu().numpy()
                # vangrad_values shape:  (1,1,224,224)
                model.eval()

                ### Acquire SHAP from explanation too
                e = shap.DeepExplainer(super_pixel_model, background)
                shap_values = e.shap_values(inputImg) ### (1,1,224,224)
                # print("explanation.segments shape: ", explanation.segments.shape) (224,224)

                ### Add all scores for continuity, center(excluded) to left(included)
                limeContScores = [dict_heatmap[0], dict_heatmap[1], dict_heatmap[2], dict_heatmap[3]]
                r1ExpScoresLIME[translationNum].append(float(limeContScores[0]))
                r2ExpScoresLIME[translationNum].append(float(limeContScores[1]))
                r3ExpScoresLIME[translationNum].append(float(limeContScores[2]))
                r4ExpScoresLIME[translationNum].append(float(limeContScores[3]))
                fExpScoresLIME[translationNum].append(float(sum(limeContScores)))

                vangradContScores = [np.sum(vangrad_values[0,0][segments == 0]), np.sum(vangrad_values[0,0][segments == 1]),\
                np.sum(vangrad_values[0,0][segments == 2]), np.sum(vangrad_values[0,0][segments == 3])]
                r1ExpScoresVangrad[translationNum].append(float(vangradContScores[0]))
                r2ExpScoresVangrad[translationNum].append(float(vangradContScores[1]))
                r3ExpScoresVangrad[translationNum].append(float(vangradContScores[2]))
                r4ExpScoresVangrad[translationNum].append(float(vangradContScores[3]))
                fExpScoresVangrad[translationNum].append(float(sum(vangradContScores)))

                shapContScores = [np.sum(shap_values[0,0][segments == 0]), np.sum(shap_values[0,0][segments == 1]),\
                np.sum(shap_values[0,0][segments == 2]), np.sum(shap_values[0,0][segments == 3])]
                r1ExpScoresSHAP[translationNum].append(float(shapContScores[0]))
                r2ExpScoresSHAP[translationNum].append(float(shapContScores[1]))
                r3ExpScoresSHAP[translationNum].append(float(shapContScores[2]))
                r4ExpScoresSHAP[translationNum].append(float(shapContScores[3]))
                fExpScoresSHAP[translationNum].append(float(sum(shapContScores)))

            ### Right Translation, image will move slowly from center to the right (does not include normal)
            for xrightOff in range(1, halfIntervals+1):
                translationNum += 1

                blankImg = torch.zeros(size=image_tensors.shape, dtype=image_tensors.dtype)
                ### 1 to 16 (zero is not included, which should be separate)
                xOffsetRight = int(pixelGap*xrightOff)
                sourceStartX = 0
                sourceEndX = image_tensors.shape[3] - xOffsetRight
                targetStartX = xOffsetRight
                targetEndX = image_tensors.shape[3]
                blankImg[:,:,:,targetStartX:targetEndX] = image_tensors[:,:,:,sourceStartX:sourceEndX]

                limeImg = torch.clone(blankImg)
                detachedImg = limeImg.cpu().detach().numpy()[0]
                inputImg = np.array([detachedImg])
                inputImg = torch.from_numpy(inputImg).to(device)
                # print("inputImg shape: ", inputImg.shape) # (1,1,224,224)
                img_numpy = detachedImg.squeeze(0).astype('float64')
                img_numpy = gray2rgb(img_numpy)
                # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(img_numpy, inputImg,
                                                         super_pixel_model, # classification function
                                                         top_labels=1,
                                                         hide_color=0,
                                                         num_samples=100,
                                                         batch_size=1,
                                                         squaredSegm=4) # number of images that will be sent to classification function
                ind =  explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                # Need to acquire segment keys. Here, we reverse because we will hiding the largest to smallest value
                sortedKeys = [k for k, v in sorted(dict_heatmap.items(), key=lambda item: item[1])]
                sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
                segments = explanation.segments ### (224,224)

                ### Acquire vanilla gradients
                model.train()
                vanGradImg = torch.clone(blankImg)
                vanGradImg = Variable(vanGradImg.to(device), requires_grad=True)
                target = converter.encode(labels)
                preds = model(vanGradImg, text=target, seqlen=converter.batch_max_length)
                one_hot = np.zeros(preds.shape, dtype=np.float32)
                one_hot = Variable(torch.from_numpy(one_hot).to(device), requires_grad=False)
                for idx, val in enumerate(target[0]):### Important for batchsize to be one
                    one_hot[0,idx,val] = 1.0
                one_hot = torch.sum(one_hot * preds)
                model.zero_grad()
                one_hot.backward(retain_graph=True)
                vangrad_values = vanGradImg.grad.data.cpu().numpy()
                # vangrad_values shape:  (1,1,224,224)
                model.eval()

                ### Acquire SHAP from explanation too
                e = shap.DeepExplainer(super_pixel_model, background)
                shap_values = e.shap_values(inputImg) ### (1,1,224,224)
                # print("explanation.segments shape: ", explanation.segments.shape) (224,224)

                ### Add all scores for continuity, center(excluded) to right(included)
                limeContScores = [dict_heatmap[0], dict_heatmap[1], dict_heatmap[2], dict_heatmap[3]]
                r1ExpScoresLIME[translationNum].append(float(limeContScores[0]))
                r2ExpScoresLIME[translationNum].append(float(limeContScores[1]))
                r3ExpScoresLIME[translationNum].append(float(limeContScores[2]))
                r4ExpScoresLIME[translationNum].append(float(limeContScores[3]))
                fExpScoresLIME[translationNum].append(float(sum(limeContScores)))

                vangradContScores = [np.sum(vangrad_values[0,0][segments == 0]), np.sum(vangrad_values[0,0][segments == 1]),\
                np.sum(vangrad_values[0,0][segments == 2]), np.sum(vangrad_values[0,0][segments == 3])]
                r1ExpScoresVangrad[translationNum].append(float(vangradContScores[0]))
                r2ExpScoresVangrad[translationNum].append(float(vangradContScores[1]))
                r3ExpScoresVangrad[translationNum].append(float(vangradContScores[2]))
                r4ExpScoresVangrad[translationNum].append(float(vangradContScores[3]))
                fExpScoresVangrad[translationNum].append(float(sum(vangradContScores)))

                shapContScores = [np.sum(shap_values[0,0][segments == 0]), np.sum(shap_values[0,0][segments == 1]),\
                np.sum(shap_values[0,0][segments == 2]), np.sum(shap_values[0,0][segments == 3])]
                r1ExpScoresSHAP[translationNum].append(float(shapContScores[0]))
                r2ExpScoresSHAP[translationNum].append(float(shapContScores[1]))
                r3ExpScoresSHAP[translationNum].append(float(shapContScores[2]))
                r4ExpScoresSHAP[translationNum].append(float(shapContScores[3]))
                fExpScoresSHAP[translationNum].append(float(sum(shapContScores)))
            testImgCount += 1
            print("Total Imgs: ", testImgCount)

            results_dict = {}
            results_dict["r1ExpScoresVangrad"] = r1ExpScoresVangrad
            results_dict["r1ExpScoresLIME"] = r1ExpScoresLIME
            results_dict["r1ExpScoresSHAP"] = r1ExpScoresSHAP
            results_dict["r2ExpScoresVangrad"] = r2ExpScoresVangrad
            results_dict["r2ExpScoresLIME"] = r2ExpScoresLIME
            results_dict["r2ExpScoresSHAP"] = r2ExpScoresSHAP
            results_dict["r3ExpScoresVangrad"] = r3ExpScoresVangrad
            results_dict["r3ExpScoresLIME"] = r3ExpScoresLIME
            results_dict["r3ExpScoresSHAP"] = r3ExpScoresSHAP
            results_dict["r4ExpScoresVangrad"] = r4ExpScoresVangrad
            results_dict["r4ExpScoresLIME"] = r4ExpScoresLIME
            results_dict["r4ExpScoresSHAP"] = r4ExpScoresSHAP
            results_dict["fExpScoresVangrad"] = fExpScoresVangrad
            results_dict["fExpScoresLIME"] = fExpScoresLIME
            results_dict["fExpScoresSHAP"] = fExpScoresSHAP
            with open("continuity_eval_results.pkl", 'wb') as f:
                pickle.dump(results_dict, f)

### This is because we have chosen VITSTR's LIME-based segmentation which will
### will be used for other STR networks
def acquireLIMESegmentationOnly(opt):
    ### Extra variables
    shapImgLs = []
    costSamples = []
    shapValueLs = []
    maxTestSamples = 10000 ### So that DeepSHAP will not give CPU error
    middleMaskThreshold = 0.0005
    # maxImgPrintout = 50 ### Maximum number for each hit and nothit prediction shap printout
    testImgIdx = 1### 1 to len(testList)
    maxTrainCollect = 1
    maxImgBG = 15
    imgOutWidth = 224
    imgOutHeight = 224
    gridPixelsX = 28 ### There is a 14X14 pixel amount to be covered for each rectangular grid
    gridPixelsY = 28
    explainabilityDirs = "./explainability/SHAP/"
    heapSHAPOutdir = "./metrics/selectivitySHAP/"
    currentFeaturesToRemove = 1 ### Increment this value every iteration after acquiring accuracy
    ### E.G. numFeaturesAccDict[1] = 0.78 means that 1 feature removed has whole test dataset accuracy of 0.78
    numFeaturesAccDict = {}
    numFeaturesConfDict = {} ### Same as accuracy dictionary above, except here you store the avearge confidence
    # maxHeapList = []

    start_time = time.time()

    if not os.path.exists(explainabilityDirs):
        os.makedirs(explainabilityDirs)

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

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        print("opt.saved_model: ", opt.saved_model)
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
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

    if opt.fast_acc:
    # # To easily compute the total accuracy of our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']
    else:
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                          'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    ### Only IIIT5k_3000
    # if opt.fast_acc:
    # # # To easily compute the total accuracy of our paper.
    #     eval_data_list = ['IIIT5k_3000']
    # else:
    #     # The evaluation datasets, dataset order is same with Table 1 in our paper.
    #     eval_data_list = ['IIIT5k_3000']

    if opt.calculate_infer_time:
        evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
    else:
        evaluation_batch_size = opt.batch_size

    testImgCount = 0
    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        for i, (orig_img_tensors, labels) in enumerate(evaluation_loader):
            # print("orig_img_tensors shape: ", orig_img_tensors.shape) # (1, 1, 224, 224)
            gt = str(labels[0])
            model_obj.setGTText(gt)
            # Acquire trainable imagenumpy
            image_tensors = torch.clone(orig_img_tensors)
            detachedImg = image_tensors.cpu().detach().numpy()[0]
            inputImg = np.array([detachedImg])
            inputImg = torch.from_numpy(inputImg).to(device)
            # print("inputImg shape: ", inputImg.shape) # (1,1,224,224)
            img_numpy = detachedImg.squeeze(0).astype('float64')
            img_numpy = gray2rgb(img_numpy)
            # print("img_numpy shape: ", img_numpy.shape) # (224,224,3)
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(img_numpy, inputImg,
                                                     super_pixel_model, # classification function
                                                     top_labels=1,
                                                     hide_color=0,
                                                     num_samples=10,
                                                     batch_size=1) # number of images that will be sent to classification function
            ind =  explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            # dict_heatmap --> {0: -0.2424142, 1: 0.92214}, where 0, 1 is the segm number and the values are the LIME scores
            # Need to acquire segment keys. Here, we reverse because we will hiding the largest to smallest value
            sortedKeys = [k for k, v in sorted(dict_heatmap.items(), key=lambda item: item[1])]
            sortedKeys = sortedKeys[::-1] ### A list that should contain largest to smallest score
            segments = explanation.segments ### (224,224)

            results_dict = {}
            results_dict["lime_segm"] = segments

            with open("./segmdataset/vitstr_lime_segm_{}.pkl".format(testImgCount), 'wb') as f:
                pickle.dump(results_dict, f)

            testImgCount += 1
            print("testImgCount: ", testImgCount)

### Continuity
def postprocess_continuity(opt):
    with open("/home/markytools/Documents/MSEEThesis/STR/deep-text-recognition-benchmark-deepshap/continuity_eval_results.pkl", 'rb') as f:
        continuity_data = pickle.load(f)

    halfIntervals=16 ### Translate to the left side 16 times, and to the right side 16 times
    # 33 datapoints for x-axis (16translations left, 1 normal, 16translations right)
    # 288 CUTE80 images
    # Collect all from leftmost translations to center(exluding)
    keystring = ["r1ExpScoresLIME", "r2ExpScoresLIME", "r3ExpScoresLIME", "r4ExpScoresLIME", "fExpScoresLIME"]
    xLabelStr = ["R1", "R2", "R3", "R4", "F"]
    for kIdx, kName in enumerate(keystring):
        expScores = []
        for x in range(1, halfIntervals+1):
            actualIdx = halfIntervals-x
            expScores.append(sum(continuity_data[kName][actualIdx]))
        # Collect on the center only
        expScores.append(sum(continuity_data[kName][0]))
        # Collect all from center(exluding) translations to rightmost
        for x in range(halfIntervals+1, halfIntervals+halfIntervals+1):
            expScores.append(sum(continuity_data[kName][x]))
        expScores = [x / sum(expScores) for x in expScores] ### Normalize
        xAxisData = [xAxisVal for xAxisVal in range(-halfIntervals, halfIntervals+1)]
        plt.plot(xAxisData, expScores, label=xLabelStr[kIdx])
    plt.legend(loc="upper left")
    plt.title('Lime Continuity')
    plt.xlabel('TranslationX')
    plt.ylabel('LimeScore')
    plt.savefig('limeContinuity.png')
    plt.clf()

    # Vanilla Gradients
    keystring = ["r1ExpScoresVangrad", "r2ExpScoresVangrad", "r3ExpScoresVangrad", "r4ExpScoresVangrad", "fExpScoresVangrad"]
    xLabelStr = ["R1", "R2", "R3", "R4", "F"]
    for kIdx, kName in enumerate(keystring):
        expScores = []
        for x in range(1, halfIntervals+1):
            actualIdx = halfIntervals-x
            expScores.append(sum(continuity_data[kName][actualIdx]))
        # Collect on the center only
        expScores.append(sum(continuity_data[kName][0]))
        # Collect all from center(exluding) translations to rightmost
        for x in range(halfIntervals+1, halfIntervals+halfIntervals+1):
            expScores.append(sum(continuity_data[kName][x]))
        expScores = [x / sum(expScores) for x in expScores] ### Normalize
        xAxisData = [xAxisVal for xAxisVal in range(-halfIntervals, halfIntervals+1)]
        plt.plot(xAxisData, expScores, label=xLabelStr[kIdx])
    plt.legend(loc="upper left")
    plt.title('VanillaGradients Continuity')
    plt.xlabel('TranslationX')
    plt.ylabel('VanGrad')
    plt.savefig('vangradContinuity.png')
    plt.clf()

    # SHAP
    keystring = ["r1ExpScoresSHAP", "r2ExpScoresSHAP", "r3ExpScoresSHAP", "r4ExpScoresSHAP", "fExpScoresSHAP"]
    xLabelStr = ["R1", "R2", "R3", "R4", "F"]
    for kIdx, kName in enumerate(keystring):
        expScores = []
        for x in range(1, halfIntervals+1):
            actualIdx = halfIntervals-x
            expScores.append(sum(continuity_data[kName][actualIdx]))
        # Collect on the center only
        expScores.append(sum(continuity_data[kName][0]))
        # Collect all from center(exluding) translations to rightmost
        for x in range(halfIntervals+1, halfIntervals+halfIntervals+1):
            expScores.append(sum(continuity_data[kName][x]))
        expScores = [x / sum(expScores) for x in expScores] ### Normalize
        xAxisData = [xAxisVal for xAxisVal in range(-halfIntervals, halfIntervals+1)]
        plt.plot(xAxisData, expScores, label=xLabelStr[kIdx])
    plt.legend(loc="upper left")
    plt.title('SHAP Continuity')
    plt.xlabel('TranslationX')
    plt.ylabel('SHAP')
    plt.savefig('shapContinuity.png')
    plt.clf()

### testing for smoothgrad, guided backprop, etc.
def testOtherAttrMethods(opt):
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

    if validators.url(opt.saved_model):
        print("opt.saved_model: ", opt.saved_model)
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])

    """ evaluation """
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
    super_pixel_model.zero_grad()

    sample_img = cv2.imread("/home/markytools/Documents/MSEEThesis/STR/datasets/synthtigerseg/results/images/0/0.jpg", cv2.IMREAD_GRAYSCALE)
    # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    sample_img = cv2.resize(sample_img, (224, 224))
    # print("sample_img shape: ", sample_img.shape) (224,224)
    # print("sample_img max: ", sample_img.max()) 255
    # print("sample_img min: ", sample_img.min()) 0
    sample_img = sample_img / 255.
    inputImgTensor = torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
    inputImgTensor = Variable(inputImgTensor, requires_grad=True)

    guided_grad = GuidedBackpropGrad(pretrained_model=super_pixel_model, cuda=True)
    guided_saliency = guided_grad(inputImgTensor, index=0)
    # print("guided_saliency shape: ", guided_saliency.shape)
    # print("guided_saliency max(): ", guided_saliency.max())
    # print("guided_saliency min(): ", guided_saliency.min())
    save_as_gray_image(guided_saliency, os.path.join("./attrMethResults/", 'guided_grad.jpg'))
    print('Saved guided backprop gradient image')

if __name__ == '__main__':
    # deleteInf()
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    acquire_selectivity_auc(opt)
