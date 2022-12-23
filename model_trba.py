"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules_trba.transformation import TPS_SpatialTransformerNetwork
from modules_trba.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules_trba.sequence_modeling import BidirectionalLSTM
from modules_trba.prediction import Attention
import numpy as np
import torch
import torch.nn.functional as F
import random
import copy
# from torch_edit_distance import levenshtein_distance

class STRScore(nn.Module):
    def __init__(self, opt, converter, device, gtStr="", enableSingleCharAttrAve=False):
        super(STRScore, self).__init__()
        self.opt = opt
        self.converter = converter
        self.device = device
        self.gtStr = gtStr
        self.enableSingleCharAttrAve = enableSingleCharAttrAve
        self.blank = torch.tensor([-1], dtype=torch.float).to(self.device)
        self.separator = torch.tensor([-2], dtype=torch.float).to(self.device)

    # singleChar - if >=0, then the output of STRScore will only be a single character
    # instead of a whole. The char index will be equal to the parameter "singleChar".
    def setSingleCharOutput(self, singleChar):
        self.singleChar = singleChar

    def forward(self, preds):
        bs = preds.shape[0]
        # text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        text_for_loss_length = self.opt.batch_max_length + 1
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * bs).to(self.device)
        if 'CTC' in self.opt.Prediction:
            # Calculate evaluation loss for CTC decoder.
            preds_size = torch.FloatTensor([preds.size(1)] * bs)
            if self.opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            # print("preds_index shape: ", preds_index.shape)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)
            # preds_str = self.converter.decode(preds_index, length_for_pred)
            preds = preds.log_softmax(2).permute(1, 0, 2)
        else:
            preds = preds[:, :text_for_loss_length, :]

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            # print("preds shape: ", preds.shape)
            # print("preds_index: ", preds_index)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            # print("preds_str: ", preds_str)

        # Confidence score
        # ARGMAX calculation
        sum = torch.FloatTensor([0]*bs).to(self.device)
        if self.enableSingleCharAttrAve:
            sum = torch.zeros((bs, preds.shape[2])).to(self.device)
        if self.opt.confidence_mode == 0:
            preds_prob = F.softmax(preds, dim=2)
            # print("preds_prob shape: ", preds_prob.shape)
            preds_max_prob, _ = preds_prob.max(dim=2)
            # print("preds_max_prob shape: ", preds_max_prob.shape)
            confidence_score_list = []
            count = 0
            for one_hot_preds, pred, pred_max_prob in zip(preds_prob, preds_str, preds_max_prob):
                if 'Attn' in self.opt.Prediction:
                    if self.enableSingleCharAttrAve:
                        one_hot = one_hot_preds[self.singleChar, :]
                        sum[count] = one_hot
                    else:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS] ### Use score of all letters
                        # pred_max_prob = pred_max_prob[0:1] ### Use score of first letter only
                        if pred_max_prob.shape[0] == 0: continue
                        if self.opt.scorer == "cumprod":
                            confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                        elif self.opt.scorer == "mean":
                            confidence_score = torch.mean(pred_max_prob) ### Maximum is 1
                        sum[count] += confidence_score
                        sum = sum.unsqueeze(1)
                elif 'CTC' in self.opt.Prediction:
                    if self.opt.scorer == "cumprod":
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                    elif self.opt.scorer == "mean":
                        confidence_score = torch.mean(pred_max_prob) ### Maximum is 1
                    sum[count] += confidence_score
                    sum = sum.unsqueeze(1)
                count += 1
                # return sum.detach().cpu().numpy()
                # print("sumshape: ", sum.shape)
        elif self.opt.confidence_mode == 1:
            preds_prob = F.softmax(preds, dim=2)
            ### Predicted indices
            preds_max_prob = torch.argmax(preds_prob, 2)
            # print("preds_max_prob shape: ", preds_max_prob.shape)
            ### Ground truth indices
            gtIndices, _ = self.converter.encode([self.gtStr for i in range(0,preds_prob.shape[0])], batch_max_length=self.opt.batch_max_length-1)
            # print("gtIndices shape: ", gtIndices.shape)
            ### Acquire levenstein distance
            m = torch.tensor([preds_prob.shape[1] for i in range(0, gtIndices.shape[0])], dtype=torch.float).to(self.device)
            n = torch.tensor([preds_prob.shape[1] for i in range(0, gtIndices.shape[0])], dtype=torch.float).to(self.device)
            # print("m: ", m)
            # print("preds_max_prob dtype: ", preds_max_prob.dtype)
            # print("gtIndices dtype: ", gtIndices.dtype)
            preds_max_prob = preds_max_prob.type(torch.float)
            gtIndices = gtIndices.type(torch.float)
            r = levenshtein_distance(preds_max_prob.to(self.device), gtIndices.to(self.device), n, m, torch.cat([self.blank, self.separator]), torch.empty([], dtype=torch.float).to(self.device))
            # print("r shape: ", r.shape)
            # confidence_score_list = []
            # count = 0
            # for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            #     if 'Attn' in self.opt.Prediction:
            #         pred_EOS = pred.find('[s]')
            #         pred = pred[:pred_EOS]
            #         pred_max_prob = pred_max_prob[:pred_EOS] ### Use score of all letters
            #         # pred_max_prob = pred_max_prob[0:1] ### Use score of first letter only
            #         if pred_max_prob.shape[0] == 0: continue
            #         confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            #         sum[count] += confidence_score
            #     count += 1
            # return sum.detach().cpu().numpy()
            # print("sumshape: ", sum.shape)
            # sum = sum.unsqueeze(1)
            rSoft = F.softmax(r[:,2].type(torch.float))
            # rSoft = rSoft.contiguous()
            rNorm = rSoft.max()-rSoft
            sum = rNorm.unsqueeze(1)
            print("sum shape: ", sum.shape)
        return sum

class SuperPixler(nn.Module):
  def __init__(self, n_super_pixel, imageList, super_pixel_width, super_pixel_height, opt):
    super(SuperPixler, self).__init__()
    self.opt = opt
    self.imageList = imageList
    self.n_super_pixel = n_super_pixel
    # self.image = image
    # self.image = image.transpose(2, 0, 1) # model expects images in BRG, not RGB, so transpose color channels
    # self.mean_color = self.image.mean()
    # self.image = np.expand_dims(self.image, axis=0)
    self.super_pixel_width = super_pixel_width
    self.super_pixel_height = super_pixel_height
  # def setImage(self, image):
  #     self.image = image
  #     self.image_height = image.shape[2]
  #     self.image_width = image.shape[3]
  def sampleImages(self, size):
        newImgList = []
        for i in range(0, size):
            randIdx = random.randint(0, len(self.imageList)-1)
            newImgList.append(copy.deepcopy(self.imageList[randIdx]))
        return np.array(newImgList)
  def forward(self, x):
    """
    In the forward step we accept the super pixel masks and transform them to a batch of images
    """
    # x = self.sampleMasks(image.shape[0])
    image = self.sampleImages(x.shape[0])
    self.image = image
    self.image_height = image.shape[2]
    self.image_width = image.shape[3]
    self.mean_color = self.image.mean()
    # self.mean_color = self.image.mean(axis=(1,2,3))
    # pixeled_image = np.repeat(self.image.copy(), x.shape[0], axis=0)# WARNING:
    pixeled_image = self.image.copy()
    # print("pixeled_image shape: ", pixeled_image.shape)
    # print("x shape: ", x.shape)
    for i, super_pixel in enumerate(x.T):
        images_to_pixelate = [bool(p) for p in super_pixel]
        # print("super_pixel shape: ", super_pixel.shape)
        # print("images_to_pixelate len: ", len(images_to_pixelate))
        # print("i: {}, superPix: {}, images_to_pixelate: {}".format(i, super_pixel, images_to_pixelate))
        x = (i*self.super_pixel_height//self.image_height)*self.super_pixel_width
        y = i*self.super_pixel_height%self.image_height
        ### Reshape image means since it has n-dim size, not a single number. Need to repeat sideways.
        # origShapeToApply = pixeled_image[images_to_pixelate,:,y:y+self.super_pixel_height,x:x+self.super_pixel_width].shape
        # print("origShapeToApply: ", origShapeToApply)
        # mean_color_spec = np.tile(self.mean_color, origShapeToApply[1:]) #
        # mean_color_spec = np.reshape(mean_color_spec, origShapeToApply[::-1]).T ### reshape to reversed
        ### Apply image means
        pixeled_image[images_to_pixelate,:,y:y+self.super_pixel_height,x:x+self.super_pixel_width] = self.mean_color
    return pixeled_image

class CastNumpy(nn.Module):
    def __init__(self, device):
        super(CastNumpy, self).__init__()
        self.device = device

    def forward(self, image):
        """
        In the forward function we accept the inputs and cast them to a pytorch tensor
        """

        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        if image.ndimension() == 3:
          image = image.unsqueeze(0)
        image_half = image.half()
        return image_half.float()

class Model(nn.Module):

    def __init__(self, opt, device, feature_ext_outputs=None):
        super(Model, self).__init__()
        self.opt = opt
        self.device = device
        self.gtText = None
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

        ### Set feature map outputter modules

        if opt.output_feat_maps:
            feature_ext_outputs.set_feature_ext(self.FeatureExtraction)
            ### Define hooks
            feature_ext_outputs = feature_ext_outputs
            totalCNNLayers = 0
            idxToOutput = []
            layersList = []

            layerCount = 0
            # print("list(self.FeatureExtraction._modules.items()): ", list(self.FeatureExtraction._modules.items()))
            # print("list(self.FeatureExtraction.ConvNet_modules.items())[0][1]: ", list(self.FeatureExtraction.ConvNet._modules.items())[0][1])
            first_layer = list(self.FeatureExtraction.ConvNet._modules.items())[0][1]
            first_layer.register_backward_hook(feature_ext_outputs.append_first_grads)
            for layer in self.FeatureExtraction.modules():
                if isinstance(layer, nn.Conv2d):
                    layerCount += 1
                    if layerCount >= opt.min_layer_out and layerCount <= opt.max_layer_out:
                        layer.register_forward_hook(feature_ext_outputs.append_layer_out)
                        layer.register_backward_hook(feature_ext_outputs.append_grad_out)
    # def get_feature_ext(self):
    #     return self.FeatureExtraction
    def setGTText(self, text):
        self.gtText = text
    def forward(self, input, text="", is_train=True):
        if self.opt.is_shap:
            text = torch.LongTensor(input.shape[0], self.opt.batch_max_length + 1).fill_(0).to(self.device)
        elif self.gtText is not None:
            text = self.gtText
        else:
            text = torch.LongTensor(input.shape[0], self.opt.batch_max_length + 1).fill_(0).to(self.device)
            # print("text shape: ", text.shape) (1,26)
        tpsOut = input.contiguous()
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            tpsOut = self.Transformation(tpsOut)
        # print("Transformation feature shape: ", input.shape)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(tpsOut)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        # print("visual feature shape: ", visual_feature.shape)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
        # print("Sequence feature shape: ", contextual_feature.shape)

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        # print("prediction feature shape: ", prediction.shape)
        # return prediction, tpsOut
        return prediction
