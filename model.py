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

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.vitstr import create_vitstr

import math
import sys
import settings

# singleChar - if -1 then STRScore outputs all char, however if
# 0 - N, then it will output the single character confidence of the index 0 to N
class STRScore(nn.Module):
    def __init__(self, opt, converter, device, gtStr="", enableSingleCharAttrAve=False, model=None):
        super(STRScore, self).__init__()
        self.enableSingleCharAttrAve = enableSingleCharAttrAve
        self.singleChar = -1
        self.opt = opt
        self.converter = converter
        self.device = device
        self.gtStr = gtStr
        self.model = model # Pass here if you want to use it
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
        elif self.opt.Transformer:
            # preds_index = preds_index.view(-1, self.converter.batch_max_length)
            # print("preds shape: ", preds.shape)
            # print("preds_index: ", preds_index)
            # preds_str = self.converter.decode(preds_index, length_for_pred)
            if settings.MODEL == 'vitstr':
                _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
                preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)
            elif settings.MODEL == 'parseq':
                preds_str, confidence = self.model.tokenizer.decode(preds)
            # print("preds_str: ", preds_str)
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
            # preds_prob shape:  torch.Size([1, 25, 96])
            preds_max_prob, preds_max_idx = preds_prob.max(dim=2)
            # preds_max_prob shape:  torch.Size([1, 25])
            confidence_score_list = []
            count = 0
            for one_hot_preds, pred, pred_max_prob in zip(preds_prob, preds_str, preds_max_prob):
                if self.opt.Transformer:
                    if settings.MODEL == 'vitstr':
                        if self.enableSingleCharAttrAve:
                            one_hot = one_hot_preds[self.singleChar, :]
                            pred = pred[self.singleChar]
                            pred_max_prob = pred_max_prob[self.singleChar]
                        else:
                            pred_EOS = pred.find('[s]')
                            pred = pred[:pred_EOS]
                            pred_max_prob = pred_max_prob[:pred_EOS]

                        # if pred_max_prob.shape[0] == 0: continue
                        if self.enableSingleCharAttrAve:
                            sum[count] = one_hot
                            # sum = one_hot
                            # sum shape:  torch.Size([96])
                            # sum = sum.unsqueeze(0)
                        else:
                            if self.opt.scorer == "cumprod":
                                confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                                sum[count] += confidence_score
                            elif self.opt.scorer == "mean":
                                confidence_score = torch.mean(pred_max_prob) ### Maximum is 1
                                sum[count] += confidence_score
                            sum = sum.unsqueeze(1)

                    elif settings.MODEL == 'parseq':
                        if self.enableSingleCharAttrAve:
                            one_hot = one_hot_preds[self.singleChar, :]
                            # pred = pred[self.singleChar]
                            pred_max_prob = pred_max_prob[self.singleChar]
                        else:
                            pred_EOS = len(pred) # Predition string already has no EOS, fully intact
                            pred_max_prob = pred_max_prob[:pred_EOS]

                        # if pred_max_prob.shape[0] == 0: continue
                        if self.enableSingleCharAttrAve:
                            sum[count] = one_hot
                            # sum shape:  torch.Size([96])
                            # sum = sum.unsqueeze(0)
                        else:
                            if self.opt.scorer == "cumprod":
                                confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                                sum[count] += confidence_score
                            elif self.opt.scorer == "mean":
                                confidence_score = torch.mean(pred_max_prob) ### Maximum is 1
                                sum[count] += confidence_score
                            sum = sum.unsqueeze(1)
                elif 'Attn' in self.opt.Prediction:
                    # if pred_max_prob.shape[0] == 0: continue
                    if self.enableSingleCharAttrAve:
                        one_hot = one_hot_preds[self.singleChar, :]
                        sum[count] = one_hot
                    else:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS] ### Use score of all letters
                        # pred_max_prob = pred_max_prob[0:1] ### Use score of first letter only
                        if pred_max_prob.shape[0] == 0: continue
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                        sum[count] += confidence_score
                        sum = sum.unsqueeze(1)
                elif 'CTC' in self.opt.Prediction:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
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
        return sum

class Model(nn.Module):

    def __init__(self, opt, device=None, converter=None, gt_text=""):
        super(Model, self).__init__()
        self.opt = opt
        self.device = device
        self.converter = converter
        self.gt_text = gt_text
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction,
                       'ViTSTR': opt.Transformer}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        if opt.Transformer:
            self.vitstr = create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)
            return

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
    def set_labels(self, labels):
        self.labels = labels
    def patch_embed_func(self):
        if self.opt.Transformer:
            return self.vitstr.patch_embed_func()
        return None
    def setGTText(self, text):
        self.gt_text = text
    def forward(self, input, text="", seqlen=25, is_train=False):
        # text = torch.FloatTensor(input.shape[0], self.opt.batch_max_length + 1).fill_(0).to(self.device)
        # text = self.converter.encode(self.labels)
        if settings.MODEL == 'trba':
            text = self.gt_text
        if not self.stages['ViTSTR']:
            assert(len(text)>0)
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        if self.stages['ViTSTR']:
            prediction = self.vitstr(input, seqlen=seqlen)
            return prediction

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
