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

from modules_srn.transformation import TPS_SpatialTransformerNetwork
from modules_srn.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules_srn.sequence_modeling import BidirectionalLSTM
from modules_srn.prediction import Attention
from modules_srn.resnet_aster import ResNet_ASTER

from modules_srn.bert import Bert_Ocr
from modules_srn.bert import Config

from modules_srn.SRN_modules import Transforme_Encoder, SRN_Decoder, Torch_transformer_encoder
from modules_srn.resnet_fpn import ResNet_FPN
import settings
import sys

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
        preds = preds[2] # Access second index
        bs = preds.shape[0]
        # text_for_loss, length_for_loss = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        text_for_loss_length = self.opt.batch_max_length + 1

        # _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        # preds_index = preds_index.view(-1, self.converter.batch_max_length)
        # print("preds shape: ", preds.shape)
        # print("preds_index: ", preds_index)
        # preds_str = self.converter.decode(preds_index, length_for_pred)
        if settings.MODEL == 'vitstr':
            preds_str = self.converter.decode(preds_index[:, 1:], length_for_pred)
        elif settings.MODEL == 'srn':
            _, preds_index = preds.max(2)
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * bs).to(self.device)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            # sys.exit()
        elif settings.MODEL == 'parseq':
            preds_str, confidence = self.model.tokenizer.decode(preds)

        # Confidence score
        # ARGMAX calculation
        sum = torch.FloatTensor([0]*bs).to(self.device)
        if self.opt.confidence_mode == 0:
            preds_prob = F.softmax(preds, dim=2)
            # preds_prob shape:  torch.Size([1, 25, 96])
            preds_max_prob, preds_max_idx = preds_prob.max(dim=2)
            # preds_max_prob shape:  torch.Size([1, 25])
            confidence_score_list = []
            count = 0
            for one_hot_preds, pred, pred_max_prob in zip(preds_prob, preds_str, preds_max_prob):
                if settings.MODEL == 'vitstr' or settings.MODEL == 'srn':
                    if self.enableSingleCharAttrAve:
                        one_hot = one_hot_preds[self.singleChar, :]
                        # pred = pred[self.singleChar]
                        pred_max_prob = pred_max_prob[self.singleChar]
                    else:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # if pred_max_prob.shape[0] == 0: continue
                    if self.enableSingleCharAttrAve:
                        sum = one_hot
                        # sum shape:  torch.Size([96])
                        sum = sum.unsqueeze(0)
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
                        sum = one_hot
                        # sum shape:  torch.Size([96])
                        sum = sum.unsqueeze(0)
                    else:
                        if self.opt.scorer == "cumprod":
                            confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                            sum[count] += confidence_score
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
        return sum

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
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
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        elif opt.FeatureExtraction == 'AsterRes':
            self.FeatureExtraction = ResNet_ASTER(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResnetFpn':
            self.FeatureExtraction = ResNet_FPN()
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512


        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        elif opt.SequenceModeling == 'Bert':
            cfg = Config()
            cfg.dim = opt.output_channel; cfg.dim_c = opt.output_channel              # 降维减少计算量
            cfg.p_dim = opt.position_dim                        # 一张图片cnn编码之后的特征序列长度
            cfg.max_vocab_size = opt.batch_max_length + 1                # 一张图片中最多的文字个数, +1 for EOS
            cfg.len_alphabet = opt.alphabet_size                # 文字的类别个数
            self.SequenceModeling = Bert_Ocr(cfg)
        elif opt.SequenceModeling == 'SRN':
            self.SequenceModeling = Transforme_Encoder(n_layers=2, n_position=opt.position_dim)
            # self.SequenceModeling = Torch_transformer_encoder(n_layers=2, n_position=opt.position_dim)
            self.SequenceModeling_output = 512
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        elif opt.Prediction == 'Bert_pred':
            pass
        elif opt.Prediction == 'SRN':
            self.Prediction = SRN_Decoder(n_position=opt.position_dim, N_max_character=opt.batch_max_character + 1, n_class=opt.alphabet_size)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text=None, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)


        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        # if self.stages['Feat'] == 'AsterRes' or self.stages['Feat'] == 'ResnetFpn':
        if self.stages['Feat'] == 'AsterRes' or self.stages['Feat'] == 'ResnetFpn':
            b, c, h, w = visual_feature.shape
            visual_feature = visual_feature.permute(0, 1, 3, 2)
            visual_feature = visual_feature.contiguous().view(b, c, -1)
            visual_feature = visual_feature.permute(0, 2, 1)    # batch, seq, feature
        else:
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)


        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        elif self.stages['Seq'] == 'Bert':
            pad_mask = text
            contextual_feature = self.SequenceModeling(visual_feature, pad_mask)
        elif self.stages['Seq'] == 'SRN':
            contextual_feature = self.SequenceModeling(visual_feature, src_mask=None)[0]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM


        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        elif self.stages['Pred'] == 'Bert_pred':
            prediction = contextual_feature
        elif self.stages['Pred'] == 'SRN':
            prediction = self.Prediction(contextual_feature)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
