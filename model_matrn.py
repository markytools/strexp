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

import sys
import math

class STRScore(nn.Module):
    def __init__(self, config, charsetMapper, postprocessFunc, device, enableSingleCharAttrAve=False):
        super(STRScore, self).__init__()
        self.config = config
        self.charsetMapper = charsetMapper
        self.postprocess = postprocessFunc
        self.device = device
        self.enableSingleCharAttrAve = enableSingleCharAttrAve

    # singleChar - if >=0, then the output of STRScore will only be a single character
    # instead of a whole. The char index will be equal to the parameter "singleChar".
    def setSingleCharOutput(self, singleChar):
        self.singleChar = singleChar

    ### Output of ABINET model
    ### Shape with 1 batchsize: torch.Size([1, 26, 37])
    def forward(self, preds):
        # Acquire predicted text
        pt_text, _, __ = self.postprocess(preds[0], self.charsetMapper, self.config.model_eval)
        preds = preds[0]["logits"]
        # preds shape:  torch.Size([50, 26, 37])
        # Confidence score
        bs = preds.shape[0]
        # ARGMAX calculation
        sum = torch.FloatTensor([0]*len(preds)).to(self.device)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, preds_max_index = preds_prob.max(dim=2)
        if self.enableSingleCharAttrAve:
            preds_max_prob = preds_max_prob[:,self.singleChar]
            preds_max_prob = preds_max_prob.unsqueeze(0)
        if self.enableSingleCharAttrAve:
            sum = torch.zeros((bs, len(self.config.character)-1)).to(self.device)
        # print("preds_max_prob shape: ", preds_max_prob.shape) (1,26)
        confidence_score_list = []
        count = 0
        for one_hot_preds, pred, pred_max_prob in zip(preds_prob, pt_text, preds_max_prob):
            if self.enableSingleCharAttrAve:
                one_hot = one_hot_preds[self.singleChar, :]
                sum[count] = one_hot
                # sum = sum.unsqueeze(0)
            else:
                pred_EOS = len(pred)
                # pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS] ### Use score of all letters excluding null char
                # pred_max_prob = pred_max_prob[0:1] ### Use score of first letter only
                if pred_max_prob.shape[0] == 0: continue
                if self.config.scorer == "cumprod":
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1] ### Maximum is 1
                elif self.config.scorer == "mean":
                    confidence_score = torch.mean(pred_max_prob) ### Maximum is 1
                sum[count] += confidence_score
            count += 1
        if self.enableSingleCharAttrAve:
            pass
        else:
            sum = sum.unsqueeze(1)
        return sum
