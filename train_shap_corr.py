import os
import time
import string
import argparse
import re
import validators
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance
import pickle

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model, STRScore
from utils import get_args, AccuracyMeter
import matplotlib.pyplot as plt
import settings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getPredAndConf(opt, model, scoring, image, converter, labels):
    batch_size = image.size(0)
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    if not opt.Transformer:
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

    if settings.MODEL=="vitstr":
        target = converter.encode(labels)
        preds = model(image, text=target, seqlen=converter.batch_max_length)

        confScore = scoring(preds)
        confScore = confScore.detach().cpu().numpy()

        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)

        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)
        preds_str = preds_str[0]
        preds_str = preds_str[:preds_str.find('[s]')]

    elif settings.MODEL=="trba":
        preds = model(image)
        confScore = scoring(preds)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        # print("preds_str: ", preds_str) # ['ronaldo[s]
        preds_str = preds_str[0]
        preds_str = preds_str[:preds_str.find('[s]')]

    elif settings.MODEL=="srn":
        target = converter.encode(labels)
        preds = model(image, None)

        _, preds_index = preds[2].max(2)

        confScore = scoring(preds)
        confScore = confScore.detach().cpu().numpy()

        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        # length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index, length_for_pred)
        preds_str = preds_str[0]
        # preds_str = preds_str[:preds_str.find('[s]')]
        preds = preds[2]

    elif settings.MODEL=="parseq":
        target = converter.encode(labels)
        preds = model(image)

        predStr, confidence = model.tokenizer.decode(preds)

        confScore = scoring(preds)
        confScore = confScore.detach().cpu().numpy()

        # _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        # preds_index = preds_index.view(-1, converter.batch_max_length)
        #
        # length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        # preds_str = converter.decode(preds_index[:, 0:], length_for_pred)
        preds_str = predStr[0]
        # preds_str = preds_str[:preds_str.find('[s]')]
        # pred = pred[:pred_EOS]
    return preds_str, confScore
