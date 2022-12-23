import logging
import torch.nn as nn
from fastai.vision import *

from modules_matrn.attention import *
from modules_matrn.backbone import ResTranformer, ResNetWithPosEnc
from modules_matrn.model import Model


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        self.num_more_attention = ifnone(config.model_vision_num_more_attention, -1)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTranformer(config)
        else: self.backbone = ResNetWithPosEnc(config)

        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8*32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

        if self.num_more_attention > 0:
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.more_attention = nn.ModuleList([
                PositionAttention(
                    max_length=config.dataset_max_length + 1,  # additional stop token
                    mode=mode,
                ) for _ in range(self.num_more_attention)
            ])


    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)

        if self.num_more_attention > 0:
            for attn in self.more_attention:
                attn_vecs, attn_scores = attn(features, attn_vecs)

        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight':self.loss_weight, 'name': 'vision',
                'backbone_feature': features}
