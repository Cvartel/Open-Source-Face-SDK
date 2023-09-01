import catboost as cb
import torch
from torch import nn

from .bn_feature_extractor import BNFeatureExtractor


class QualityEstimatorPipeline(nn.Module):
    # Used to combine face recognition model, extract it's features and estimate quality.
    def __init__(self, backbone, regressor, device="cpu"):
        super(QualityEstimatorPipeline, self).__init__()
        self.backbone = BNFeatureExtractor(backbone, device=device)
        self.backbone.eval()
        self.regressor = regressor
        self.device = device

    def forward(self, x, return_emb=False):
        emb, features = self.backbone(x)
        quality = self.regressor.predict(features.detach().cpu().numpy())
        quality = torch.Tensor(quality)
        if return_emb:
            return emb, quality
        else:
            return quality
