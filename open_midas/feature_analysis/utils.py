import os
import warnings
from types import MethodType

import cv2
import numpy as np
import pandas as pd
import torch
from recognition.models.iresnet_custom import iresnet18, iresnet200
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

QUANTILE_EJECT = 0.05

cos_sim = nn.CosineSimilarity()

labels_revert = {
    0.0: 0,  # 'real',
    1.0: 3,  # 'photo',
    2.0: 3,  # 'photo',
    3.0: 3,  # 'photo',
    4.0: 4,  # 'regions',
    5.0: 4,  # 'regions',
    6.0: 1,  # '2d_mask',
    7.0: 5,  # 'replay',
    8.0: 5,  # 'replay',
    9.0: 5,  # 'replay',
    10.0: 2,  # '3d_mask',
}

name_features = ["mean of channel ", "variance of channel", "L2 between mean of channel and bn running mean",
                 "L2 between variance of channel and bn running variance", "mean of channel variance",
                 "variance of channel variance",
                 "mean of channel variance after centering", "variance of channel variance after centering"]


def set_except_bn_eval(model, global_values, bn_ids=None):
    if bn_ids is None:
        bn_ids = list(range(len(list(model.modules()))))
    bn_id = 0
    for layer in model.modules():
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        if 'batchnorm' in str(layer.__class__):
            if bn_id in bn_ids:
                if 'BatchNorm1d' in str(layer) or 'BatchNorm2d' in str(layer):
                    layer.forward = MethodType(lambda a, b:
                                               save_bn_input_forward(a, b, global_values), layer)
                    layer.normal = torch.distributions.normal.Normal(layer.running_mean, layer.running_var + 1e-07)
                else:
                    raise
            else:
                if 'BatchNorm1d' in str(layer):
                    layer.forward = MethodType(lambda a, b: super(torch.nn.BatchNorm1d, a).forward(b), layer)
                elif 'BatchNorm2d' in str(layer):
                    layer.forward = MethodType(lambda a, b: super(torch.nn.BatchNorm2d, a).forward(b), layer)
                else:
                    raise
            bn_id += 1


def save_bn_input_forward(self, x, global_values):
    with torch.no_grad():
        global_values.append(x.detach())
    out = super(self.__class__, self).forward(x)
    return out


def get_running_vals(model):
    weights = []
    biases = []
    means = []
    vars = []

    for m in model.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') == 0:  # not BN
            means.append(m.running_mean.detach().flatten())
            vars.append(m.running_var.detach().flatten())
            weights.append(m.weight.detach().flatten())
            biases.append(m.bias.detach().flatten())

    return {'means': means, 'vars': vars, 'weights': weights, 'biases': biases}


@torch.no_grad()
def make_feature_tensor_v2(bn_outputs, running, batch, device="cuda"):
    """
    features: "mean of channel ", "variance of channel","L2 between mean of channel and bn running mean",
                                     "L2 between variance of channel and bn running variance",
    """
    with torch.no_grad():
        n_bn_layers = len(bn_outputs)
        _means = torch.zeros((n_bn_layers, batch), device=device)
        _vars = torch.zeros((n_bn_layers, batch), device=device)
        means_dist = torch.zeros((n_bn_layers, batch), device=device)
        vars_dist = torch.zeros((n_bn_layers, batch), device=device)

        for j, current_bn in enumerate(bn_outputs):
            n_dims = len(current_bn.shape)
            without_first_dim = tuple(range(n_dims))[1:]
            _vars[j, :], _means[j, :] = torch.var_mean(current_bn, dim=without_first_dim)

            if n_dims > 2:
                current_bn = current_bn.flatten(start_dim=2)
                cv, cm = torch.var_mean(current_bn, dim=-1)
            else:
                cv = torch.zeros(current_bn.shape, device=device)
                cm = current_bn

            running_means = torch.tensor(running['means'][j], device=device)
            running_vars = torch.tensor(running['vars'][j], device=device)

            vars_dist[j, :] = torch.norm(cv - running_vars, dim=1)
            means_dist[j, :] = torch.norm(cm - running_means, dim=1)

        features = np.concatenate((means_dist.detach().cpu().numpy(), vars_dist.detach().cpu().numpy(),
                                   _means.detach().cpu().numpy(), _vars.detach().cpu().numpy()), axis=0)

    return features.T


@torch.no_grad()
def make_feature_tensor_v1(bn_outputs, running, batch, device="cuda"):
    """
    Features:  "mean of channel variance", "variance of channel variance",
    "mean of channel variance after centering", "variance of channel variance after centering"
    """

    def get_var_mean_from_tensor(x):
        assert len(x.shape) >= 2
        if len(x.shape) > 2:
            flat_xy_dims = tuple(range(len(x.shape)))[2:]
            v = torch.var(x.detach(), dim=flat_xy_dims)
            return torch.var_mean(v, dim=1)
        else:
            return torch.var_mean(x, dim=1)

    n_bn_layers = len(bn_outputs)
    _means = torch.zeros((n_bn_layers, batch), device=device)
    _vars = torch.zeros((n_bn_layers, batch), device=device)
    means_dist = torch.zeros((n_bn_layers, batch), device=device)
    vars_dist = torch.zeros((n_bn_layers, batch), device=device)

    eps = 1e-6
    for j, current_bn in enumerate(bn_outputs):
        n_dims = len(current_bn.shape)
        _vars[j, :], _means[j, :] = get_var_mean_from_tensor(current_bn)

        running_means = torch.tensor(running['means'][j], device=device)
        running_vars = torch.tensor(running['vars'][j], device=device)

        if len(current_bn.shape) > 2:
            denom = torch.sqrt(running_vars) + eps
            out = current_bn - running_means[None, :, None, None]
            out /= denom[None, :, None, None]
        else:
            denom = torch.sqrt(running_vars) + eps
            out = current_bn - running_means
            out /= denom

        vars_dist[j, :], means_dist[j, :] = get_var_mean_from_tensor(out)

    features = np.concatenate((_means.detach().cpu().numpy(), _vars.detach().cpu().numpy(),
                               means_dist.detach().cpu().numpy(), vars_dist.detach().cpu().numpy()), axis=0)

    return features.T


class SimpleDataset(Dataset):
    def __init__(self, root_path):
        self.name_list = os.listdir(root_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.root = root_path

    def __getitem__(self, key):
        img = transforms.Resize(size=(112, 112))(self.transform(
            cv2.cvtColor(cv2.imread(self.root + '/' + self.name_list[key], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)))
        return img

    def __len__(self):
        return len(self.name_list)


class FeatureMaker():
    def __init__(self, model_type, backbone_path, device):
        self.device = device
        if model_type == 'r18':
            self.backbone = iresnet18().to(self.device)
        elif model_type == 'r200':
            self.backbone = iresnet200(conv_bias=True, disable_bn2_bn3=True).to(self.device)

        else:
            raise (ValueError)

        self.backbone.load_state_dict(torch.load(backbone_path, map_location=self.device), strict=False)
        self.backbone.eval()
        self.backbone.to(self.device)
        self.BN_OUTPUTS = []
        set_except_bn_eval(self.backbone, self.BN_OUTPUTS, None)
        self.RUNNING = get_running_vals(self.backbone)

    def forward(self, images):
        batch = images.shape[0]
        _ = self.backbone(images.to(self.device))
        feature1 = make_feature_tensor_v2(self.BN_OUTPUTS, self.RUNNING, batch, self.device)
        feature2 = make_feature_tensor_v1(self.BN_OUTPUTS, self.RUNNING, batch, self.device)
        self.BN_OUTPUTS.clear()
        bn_feature = np.hstack((feature1, feature2))
        return bn_feature  # _.detach().cpu().numpy()

    def __call__(self, x):
        return self.forward(x)


def make_and_save_features(n_batchnorms, save_path, dataset, feature_maker, batch=32):
    COLUMN_NAMES = [f"{k}_{i}" for k in
                    ["mean of channel ", "variance of channel", "L2 between mean of channel and bn running mean",
                     "L2 between variance of channel and bn running variance", "mean of channel variance",
                     "variance of channel variance",
                     "mean of channel variance after centering", "variance of channel variance after centering"] for i
                    in range(n_batchnorms)] + ["label"]

    dataloader_s = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    df = pd.DataFrame(columns=COLUMN_NAMES)
    df.to_csv(save_path, index=False)

    for num, batch in enumerate(tqdm(dataloader_s)):
        tensors, labels = batch
        features = feature_maker(tensors)
        dts = pd.DataFrame(np.hstack((features, np.array(labels)[:, None])))
        dts.columns = COLUMN_NAMES
        df = pd.concat([df, dts], ignore_index=True)
        if num % 100 == 0:
            df.to_csv(save_path, mode='a', index=False, header=False)
            df = None
    df.to_csv(save_path, mode='a', index=False, header=False)
