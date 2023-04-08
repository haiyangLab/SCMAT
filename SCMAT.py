#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 16:17
# @Author  : Chiancc

import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch.nn as nn
import torch
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from timm.models.layers import to_2tuple, DropPath, Mlp
from timm.models.mlp_mixer import MixerBlock
from torch.autograd import Variable

from timm.models.vision_transformer import Block
import torch.nn.functional as F
from timm.models.layers import PatchEmbed

from utils.myutils import parse_model_config
from utils.logger import *
import umap
from umap import UMAP


def gmm(n_clusters=28):
    model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=0)
    return model


def tsne(X):
    model = TSNE(n_components=2)
    return model.fit_transform(X)


def umap(X):
    model = UMAP(n_neighbors=30, min_dist=0.1)
    return model.fit_transform(X)
    # UMAP(n_neighbors=30, min_dist=0.1).fit_transform(X)


def create_modules(module_cfg):
    """
      Constructs module list of layer blocks from module configuration in module_cfg
      Args:
          module_cfg: module config file(.cfg)
      Returns:
      """
    hyperparams = module_cfg.pop(0)
    hyperparams.update({
        'batch_size': int(hyperparams['batch_size']),
        'optimizer': hyperparams.get('optimizer'),
        'input_feature_number': int(hyperparams['input_feature_number']),
        'learning_rate': float(hyperparams['learning_rate']),
        'latent_dim': int(hyperparams['latent_dim'])
    })

    module_list = nn.ModuleList()
    filters = [hyperparams['input_feature_number']]
    for module_i, module_def in enumerate(module_cfg):
        modules = nn.Sequential()
        if 'dense_block' in module_def['type']:
            batch_norm = int(module_def["batch_normalize"])
            output = int(module_def["output"])  # out features
            active = module_def.get("activation", "")

            modules.add_module(
                f"linear_{module_i}",
                nn.Linear(filters[-1], output)
            )
            if batch_norm:
                modules.add_module(f"batch_norm_{module_i}",
                                   nn.BatchNorm1d(output, momentum=0.1, eps=1e-5))
            if active == "gelu":
                modules.add_module(f"leaky_{module_i}", nn.GELU())
            elif active == "sigmoid":
                modules.add_module(f"mish_{module_i}", nn.Sigmoid())
        elif 'dividing' in module_def['type']:
            output = int(module_def["output"])
            modules.add_module(
                f"dividing_layer",
            )

        filters.append(output)
        module_list.append(modules)

    return hyperparams, module_list


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=256, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图片尺寸224*224
        patch_size = to_2tuple(patch_size)  # 下采样倍数，一个grid cell包含了16*16的图片信息
        self.img_size = img_size
        self.patch_size = patch_size
        # grid_size是经过 PatchEmbed 后的特征层的尺寸
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # (14, 14)
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # path个数 14*14=196
        self.flatten = flatten

        # stride=patch_size -->stride=1
        # 通过一个卷积，完成patchEmbed
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # 如果使用了norm层，如BatchNorm2d，将通道数传入，以进行归一化，否则进行恒等映射
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape  # batch,channels,height,weight
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        # proj: [B, C, H, W] -> [B, C, H,W] , [B,1,224,224]-> [B,256,14,14]
        # flatten: [B, C, H, W] -> [B, C, HW] , [B,256,14,14]-> [B,256,196]
        # transpose: [B, C, HW] -> [B, HW, C] , [B,256,196]-> [B,196,256]
        x = self.proj(x)  # BCHW
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC   N = HW/P**2 64,C,256
        x = self.norm(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, data_type, data_size, act_layer=None, train=True):
        super(ViTEncoder, self).__init__()
        self.patch_size = 16
        embed_dim = 256
        drop_rate = 0.01
        drop_path_rate = 0.5
        num_heads = 8
        depth = 6
        mlp_ratio = 4.
        attn_drop_rate = 0.5
        qkv_bias = True
        self.use_gpu = torch.cuda.is_available()

        self.train = train
        self.length = sum(data_size)
        patch_num = math.ceil(math.sqrt(self.length) / self.patch_size)
        self.img_size = patch_num * self.patch_size

        self.idx = np.arange(0, self.length)
        np.random.shuffle(self.idx)

        log.info(f"self.img_size: {self.img_size} X {self.img_size} = {self.img_size * self.img_size} ")
        log.info(f"latent_dim: {patch_num * patch_num}")
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=1,
        #                               embed_dim=embed_dim)

        # print("num_patches: ", self.patch_embed.num_patches)

        ########
        # self-attention
        #########
        # self.drop = nn.Dropout(p=drop_rate)
        #
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.encoder = nn.Sequential(*[
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
        #         attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)])

        # #######
        # # MLP-mixer
        # ########
        drop_rate = 0.
        drop_path_rate = 0.
        mlp_ratio = (0.5, 4.0)
        depth = 8
        self.encoder = nn.Sequential(*[
            MixerBlock(
                embed_dim, patch_num * patch_num, mlp_ratio, mlp_layer=Mlp, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.encoder2latent = nn.Linear(self.img_size * self.img_size, patch_num * patch_num)  # concat
        # self.encoder2latent = nn.Linear(embed_dim, 1)
        self.bn = nn.BatchNorm1d(patch_num * patch_num)
        self.act = nn.GELU()

    def data_process(self, x):
        x = torch.cat(x, dim=1)
        # self.idx = torch.randperm(x.shape[0])
        # x = x[:, self.idx]  # 打乱

        batch_size, feature_size = x.shape
        padding_token = torch.zeros(abs(self.img_size * self.img_size - feature_size)).repeat(batch_size, 1)
        if self.use_gpu:
            padding_token = padding_token.cuda()

        res = torch.cat([x, padding_token], dim=1)

        # if patch_embed
        # res = res.reshape(batch_size, 1, self.img_size, self.img_size)
        res = res.reshape(batch_size, -1, self.patch_size * self.patch_size)
        return res

    def forward(self, x):
        x = self.data_process(x)  # BCHW
        # log.info(f"input_x:{input_x.shape}")

        x = self.encoder(x)  # 64,196,256 BN(p*p*c)
        x = self.norm(x)
        # print("transformer:", x.shape)

        x = x.reshape(x.shape[0], -1)
        x = self.encoder2latent(x)  # 64, 196

        x = self.bn(x)
        x = self.act(x)

        return x


class ViTDecoder(nn.Module):
    def __init__(self, data_type, data_size, latent_dim):
        super(ViTDecoder, self).__init__()
        self.length = sum(data_size)

        self.decoder_latent = nn.ModuleList()
        for name, size in zip(data_type, data_size):
            self.decoder_latent.add_module(
                f"decoder_{name}", nn.Linear(latent_dim, size)
            )

    def forward(self, x):
        res = []
        for dense in self.decoder_latent:
            res.append(dense(x))
        return res


class LinearEncoder(nn.Module):
    def __init__(self, data_type, data_size, latent_dim):
        super(LinearEncoder, self).__init__()

        # self.weight = [1 for _ in range(data_size)]
        weight = [1 / len(data_size) for _ in data_size]  # 固定权重
        # weight = [i/sum(data_size) for i in data_size]   # 可变权重

        self.dim = [int(weight[i] * latent_dim) for i in range(len(data_type))]
        if sum(self.dim) < latent_dim:
            self.dim[-1] += latent_dim - sum(self.dim)

        self.encoder_input = nn.ModuleList()
        for i, dtype in enumerate(data_type):
            self.encoder_input.add_module(f"encoder_linear_{dtype}", nn.Linear(data_size[i], self.dim[i]))

        self.encoder2latent = nn.Sequential(OrderedDict([
            ("encoder_dense", nn.Linear(latent_dim, latent_dim)),
            ("encoder_bn", nn.BatchNorm1d(latent_dim)),
            ("encoder_gelu", nn.GELU()),
        ])
        )

    def forward(self, x):
        input_tmp = []
        for i, dense in enumerate(self.encoder_input):
            input_tmp.append(dense(x[i]))
        x = torch.cat(input_tmp, dim=1)
        x = self.encoder2latent(x)
        return x


class LinearDecoder(nn.Module):
    def __init__(self, data_type, data_size, latent_dim):
        super(LinearDecoder, self).__init__()

        self.latent2decoder = nn.Sequential(OrderedDict([
            ("decoder_dense", nn.Linear(latent_dim, latent_dim)),
            ("decoder_bn", nn.BatchNorm1d(latent_dim)),
            ("decoder_gelu", nn.GELU())])
        )

        self.decoder_output = nn.ModuleList()
        for i, dtype in enumerate(data_type):
            self.decoder_output.add_module(f"decoder_linear_{dtype}",
                                           nn.Linear(latent_dim, data_size[i]))

    def forward(self, x):
        res = []
        x = self.latent2decoder(x)
        for i, dense in enumerate(self.decoder_output):
            res.append(dense(x))
        return res


class Discriminator(nn.Module):
    def __init__(self, data_type, data_size, latent_dim):
        super(Discriminator, self).__init__()
        self.desc = nn.Sequential(OrderedDict([
            ("desc_dense", nn.Linear(latent_dim, 1)),
            ('bn', nn.BatchNorm1d(1)),
            ("desc_softmax", nn.Softmax()),
            # nn.Softmax(dim=0)
        ])
        )
        self.act = nn.Softmax()

    def forward(self, x):
        x = self.desc(x)
        # log.info(f'test y{x.cpu().reshape(-1)}')
        # x = self.act(x)

        return x


class SCMAT(nn.Module):
    def __init__(self, model_config, data_type, data_size, model='Linear'):
        super(SCMAT, self).__init__()
        log.info(f"######### SCMAT {model} model init #########")
        self.module_cfg = parse_model_config(model_config)
        self.hyperparams, _ = create_modules(self.module_cfg)
        log.info(f"hyperparamas: {self.hyperparams}")
        self.use_gpu = torch.cuda.is_available()

        self.latent_dim = self.hyperparams['latent_dim']
        if model == 'transformer':
            self.length = sum(data_size)
            log.info(f'length: {self.length}')
            self.patch_size = 16
            # math.ceil(x) # 返回不小于x的最接近的整数
            patch_num = math.ceil(math.sqrt(self.length) / self.patch_size)
            log.info(f'patch_num: {patch_num}')
            self.latent_dim = patch_num * patch_num

        log.info(f'latent dim: {self.latent_dim}')

        self.encoder = LinearEncoder(data_type, data_size, self.latent_dim)  # 64,196
        if model == "transformer":
            self.encoder = ViTEncoder(data_type, data_size)

        self.encoder_mean = nn.Sequential(OrderedDict([
            ("decoder_mean", nn.Linear(self.latent_dim, self.latent_dim)),
            ("bn", nn.BatchNorm1d(self.latent_dim)),
            # ("gelu", nn.GELU())
        ])
        )

        self.encoder_log_var = nn.Sequential(OrderedDict([
            ("decoder_log_var", nn.Linear(self.latent_dim, self.latent_dim)),
            ("bn", nn.BatchNorm1d(self.latent_dim)),
            # ("gelu", nn.GELU())
        ])
        )

        self.decoder = LinearDecoder(data_type, data_size, self.latent_dim)
        if model == 'transformer':
            self.decoder = ViTDecoder(data_type, data_size, self.latent_dim)

        self.disc = Discriminator(data_type, data_size, self.latent_dim)
        # log.info(f'disc:{self.disc}')

    def encode(self, x):
        x = self.encoder(x)  # (64,196)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        return self.reparameterize(mean, log_var)

    def reparameterize(self, mean, log_var):
        """
        mean + randn * e**( log_var * 0.5)
        """
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        if self.use_gpu:
            eps = eps.cuda()
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mean)

    def decode(self, x):
        con_x = self.decoder(x)
        return con_x

    def forward(self, x):
        feature = self.encode(x)
        disc_res = self.disc(feature)
        con_x = self.decode(feature)
        # feature2 = self.encode(con_x)
        # disc_res2 = self.disc(feature2)
        # return feature, disc_res, con_x, feature2, disc_res2
        return feature, disc_res, con_x
