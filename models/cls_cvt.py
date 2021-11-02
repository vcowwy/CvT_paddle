from functools import partial
from itertools import repeat
import collections.abc as container_abcs

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy

import paddle
import paddlenlp
import paddle.nn as nn
import paddle.nn.functional as F

from .rearrange import rearrange
from .rearrange import Rearrange

from ppcls.vision_transformer import DropPath
from ppcls.vision_transformer import trunc_normal_
#from ppcls.vision_transformer import zeros_
#from ppcls.vision_transformer import ones_
from ppcls.vision_transformer import Identity

from .registry import register_model


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        ret = super().forward(paddle.to_tensor(x, dtype=paddle.float32))
        return paddle.to_tensor(ret, dtype=orig_type)


class QuickGELU(nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(1.702 * x)


class Mlp(nn.Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads

        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                ('conv', nn.Conv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    groups=dim_in,
                    bias_attr=False)),
                ('bn', nn.BatchNorm2D(dim_in)),
                ('rearrage', Rearrange()))
        elif method == 'avg':
            proj = nn.Sequential(
                ('avg', nn.AvgPool2D(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True)),
                ('rearrage', Rearrange()))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = paddle.split(x, [1, h * w], 1)

        x = rearrange(x, 1, h_new=h)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 0)

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 0)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 0)

        if self.with_cls_token:
            q = paddle.concat((cls_token, q), axis=1)
            k = paddle.concat((cls_token, k), axis=1)
            v = paddle.concat((cls_token, v), axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 2, h_new=self.num_heads)
        k = rearrange(self.proj_k(k), 2, h_new=self.num_heads)
        v = rearrange(self.proj_v(v), 2, h_new=self.num_heads)

        attn_score = paddlenlp.ops.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = paddlenlp.ops.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 3)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        flops += T_Q * T_KV * module.dim
        flops += T_Q * module.dim * T_KV
        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [p.numel() for p in module.conv_proj_q.conv.parameters()]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [p.numel() for p in module.conv_proj_k.conv.parameters()]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [p.numel() for p in module.conv_proj_v.conv.parameters()]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias,
            attn_drop, drop, **kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding
    """
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 0)
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 1, h_new=H)

        return x


class VisionTransformer(nn.Layer):
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer)

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = self.create_parameter(
                shape=[1, 1, embed_dim],
                attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.0)))
            self.cls_token.stop_gradient = False
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, stop=drop_path_rate, num=depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.LayerList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, paddle.nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            #trunc_normal_(m.weight)
            m._weight_attr = paddle.nn.initializer.TruncatedNormal(std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                #zeros_(m.bias)
                m._bias_attr = paddle.nn.initializer.Constant(value=0.0)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm2D)):
            #zeros_(m.bias)
            m._bias_attr = paddle.nn.initializer.Constant(value=0.0)
            #ones_(m.weight)
            m._weight_attr = paddle.nn.initializer.Constant(value=1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, paddle.nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            #nn.init.xavier_uniform_(m.weight)
            m._weight_attr = nn.initializer.XavierNormal()
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                #zeros_(m.bias)
                m._bias_attr = paddle.nn.initializer.Constant(value=0.0)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.BatchNorm2D)):
            #zeros_(m.bias)
            m._bias_attr = paddle.nn.initializer.Constant(value=0.0)
            #ones_(m.weight)
            m._weight_attr = paddle.nn.initializer.Constant(value=1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x = rearrange(x, 0)

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat((cls_tokens, x), axis=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H * W], 1)
        x = rearrange(x, 1, h_new=H)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Layer):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i]
            }

            stage = VisionTransformer(in_chans=in_chans,
                                      init=init,
                                      act_layer=act_layer,
                                      norm_layer=norm_layer,
                                      **kwargs
                                      )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else Identity()
        #trunc_normal_(self.head.weight)
        self.head._weight_attr = paddle.nn.initializer.TruncatedNormal(std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = paddle.load(pretrained)
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (k.split('.')[0] in pretrained_layers
                             or pretrained_layers[0] == '*')

                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.shape() != model_dict[k].shape():
                        size_pretrained = v.shape()
                        size_new = model_dict[k].shape()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'.
                            format(gs_old, gs_new))

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = gs_new / gs_old, gs_new / gs_old, 1
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = paddle.to_tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )
                    need_init_state_dict[k] = v

            self.load_state_dict(need_init_state_dict, strict=False)

    """@torch.jit.ignore"""
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x)
        else:
            x = rearrange(x, 0)
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def get_cls_model(config, **kwargs):
    msvit_spec = config.MODEL.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, epsilon=1e-05),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

    if config.MODEL.INIT_WEIGHTS:
        msvit.init_weights(
            config.MODEL.PRETRAINED,
            config.MODEL.PRETRAINED_LAYERS,
            config.VERBOSE)

    return msvit
