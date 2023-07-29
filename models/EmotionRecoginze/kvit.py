from math import sqrt

import torch
from torch import nn, einsum
# import pytorch_lightning as pl
# import torch.nn.functional as F
from torchvision import models
from einops.layers.torch import Rearrange, Reduce

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num


def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# classes
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim//2, 1),
            # nn.BatchNorm2d(dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim//2, dim, 1),
            # nn.BatchNorm2d(dim),
            nn.Dropout(dropout)

        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        # inner_dim = dim * heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.attn_drop = nn.Dropout(0.)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.attn_drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class ClassTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                # FeedForward(dim, mlp_mult, dropout=dropout)
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# depthwise convolution, for pooling

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ClassEmbeding(nn.Module):
    def __init__(self, img_size, dim_in, dim_out, patch_num=16, dropout=0.):
        super().__init__()
        self.patchdim = dim_in * ((img_size / patch_num) ** 2)
        self.net = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', h=patch_num, w=patch_num),
            nn.Linear(self.patchdim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class ConvEmbeding(nn.Module):
    def __init__(self, img_size, dim_in=64, dim_out=64, patch_num=16, dropout=0.):
        super().__init__()
        in_size = img_size / 4
        resnet = models.resnet18(True)
        self.convhead = nn.Sequential(*list(resnet.children())[0:4])
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, stride=2, padding=1),
            nn.Conv2d(dim_in, dim_out, 1),
            # nn.Conv2d(3,dim,3,stride=2,padding=3),
            nn.BatchNorm2d(dim_out),
        )

    def forward(self, x):
        x = self.convhead(x)
        return self.net(x)


# pooling layer

class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h=int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim=1)


def ConvPool(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        nn.BatchNorm2d(dim_out),
        # LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )

class Baseline_clf(nn.Module):  # moren
    def __init__(self, inplace=4):
        super(Baseline_clf, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=12, stride=4, padding=0, bias=False,groups=2)
        self.bn = nn.BatchNorm2d(2)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        # mask = torch.tensor([]).to("cuda:0")
        # createVar = locals()
        # for i in range(x.size(1)):
        #     createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
        #     createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
        #     mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        mask = self.conv1(x)
        x = self.bn(mask)
        xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        xmin, _ = torch.min(x, 1, keepdim=True)
        x = xmean + self.alpha * global_mean
        return x, self.alpha

class Emotion_base(nn.Module):  # moren
    def __init__(self, inplace=100, outplace=6):
        super(Emotion_base, self).__init__()
        self.fc_class = nn.Sequential(
            nn.Linear(inplace, outplace),
            nn.BatchNorm1d(outplace)
        )
        self.fc_basclass = nn.Sequential(
            nn.Linear(inplace, outplace),
            nn.BatchNorm1d(outplace),
            nn.Sigmoid()
        )
        self.fc_arounce = nn.Sequential(
            nn.Linear(inplace, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        arounce = self.fc_arounce(x)
        mask = self.fc_basclass(x)
        x = self.fc_class(x)
        x = x*mask
        x = self.softmax(x)
        x = x*arounce

        return x, arounce
# main class

class KViT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim=64,
            depth=(4, 6),
            heads=(8, 4),
            mlp_dim=(4, 8),
            dim_head=64,
            dropout=0.,
            emb_dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth,
                          tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))
        self.dim = (dim, dim * 3,dim * 9)
        self.depth = depth
        self.heads = heads
        self.mlp_mult = mlp_dim

        patch_dim = 3 * patch_size ** 2
        resnet = models.resnet18(True)
        self.reshead = nn.Sequential(*list(resnet.children())[0:4])
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.Conv2d(64,64,1),
            nn.BatchNorm2d(64),
        )
        # self.to_patch_embedding = ConvEmbeding(128, dim_in=64, dim_out=64)
        self.use_classtoken = False
        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 36))
        self.dropout = nn.Dropout(emb_dropout)
        self.arrangement = nn.PixelShuffle(12)
        self.arm = Baseline_clf()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        layers = []

        for ind in range(len(self.depth)):
            not_last = ind < (len(self.depth) - 1)
            layers.append(Transformer(self.dim[ind], self.depth[ind], self.heads[ind], self.mlp_mult[ind], dropout))
            # if not_last:
            #     layers.append(ConvPool(self.dim[ind], self.dim[ind + 1]))
            layers.append(ConvPool(self.dim[ind], self.dim[ind + 1]))

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(
            LayerNorm(self.dim[-1]),
            nn.AdaptiveMaxPool2d(3),
            Reduce('b c h w -> b c', 'mean')
        )
        from vit import Transformer as classTransformer
        self.mlp_tf = classTransformer(dim=36, depth=4, heads=8, dim_head=12, mlp_dim=128)
        self.mlp_fc = nn.Linear(100, num_classes)

    def forward(self, img):
        x = self.reshead(img)
        x = self.to_patch_embedding(x)
        b, n, _ ,_ = x.shape

        # if self.use_classtoken:
        #     cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        #     x = torch.cat((cls_tokens, x), dim=1)
        #     x += self.pos_embedding[:, :n + 1]
        # else:
        #     x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.layers(x)
        x = self.arrangement(x)
        x, alpha = self.arm(x)
        out = x.view(x.size(0), -1)
        # out = self.mlp_fc(x)
        return out

class KViT_backbone(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            dim=64,
            depth=(4, 6),
            heads=(8, 4),
            mlp_dim=(4, 8),
            dim_head=64,
            dropout=0.,
            emb_dropout=0.
    ):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth,
                          tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))
        self.dims = (dim, dim * 3, dim * 9)
        self.depth = depth
        self.heads = heads
        self.mlp_mult = mlp_dim
        self.dropout = nn.Dropout(emb_dropout)
        patch_dim = 3 * patch_size ** 2
        resnet = models.resnet18(True)
        self.reshead = nn.Sequential(*list(resnet.children())[0:4])
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(64, self.dims[0], 3, stride=2, padding=1),
            nn.Conv2d(self.dims[0],self.dims[0],1),
            nn.BatchNorm2d(self.dims[0]),
        )

        layers = []
        for ind in range(len(self.depth)):
            not_last = ind < (len(self.depth) - 1)
            layers.append(Transformer(self.dims[ind], self.depth[ind], self.heads[ind], self.mlp_mult[ind], dropout))
            # if not_last:
            #     layers.append(ConvPool(self.dim[ind], self.dim[ind + 1]))
            layers.append(ConvPool(self.dims[ind], self.dims[ind + 1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, img):
        x = self.reshead(img)
        x = self.to_patch_embedding(x)
        b, n, _ ,_ = x.shape

        # if self.use_classtoken:
        #     cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        #     x = torch.cat((cls_tokens, x), dim=1)
        #     x += self.pos_embedding[:, :n + 1]
        # else:
        #     x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.layers(x)
        return x

class KViT_classifier(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_classes,

    ):
        super().__init__()

        # self.to_patch_embedding = ConvEmbeding(128, dim_in=64, dim_out=64)
        self.dim = dim
        self.arrangement = nn.PixelShuffle(12)
        self.mlp_baseline = Baseline_clf()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.mlp_head = nn.Sequential(
            LayerNorm(self.dim),
            nn.AdaptiveMaxPool2d(3),
            Reduce('b c h w -> b c', 'mean')
        )
        self.mlp_fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.arrangement(x)
        x, alpha = self.mlp_baseline(x)
        x = x.view(x.size(0), -1)
        out = self.mlp_fc(x)
        return self.sigmoid(out)

class kvit_pretrained(nn.Module):  # moren
    def __init__(self,ckpt_path = "kvit-epoch=39-val_acc=0.8934.ckpt"):
        super(kvit_pretrained, self).__init__()
        self.model_backbone = KViT_backbone(
            image_size=128,
            patch_size=16,
            dim=64,
            depth=(3, 4),
            heads=(8, 4),
            mlp_dim=(4, 8),
            dim_head=64,
            dropout=0,
            emb_dropout=0)
        self.model_classifier = KViT_classifier(dim=576, num_classes=5)
    def forward(self, x):
        x = self.model_backbone(x)
        out = self.model_classifier(x)
        return out


class test(nn.Module):  # moren
    def __init__(self):
        super(test, self).__init__()
        self.model = KViT(
            image_size=128,
            patch_size=16,
            num_classes=5,
            dim=64,
            depth=(3, 4),
            heads=(8, 4),
            mlp_dim=(4, 8),
            dim_head=64,
            dropout=0,
            emb_dropout=0)
        self.mlp_fc = nn.Linear(100, 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        out = self.mlp_fc(x)
        return self.sigmoid(out), x