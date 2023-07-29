import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms, datasets,models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from visdom import Visdom


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim//2, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
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

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        nn.BatchNorm2d(dim_out),
        # LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )

def Aggregate1(dim):
    return nn.Sequential(
        LayerNorm(dim),
        nn.AvgPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        # pos_emb = self.pos_emb[:(h * w)]
        # pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        # x = x + pos_emb
        x = x

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
class Amend_raf(nn.Module):  # moren
    def __init__(self, inplace=4):
        super(Amend_raf, self).__init__()
        self.de_albino = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=12, stride=4, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(inplace)
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        mask = torch.tensor([]).to("cuda:0")
        createVar = locals()
        for i in range(x.size(1)):
            createVar['x' + str(i)] = torch.unsqueeze(x[:, i], 1)
            createVar['x' + str(i)] = self.de_albino(createVar['x' + str(i)])
            mask = torch.cat((mask, createVar['x' + str(i)]), 1)
        x = self.bn(mask)
        xmax, _ = torch.max(x, 1, keepdim=True)
        global_mean = x.mean(dim=[0, 1])
        xmean = torch.mean(x, 1, keepdim=True)
        xmin, _ = torch.min(x, 1, keepdim=True)
        x = xmean + self.alpha * global_mean

        return x, self.alpha

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dims = [dim, dim * 3, dim * 9, dim * 27]
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(3,dim,1),
            # nn.Conv2d(dim,dim,patch_size,stride=patch_size),
            #
            # Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            # nn.Conv2d(patch_dim, dim, 1),
            nn.Conv2d(self.dims[0], self.dims[0], 3, stride=2, padding=1),
            nn.Conv2d(self.dims[0],self.dims[0],1),
            # nn.Conv2d(3,dim,3,stride=2,padding=3),
            nn.BatchNorm2d(dim),
            # nn.MaxPool2d(3,stride=2,padding=3),

        )
        resnet = models.resnet18(True)
        self. reshead = nn.Sequential(*list(resnet.children())[0:4])
        print(self.reshead)
        self.tf1 = Transformer(self.dims[0], 4, 8, 4, dropout)
        self.agg1 = Aggregate(self.dims[0], self.dims[0] * 3)
        self.lagg1 = Aggregate1(self.dims[0])
        self.tf2 = Transformer(self.dims[1], 6, 4, 8, dropout)
        self.agg2 = Aggregate(self.dims[1], self.dims[1] * 3)
        self.lagg2 = Aggregate1(self.dims[1])
        self.tf3 = Transformer(self.dims[2], 1, 1, 4, dropout)
        self.agg3 = Aggregate(self.dims[2], self.dims[2] * 3)
        self.lagg3 = Aggregate1(self.dims[2])
        self.fcout = nn.Linear(self.dims[2], num_classes)


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(self.dims[2]),
            nn.AdaptiveMaxPool2d(3),
            Reduce('b c h w -> b c', 'mean')
        )
        self.arrangement = nn.PixelShuffle(12)
        self.arm = Amend_raf()
        self.fc = nn.Linear(100, 400)
        self.fcout = nn.Linear(400, num_classes)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.reshead(img)
        x = self.to_patch_embedding(x)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=1, b2=1)
        x = self.tf1(x)
        x = self.agg1(x)
        # x = torch.cat((self.lagg1(x), ), dim=1)
        x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=1, b2=1)
        # x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=1, b2=1)
        x = self.tf2(x)
        x = self.agg2(x)
        # x = torch.cat((self.lagg2(x), self.agg2(x)), dim=1)
        # x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=1, b2=1)
        # x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=1, b2=1)
        # x = self.tf3(x)
        # x = torch.cat((self.lagg3(x), self.agg3(x)), dim=1)
        # x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=1, b2=1)
        # return self.se(x)
        # x = self.mlp_head(x)
        # return self.fcout(x),1
        x = self.arrangement(x)

        x, alpha = self.arm(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.fcout(x)
        out = self.sigmoid(out)
        return out,x

def ftt_pretrain(imagesize = 128,class_num = 7):
    model = ViT(
            image_size=imagesize,
            patch_size=2,
            num_classes=class_num,
            dim=64,
            depth=3,
            heads=3,
            mlp_dim=16,
            dropout=0,
            emb_dropout=0
        )
    checkpoint = torch.load("kvit-ckp.pth")
    del checkpoint["model_state_dict"]['fcout' + '.weight']
    del checkpoint["model_state_dict"]['fcout' + '.bias']
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model

# ftt_pretrain()
