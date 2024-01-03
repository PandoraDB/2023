import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# Embedding层代码解读
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size  # 每个patch的大小
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 224/16 -> 14*14
        # self.num_patches = self.grid_size[0] * self.grid_size[1]  # patches的数目


        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 卷积核大小和patch_size都是16*16
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # 如果没有传入norm层，就使用identity

    def forward(self, x):
        B, C, H, W = x.shape  # 注意，在vit模型中输入大小必须是固定的，高宽和设定值相同
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# MLP Head层代码解读
class Mlp(nn.Module):  # Encoder中的MLP Block
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 如果没有传入out features，就默认是in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # 默认是GELU激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    # drop_prob是进行droppath的概率
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    # 在ViT中，shape是(B,1,1),B是batch size
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 按shape,产生0-1之间的随机向量,并加上keep_prob
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 向下取整，二值化，这样random_tensor里1出现的概率的期望就是keep_prob
    random_tensor.floor_()  # binarize
    # 将一定图层变为0
    output = x.div(keep_prob) * random_tensor
    return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # q,k,v向量长度
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # (B,C,W,H)-> (B, N, C)
        # x = x.flatten(2).transpose(1, 2)
        # x = x.norm(x)
        # 这里C对应上面的E，向量的长度
        B, N, C = x.shape
        # (B, N, C) -> (3，B，num_heads, N, C//num_heads), //是向下取整的意思。
        qkv = self.qkv(x)
        # print(qkv.shape)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 将qkv在0维度上切成三个数据块，q,k,v:(B，num_heads, N, C//num_heads)
        # 这里的效果是从每个向量产生三个向量，分别是query，key和value
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # @矩阵相乘获得score (B,num_heads,N,N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # (B,num_heads,N,N)@(B,num_heads,N,C//num_heads)->(B,num_heads,N,C//num_heads)
        # (B,num_heads,N,C//num_heads) ->(B,N,num_heads,C//num_heads)
        # (B,N,num_heads,C//num_heads) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # (B, N, C) -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 将每个样本的每个通道的特征向量做归一化
        # 也就是说每个特征向量是独立做归一化的
        # 我们这里虽然是图片数据，但图片被切割成了patch，用的是语义的逻辑
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # 全连接，激励，drop，全连接，drop,若out_features没填，那么输出维度不变。
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()  # 如果没有传入norm层，就使用identity
    def forward(self, x):
        # 最后一维归一化，multi-head attention, drop_path
        W = x.shape[3]
        H = x.shape[2]
        # (B,C,W,H)-> (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # (B, N, C) -> (B, N, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # (B, N, C) -> (B, N, C)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        #(B, N, C) -> (B.C.W.H)
        # x = self.transpose(1,2)
        x = rearrange(x, 'b (h w) c-> b c h w', h=H)
        return x

if __name__ == '__main__':
    block = Block(dim=480, num_heads=4)
    input_features = torch.randn(2, 480, 26, 26)
    out_put = block(input_features)
    print(111)