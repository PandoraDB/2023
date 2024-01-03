import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torch.nn.functional as F

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
        qkv = self.qkv(x)  #  676(26*26),480
        # print(qkv.shape)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1)
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


class SpatialAttention(nn.Module):
    # 传入卷积核大小
    def __init__(self, kernel_size=7):
        # 初始化
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 通过7x7卷积
        # 再通过sigmoid激活函数
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # 在通道上进行最大池化和平均池化
        # 再进行堆叠(通道上)
        # 堆叠后的结果卷积最后进行sigmoid
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def pad_to_even_dimensions(tensor):

    height, width = tensor.shape[-2:]

    new_size = max(height, width)
    new_size = new_size + (2 - new_size % 2) % 2  # 保证新的尺寸是2的倍数

    pad_height = (new_size - height)
    pad_width = (new_size - width)

    padded_tensor = F.pad(tensor, (0, pad_width, pad_height, 0), mode='constant', value=0)
    return padded_tensor


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # 将每个样本的每个通道的特征向量做归一化
        # 也就是说每个特征向量是独立做归一化的
        # 我们这里虽然是图片数据，但图片被切割成了patch，用的是语义的逻辑
        self.norm1 = norm_layer(dim+1*4)
        self.attn1 = Attention(dim+1*4, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = Attention(dim+1*4, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn3 = Attention(dim+1*4, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn4 = Attention(dim+1*4, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = [self.attn1, self.attn2, self.attn3, self.attn4]

        self.se1 = SpatialAttention(kernel_size=3)
        self.se2 = SpatialAttention(kernel_size=3)
        self.se3 = SpatialAttention(kernel_size=3)
        self.se4 = SpatialAttention(kernel_size=3)
        self.se = [self.se1, self.se2, self.se3, self.se4]


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()

        self.norm = norm_layer(dim) if norm_layer else nn.Identity()  # 如果没有传入norm层，就使用identity
    def forward(self, x):
        # 最后一维归一化，multi-head attention, drop_path
        # W = x.shape[3]

        if x.shape[2] % 2 != 0:
            x = pad_to_even_dimensions(x)

        H = int(x.shape[2] / 2)
        # (B,C,W,H)-> (B, N, C)
        C = x.shape[1]
        tensor = x

        # 沿着宽度维度（第四维）切分
        splits_w = tensor.chunk(2, dim=3)

        # 对每个部分沿着高度维度（第三维）切分
        splits = [part.chunk(2, dim=2) for part in splits_w]

        # 将切分后的部分组织成最终的四个 tensor
        final_tensors = [split for sublist in splits for split in sublist]
        out_image_patch = []
        out_token = []
        tokens = []

        for i, y in enumerate(final_tensors):
            se = self.se[i]
            token = se(y)
            token = token.flatten(2).transpose(1, 2)
            tokens.append(token)

        total_token = torch.cat(tokens, dim=2)

        for i, y in enumerate(final_tensors):
            # se = self.se[i]
            attn = self.attn[i]

            # token = se(y)
            # token = token.flatten(2).transpose(1, 2)
            y = y.flatten(2).transpose(1, 2)
            y = self.norm(y)
            y = torch.cat((y, total_token), dim=2)

            y = y + self.drop_path(attn(self.norm1(y)))

            # print('#####################################################')
            # print(y.shape)

            y, token = y[:, :, :C], y[:, :, C:]
            y = rearrange(y, 'b (h w) c-> b c h w', h=H)
            token = rearrange(token, 'b (h w) c-> b c h w', h=H)
            out_image_patch.append(y)
            out_token.append(token)
        # 使用 torch.stack 将它们堆叠起来
        out_token = torch.stack(out_token)
        out_token = torch.sum(out_token, dim=0)
        # test = out_token[:, i, :, :]
        out = [patch*out_token[:, i, :, :].unsqueeze(1) + patch for i, patch in enumerate(out_image_patch)]

        concatenated_h = [torch.cat([out[2*i], out[2*i+1]], dim=2) for i in range(2)]
        concatenated_wh = torch.cat(concatenated_h, dim=3)
        # print('#####################################################')
        # print(concatenated_wh.shape)
        # print(x.shape)
        # print('#####################################################')
        x = concatenated_wh + x
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# from thop import profile

if __name__ == '__main__':
    block = Block(dim=16, num_heads=4)
    input_features = torch.randn(2, 16, 64, 64)
    out_put = block(input_features)
    print(111)

    flops, params = profile(block, inputs=(input_features,))
    print(params)