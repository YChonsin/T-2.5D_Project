
import torch
import torch.nn as nn
import math

from einops import rearrange, repeat


DWCONV3D_DISABLE_CUDNN = True


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # print(x.shape)
        x = self.net(x)
        # print(x.shape)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    # 待修改
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, _):
        # B, L, C
        # print(x.shape)
        # print(self.layers[0][0].to_out[0].weight)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = self.sigmoid(self.conv(x_cat))         # [B, 1, H, W]
        return x * attn

class SEBlock_3d(nn.Module):
    '''通道注意力'''
    def __init__(self, in_channels, reduction=16):
        super(SEBlock_3d, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class S3DAdapter_4TIR(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, n_conv):
        '''

        :param in_channels: dim
        :param adapter_channels: higher than in_channels
        :param kernel_size: (3, 1, 1) temporal modeling only; (3, 3, 3) spatial and temporal modeling
        '''
        super().__init__()

        self.dim = adapter_channels
        self.fc1 = nn.Linear(in_channels, adapter_channels)

        self.conv = nn.ModuleList()

        for i in range(n_conv):
            self.conv.append(nn.Conv3d(
                adapter_channels, adapter_channels,
                kernel_size=kernel_size,
                stride=(1, 1, 1),
                padding=tuple(x // 2 for x in kernel_size),
                groups=adapter_channels,
            ))

        self.embed_dim = adapter_channels
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # nn.init.constant_(self.conv.weight, 0.)
        # nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, x_size):
        H, W, X = x_size
        BX, L, C = x.shape
        B = BX // X
        Ca = self.embed_dim
        # H = W = round(math.sqrt(L))
        assert L == H * W

        x_id = x.clone()

        # BX L C
        x = self.fc1(x)
        x = x.view(B, X, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() # B Ca X H W

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        for conv in self.conv:
            x = conv(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        # 恢复形状
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BX, L, Ca)   # BX, HW, Ca
        x = self.fc2(x)
        # BX HW C
        x_id += x
        return x_id



class TransAdapter_HX(nn.Module):

    def __init__(self, in_channels, adapter_channels, n_trans, num_heads, mlp_dim, dropout):
        '''
        H-X modeling
        :param in_channels: dim
        :param adapter_channels: higher than in_channels
        :param kernel_size: (3, 1, 1) temporal modeling only; (3, 3, 3) spatial and temporal modeling
        '''
        super().__init__()

        self.dim = adapter_channels
        self.fc1 = nn.Linear(in_channels, adapter_channels)

        self.trans = nn.ModuleList()

        for i in range(n_trans):
            self.trans.append(
                TransformerEncoder(
                    dim=adapter_channels,
                    depth=1,
                    heads=num_heads,
                    dim_head=(adapter_channels // num_heads),
                    mlp_dim=mlp_dim,
                    dropout=dropout,)
            )

        self.embed_dim = adapter_channels
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # nn.init.constant_(self.conv.weight, 0.)
        # nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, x_size):
        H, W, X = x_size
        BX, L, C = x.shape
        B = BX // X
        Ca = self.embed_dim
        # H = W = round(math.sqrt(L))
        assert L == H * W

        x_id = x.clone()

        # BW, HX, Ca
        x = self.fc1(x)
        x = x.view(B, X, H, W, Ca).permute(0, 3, 2, 1, 4).contiguous() # B W H X Ca
        x = x.view(B * W, H * X, Ca) # BW, HX, Ca
        for te in self.trans:
            x = te(x, 0)
        # BW, HX, Ca
        x = x.view(B, W, H, X, Ca).permute(0, 3, 2, 1, 4).contiguous() # B X H W Ca
        x = x.view(B * X, H * W, Ca)
        x = self.fc2(x)
        x_id += x
        return x_id



class TransAdapter_X(nn.Module):

    def __init__(self, in_channels, adapter_channels, n_trans, num_heads, mlp_dim, dropout):
        '''
        H-X modeling
        :param in_channels: dim
        :param adapter_channels: higher than in_channels
        :param kernel_size: (3, 1, 1) temporal modeling only; (3, 3, 3) spatial and temporal modeling
        '''
        super().__init__()

        self.dim = adapter_channels
        self.fc1 = nn.Linear(in_channels, adapter_channels)

        self.trans = nn.ModuleList()

        for i in range(n_trans):
            self.trans.append(
                TransformerEncoder(
                    dim=adapter_channels,
                    depth=1,
                    heads=num_heads,
                    dim_head=(adapter_channels // num_heads),
                    mlp_dim=mlp_dim,
                    dropout=dropout,)
            )

        self.embed_dim = adapter_channels
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # nn.init.constant_(self.conv.weight, 0.)
        # nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, x_size):
        H, W, X = x_size
        BX, L, C = x.shape
        B = BX // X
        Ca = self.embed_dim
        # H = W = round(math.sqrt(L))
        assert L == H * W

        x_id = x.clone()

        # BW, HX, Ca
        x = self.fc1(x)
        x = x.view(B, X, H, W, Ca).permute(0, 2, 3, 1, 4).contiguous() # B H W X Ca
        x = x.view(B * H * W, X, Ca) # BHW, X, Ca
        for te in self.trans:
            x = te(x, 0)
        # BW, HX, Ca
        x = x.view(B, H, W, X, Ca).permute(0, 3, 1, 2, 4).contiguous() # B X H W Ca
        x = x.view(B * X, H * W, Ca)
        x = self.fc2(x)
        x_id += x
        return x_id



class S3DAdapter_4TIR_v2(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size, n_conv,
                 groups=8):
        '''

        :param in_channels: dim
        :param adapter_channels: higher than in_channels
        :param kernel_size: (3, 1, 1) temporal modeling only; (3, 3, 3) spatial and temporal modeling
        '''
        super().__init__()

        self.dim = adapter_channels
        self.fc1 = nn.Linear(in_channels, adapter_channels)

        self.conv = nn.ModuleList()

        for i in range(n_conv):
            self.conv.append(
                    nn.Conv3d(
                    adapter_channels, adapter_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    padding=tuple(x // 2 for x in kernel_size),
                    groups=adapter_channels,)
            )
            self.conv.append(
                nn.GroupNorm(
                    num_groups=min(groups, adapter_channels),
                    num_channels=adapter_channels)
            )
            self.conv.append(
                nn.PReLU(adapter_channels)
            )
            self.conv.append(
                nn.Conv3d(
                    adapter_channels, adapter_channels,
                    kernel_size=1,
                    bias=False
                )
            )

        self.alpha = nn.Parameter(torch.zeros(1))

        self.embed_dim = adapter_channels
        self.fc2 = nn.Linear(adapter_channels, in_channels)

        # nn.init.constant_(self.conv.weight, 0.)
        # nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, x_size):
        H, W, X = x_size
        BX, L, C = x.shape
        B = BX // X
        Ca = self.embed_dim
        # H = W = round(math.sqrt(L))
        assert L == H * W

        x_id = x

        # BX L C
        x = self.fc1(x)
        x = x.view(B, X, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous() # B Ca X H W

        cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = cudnn_enabled and DWCONV3D_DISABLE_CUDNN
        for blk in self.conv:
            x = blk(x)
        torch.backends.cudnn.enabled = cudnn_enabled

        # 恢复形状
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BX, L, Ca)   # BX, HW, Ca
        x = self.fc2(x)
        # BX HW C
        x_id = x_id + self.alpha * x
        return x_id




if __name__ == '__main__':
    x = torch.randn((1, 25088, 32))
    model = S3DAdapter_4TIR(in_channels=32, adapter_channels=384
                            , kernel_size=(3, 1, 1), n_conv=1)
    print(x.shape)
    x = model(x, 8)
    print(x.shape)