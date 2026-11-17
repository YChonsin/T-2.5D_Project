# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from einops import rearrange
from src.networks.tl.Seismic_Dimension_Adapter import S3DAdapter_4TIR, TransAdapter_HX
from src.networks.tl.lib.PatchEmbedUnembed_t import PatchEmbed, PatchUnEmbed
# from transfer_learning.params_load import load_specific_params

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

class TransformerEncoder_Adapted(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.,

                 adapter_type='pre-stack',
                 adapter_kernel_size=(3, 3, 3),
                 adapter_channels=384,
                 ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # print(dim)
        match adapter_type:
            case 'S3D':
                self.adapter = S3DAdapter_4TIR(in_channels=dim,
                                               adapter_channels=adapter_channels,
                                               kernel_size=adapter_kernel_size, n_conv=1)
            case 'Trans':
                self.adapter = TransAdapter_HX(in_channels=dim,
                                               adapter_channels=adapter_channels,
                                               kernel_size=adapter_kernel_size,
                                               num_heads=heads,
                                               mlp_dim=mlp_dim,
                                               dropout=dropout,
                                               n_trans=1,
                                               )
            case 'None':
                self.adapter = None



        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))


        # for n, p in self.named_parameters():
        #   if 'adapter' not in n:
        #     p.requires_grad_(False)

    def forward(self, x, x_size):
        # BX, HW, C = x.shape
        H, W, X = x_size
        # print(x.shape)

        if self.adapter is not None:
            x = self.adapter(x, x_size)  # BX, HW, C



        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)
        return x




class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,

                 adapter_type='pre-stack',
                adapter_kernel_size=(3, 3, 3),
                adapter_channels=384,

                 encoder_5_exist=True,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # used for dense connections
        self.conv = nn.ModuleList()
        for i in range(2, 2 + depth):
            self.conv.append(nn.Conv2d(in_channels=i * dim, out_channels=dim,
                                       kernel_size=3, stride=1, padding=1))

        # build blocks

        self.blocks = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            self.blocks.append(
                TransformerEncoder_Adapted(dim=dim,
                                   depth=1,
                                   heads=num_heads,
                                   dim_head=(dim // num_heads),
                                   mlp_dim=int(mlp_ratio*dim),
                                   dropout=attn_drop,

                                   adapter_type=adapter_type,
                                    adapter_kernel_size=adapter_kernel_size,
                                    adapter_channels=adapter_channels,
                                   )
            )

        self.last_block = None



        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.patch_embed = PatchEmbed(
            img_size=input_resolution, patch_size=1, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=input_resolution, patch_size=1, in_chans=0, embed_dim=dim,
            norm_layer=None)




        self.down = nn.Conv3d(in_channels=dim, out_channels=dim * 2,
                              kernel_size=3, stride=2, padding=1)
        self.up = nn.ConvTranspose3d(in_channels=dim * 2, out_channels=dim,
                              kernel_size=3, stride=2, padding=1, output_padding=1)

        # self.side_blocks = nn.ModuleList()
        # for i in range(depth):
        #     self.side_blocks.append(
        #         TransformerEncoder(dim=2 * dim,
        #                            depth=1,
        #                            heads=num_heads,
        #                            dim_head=(2 * dim // num_heads),
        #                            mlp_dim=int(mlp_ratio* 2 * dim),
        #                            dropout=attn_drop,
        #                            )
        #     )

        # self.posembed = PositionalEncoding2D(dim)
        # self.posembed_3d = PositionalEncoding3D(dim)
    def forward(self, x, x_size):
        BX, HW, C = x.shape
        H, W, X = x_size
        B = BX // X
        # BX, HW, C
        # main path
        x_list = []
        for i, blk in enumerate(self.blocks):

            x_list.append(x)

            # x1 = self.patch_unembed(x, x_size)  # BX, C, H, W
            # x1 = x1.permute(0, 2, 3, 1).contiguous()  # BX, H, W, C
            # x1 = self.posembed(x1)
            # x1 = x1.permute(0, 3, 1, 2).contiguous()  # BX, C, H, W
            # x1 = self.patch_embed(x1)
            # x = x + x1

            x = blk(x, x_size)

            if i < self.depth:
                # BX, HW, C
                for tmp in x_list:
                    x = torch.cat((x, tmp), dim=2)
                count = len(x_list)

                x = x.permute(0, 2, 1)
                x = x.view(x.shape[0], x.shape[1], x_size[0], x_size[1])
                x = self.conv[count - 1](x)
                x = x.view(x.shape[0], x.shape[1], x_size[0] * x_size[1])
                x = x.permute(0, 2, 1)

            # print(f"{i}: require grad? {blk.layers[0][0].to_out[0].weight.requires_grad}")

        if self.last_block is not None:
            x = self.patch_unembed(x, x_size)  # BX, C, H, W
            x = x.view(B, X, C, H, W).permute(0, 3, 4, 1, 2).contiguous().view(B, H * W * X, C)  # B, HWX, C

            # x = x.view(B, X, H, W, C).permute(0, 2, 3, 1, 4).contiguous().view(B, H * W * X, C)  # B, HWX, C
            # x = x + x1

            x = self.last_block(x, x_size)

            x = x.view(B, H, W, X, C).permute(0, 3, 1, 2, 4).contiguous()  # B X H W C
            x = x.view(BX, HW, C)

        # # side path (abandoned)
        # BX, HW, C = x2.shape
        # H, W, X = x_size
        # B = BX // X
        # x2 = x2.view(B, X, H, W, C).permute(0, 4, 2, 3, 1).contiguous()  # B, C, H, W, X
        # x2 = self.down(x2)  # B, 2C, 1/2H, 1/2W, 1/2X
        # x2 = x2.view(B, 2*C, (H//2) * (W//2) * (X//2)).permute(0, 2, 1).contiguous()  # B, 1/2H 1/2W 1/2X, 2C
        #
        #
        # for blk in self.side_blocks:
        #     x1 = x2.view(B, (H//2), (W//2), (X//2), 2*C)  # B, 1/2H, 1/2W, 1/2X, 2C
        #     x1 = self.posembed_3d(x1)
        #     x1 = x1.view(B, (H//2) * (W//2) * (X//2), 2*C)  # B, 1/2H 1/2W 1/2X, 2C
        #     x2 = x2 + x1
        #     x_size1 = (H // 2, W // 2, X // 2)
        #     x2 = blk(x2, x_size1)
        # x2 = x2.view(B, (H//2), (W//2), (X//2), 2*C).permute(0, 4, 1, 2, 3).contiguous()
        # x2 = self.up(x2)
        # x2 = x2.view(B, H, W, X, C).permute(0, 3, 1, 2, 4).contiguous().view(BX, HW, C)
        #
        # # integration
        # x = x + x2

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv',

                 adapter_type='pre-stack',
                 adapter_kernel_size=(3, 3, 3),
                 adapter_channels=384,

                 encoder_5_exist=True,
                 ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,

                                         adapter_type=adapter_type,
                                        adapter_kernel_size=adapter_kernel_size,
                                        adapter_channels=adapter_channels,


                                        encoder_5_exist=encoder_5_exist,
                                         )

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)


        self.patch_embed = PatchEmbed(
            img_size=input_resolution, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=input_resolution, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        # BX, HW, C
        x1 = x.clone()
        x = self.residual_group(x, x_size)  # B, HWX, C
        x = self.patch_unembed(x, x_size)
        # BX, C, H, W
        x = self.conv(x)
        x = self.patch_embed(x) + x1
        return x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W, Z = self.input_resolution
        flops += H * W * Z * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops



class TranfsormerIR_3d(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        data_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """
    # upscale=1, in_chans=1, img_size=128, window_size=8,
    # img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    # mlp_ratio=2, upsampler='', resi_connection='1conv'
    def __init__(self,
                 data_size: tuple,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=96,
                 depths=[3, 1],
                 num_heads=[2, 2],
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='None',
                 resi_connection='1conv',

                 adapter_type='pre-stack',
                adapter_kernel_size=(3, 3, 3),
                adapter_channels=384,

                 encoder_5_exist=True,
                 **kwargs):
        super(TranfsormerIR_3d, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.data_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        h, w, x = data_size
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=(h, w), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=(h, w), patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.absolute_pos_embed = nn.Parameter(torch.zeros(x, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(h, w),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=data_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         adapter_type=adapter_type,
                        adapter_kernel_size=adapter_kernel_size,
                        adapter_channels=adapter_channels,

                         encoder_5_exist=encoder_5_exist,
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1,
                               padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1,
                               padding=2, bias=False)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, stride=1,
                               padding=2, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, X):
        # BX C H W
        x_size = (x.shape[2], x.shape[3], X)
        x = self.patch_embed(x)  # BX Ph*Pw C

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # BX, HW, C
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # BX HW C
        x = self.patch_unembed(x, x_size)
        # BX C H W
        return x

    def forward(self, x):
        B, C, H, W, X = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * X, C, H, W)  # BX C H W

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.data_range

        # head block
        x1 = self.conv_first(x)
        # print(x.shape, x1.shape)
        x_id = x1.clone()
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x_id
        x_id = x1.clone()
        x1 = self.conv3(x1)
        x1 = self.relu2(x1)
        x1 = self.conv4(x1)
        x1 = x1 + x_id
        # print(x.shape, x1.shape)

        res = self.forward_features(x1, X)
        # BX C H W
        res = self.conv_after_body(res) + x1
        x = x + self.conv_last(res)

        x = x / self.data_range + self.mean
        x = x.view(B, X, C, H, W).permute(0, 2, 3, 4, 1).contiguous()  # B C H W X

        return x

    def flops(self):
        flops = 0
        H, W, Z = self.patches_resolution
        flops += H * W * Z * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * Z * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops




def T25D():
    model = TranfsormerIR_3d(
        data_size=(40, 40, 16),
        patch_size=1,
        in_chans=1,
        embed_dim=32,
        depths=[4],
        num_heads=[2],
        adapter_type='S3D',
        adapter_kernel_size=(3, 3, 3),
        adapter_channels=384,
        ape=True,
        resi_connection='3conv',
        encoder_5_exist=False,
        attn_drop_rate=0.1,
    )
    return model



if __name__ == '__main__':
    # device = torch.device('cuda:0')
    #
    # model = TIR_3d_4TL_drop_0_1().to(device)
    #
    # model = nn.DataParallel(model, device_ids=[0]).to(device)
    #
    # # model_path = './trained_models/pretrained/TIR_3.pth'
    # # model = load_specific_params(model_path, model, keywords=['blocks', 'layers', 'layers'])
    # x = torch.randn(1, 1, 40, 40, 16).to(device)
    # pred = model(x)
    #
    # # 损失函数
    # criterion = nn.MSELoss()
    #
    # with torch.amp.autocast(device_type='cuda'):
    #     loss = criterion(pred, x)
    #     loss.backward()  # 反向传播，计算梯度
    # print(pred.shape)

    from src.utils import count_parameters

    count_parameters(TIR_3d_4TL_drop_0_1())
    count_parameters(TIR_3d_4TL_drop_0_1().layers[0].residual_group.blocks[0])
    count_parameters(TIR_3d_4TL_drop_0_1().layers[0].residual_group.blocks[0].adapter)
