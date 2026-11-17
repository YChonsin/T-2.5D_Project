
import torch.nn as nn



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple,
                 patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # print(patches_resolution)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple,
                 patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class PatchEmbed_3d(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple,
                 patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # data_size = (data_size, data_size, data_size)
        patch_size = (patch_size, patch_size, patch_size)
        resolution = [img_size[0] // patch_size[0],
                      img_size[1] // patch_size[1],
                      img_size[2] // patch_size[2]]
        self.data_size = img_size
        self.block_size = patch_size
        self.patches_resolution = resolution
        self.num_patches = resolution[0] * resolution[1] * resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W, Z = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # B, Ph*Pw*Pz, C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W, Z = self.data_size
        if self.norm is not None:
            flops += H * W * Z * self.embed_dim
        return flops


class PatchUnEmbed_3d(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple,
                 patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # data_size = (data_size, data_size, data_size)
        patch_size = (patch_size, patch_size, patch_size)
        resolution = [img_size[0] // patch_size[0],
                      img_size[1] // patch_size[1],
                      img_size[2] // patch_size[2]]
        self.data_size = img_size
        self.block_size = patch_size
        self.resolution = resolution
        self.num_patches = resolution[0] * resolution[1] * resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, L, C = x.shape  # B, Ph*Pw*Pz, C
        x = x.permute(0, 2, 1).view(B, self.embed_dim,
                                    x_size[0], x_size[1], x_size[2])
        # B, C, H, W, Z
        return x

    def flops(self):
        flops = 0
        return flops

if __name__ == '__main__':
    pass