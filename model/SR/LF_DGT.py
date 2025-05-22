from einops import rearrange, repeat
import math
from timm.models.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class get_model(nn.Module):
    def __init__(self, args, mode='train'):
        super().__init__()

        # Predefine common parameters
        self.channels = 64  # Number of channels used across the model
        self.angRes = args.angRes_in  # Angular resolution input
        self.scale = args.scale_factor  # Scale factor for upsampling
        self.num_layers = 6  # Overall network depth (number of ResidualGroups)
        self.layer_depth = 6  # Depth of layers within each ResidualGroup
        self.hidden_sz = 128  # Hidden size (channels) used in MLP layers
        self.inter_sz = 64  # Intermediate size in MLP layers
        self.heads = 4  # Number of attention heads
        self.spa_sr_ratio = 4
        self.attn_drop_rate = 0.0
        self.proj_drop_rate = 0.0
        self.drop_path_rate = 0.1

        # Create drop path rate list according to stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.num_layers * self.layer_depth)]

        # Initial 3D convolutional layers for feature extraction
        self.conv_init0 = nn.Conv2d(1, self.channels, kernel_size=3, padding=1, bias=False)
        self.conv_init = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # List of ResidualGroups, each group has a defined depth and drop_path rate
        self.blocks = nn.ModuleList(
            [FEDD_Block(self.channels,
                        inter_sz=self.inter_sz,
                        hidden_sz=self.hidden_sz,
                        heads=self.heads,
                        layer_depth=self.layer_depth,
                        attn_drop_rate=self.attn_drop_rate,
                        proj_drop_rate=self.proj_drop_rate,
                        spa_sr_ratio=self.spa_sr_ratio,
                        drop_path_rates=dpr[i * self.layer_depth:(i + 1) * self.layer_depth],
                        mode=mode,
                        )
             for i in range(self.num_layers)]
        )

        # Convolutional layers for merging and refinement
        self.conv_refine = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.linear_merge = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, padding=0, bias=False)

        self.upsample = Upsample(self.scale, self.channels)

    def forward(self, lr, disp_info):
        """
        Forward pass for the model.

        Args:
            lr (torch.Tensor): Low-resolution input image.
            disp_info (dict): A dictionary containing disparity indices and masks obtained
                              from the gen_disp_match_idx function.

        Returns:
            torch.Tensor: Super-resolved high-resolution image.
        """
        # Reshape low-resolution image based on angular resolution
        lr = rearrange(lr, 'b c (u h) (v w) -> b c u v h w', u=self.angRes, v=self.angRes)

        # Perform bicubic interpolation for the low-resolution input
        sr_y = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')
        sr_y = rearrange(sr_y, 'b c u v h w -> (b u v) c h w')

        # Apply initial convolutional layers for feature extraction
        x = rearrange(lr, 'b c u v h w -> (b u v) c h w')
        x = self.conv_init0(x)
        x = self.conv_init(x) + x
        x = rearrange(x, '(b u v) c h w -> b c u v h w', u=self.angRes, v=self.angRes)

        x0 = x  # Store original features for later skip connection
        disp_idx_uvhw, mask, disp_map, disp_idx_uvhwd, inv_disp_idx_uvhwd = disp_info.values()
        # disp_all = []
        # Pass through each ResidualGroup
        for block in self.blocks:
            x, disp = block(x, disp_idx_uvhw, mask, disp_map, disp_idx_uvhwd, inv_disp_idx_uvhwd, disp=None)
            # x, disp, disp_block = block(x, disp_idx_uvhw, mask, disp_map, disp_idx_uvhwd, inv_disp_idx_uvhwd, disp=None)
            # disp_all.append(disp_block)
        # disp_all = torch.cat(disp_all, dim=-1)
        # Reshape the output for merging and refinement
        x = rearrange(x, 'b c u v h w -> (b u v) c h w')
        _x0 = rearrange(x0, 'b c u v h w -> (b u v) c h w')
        x = self.linear_merge(torch.cat((self.conv_refine(x), _x0), dim=1))
        x = rearrange(x, '(b u v) c h w -> b c u v h w', u=self.angRes, v=self.angRes)
        x = x + x0  # Skip connection

        # Upsample the final output
        x = rearrange(x, 'b c u v h w -> (b u v) c h w')
        y = self.upsample(x) + sr_y
        y = rearrange(y, '(b u v) c h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)

        return y
        # return y, disp_all


class Upsample(nn.Sequential):
    def __init__(self, scale, channels):
        layers = [
            nn.Conv2d(channels, channels * scale ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        ]
        super(Upsample, self).__init__(*layers)


class FEDD_Block(nn.Module):

    def __init__(   self, 
                    channels, 
                    hidden_sz, 
                    inter_sz,
                    heads=4, 
                    layer_depth=6,
                    attn_drop_rate=0.,
                    proj_drop_rate=0., 
                    spa_sr_ratio=4, 
                    drop_path_rates=None, 
                    mode='train'):
        super().__init__()
        self.layer_depth = layer_depth

        self.blocks = nn.ModuleList([   
            Block(
                    channels=channels, 
                    hidden_sz=hidden_sz, 
                    inter_sz=inter_sz,
                    heads=heads, 
                    attn_drop_rate=attn_drop_rate,
                    proj_drop_rate=proj_drop_rate, 
                    spa_sr_ratio=spa_sr_ratio, 
                    drop_path_rates=drop_path_rates[i], 
                    mode=mode,
                    idx=i
                )for i in range(layer_depth)])

        self.conv_refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.linear_merge = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x, attn_idx, mask, disp_map, attn_idx2, attn_inv, disp=None):

        [b, c, u, v, h, w] = x.size()
        lf_size = [u, v, h, w]
        x0 = x
        x = rearrange(x, 'b c u v h w -> b (u v h w) c')
        # disp_block = []
        for blk in self.blocks:
            x, disp = blk(x, attn_idx, mask, disp_map, attn_idx2, attn_inv, disp, lf_size)
            # x, disp, disp_new = blk(x, attn_idx, mask, disp_map, attn_idx2, attn_inv, disp, lf_size)
            # if disp_new is not None:
            #     disp_block.append(disp_new)
            #     pass
        # disp_block = torch.stack(disp_block, dim=-1)
        _x0 = rearrange(x0, 'b c u v h w -> (b u v) c h w')
        x = rearrange(x, 'b (uv h w) c -> (b uv) c h w', h=h, w=w)
        x = self.linear_merge(torch.cat((self.conv_refine(x), _x0), dim=1))
        x = rearrange(x, '(b u v) c h w -> b c u v h w', u=u, v=v)
        x = x + x0

        return x, disp
        # return x, disp, disp_block




class Block(nn.Module):

    def __init__(self, channels, inter_sz, hidden_sz, heads=4, attn_drop_rate=0.,
                 proj_drop_rate=0., spa_sr_ratio=4, drop_path_rates=None, mode='train', idx=0):
        super().__init__()

        self.idx = idx

        self.norm1 = nn.LayerNorm(channels)

        if idx % 2 == 0:
            self.attn = DA_SA(channels, heads, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate)
        else:
            self.attn = S_SA(channels, heads, attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, sr_ratio=spa_sr_ratio)

        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FFN(inter_sz, hidden_sz, heads, mode)

        # Parameterized scaling factors for residual connections (HAI)
        self.gamma = nn.Parameter(1e-4 * torch.ones(channels), requires_grad=True) 

        # DropPath for each block
        self.drop_path = DropPath(drop_path_rates) if drop_path_rates > 0 else nn.Identity()

    def forward(self, x, attn_idx, mask, disp_map, attn_idx2, attn_inv, disp=None, lf_size=None):

        [u, v, h, w] = lf_size
        # disp_new = None
        _x = x
        if self.idx % 2 == 0:
            disp, attn_out = self.attn(self.norm1(x), attn_idx, mask, disp_map, attn_idx2, attn_inv, disp, u, v, h, w)
            x = x + self.drop_path(attn_out)
            # disp_new = rearrange(disp.squeeze(0), '(u v h w) head -> u v h w head', u=u, v=v, h=h)[2,2,...]
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), disp, uv=u * v, h=h))

        x = x + self.drop_path(self.ffn(self.norm2(x), disp, h, w))
        x = x + _x * self.gamma

        return x, disp
        # return x, disp, disp_new




class DA_SA(nn.Module):
    def __init__(self, dim, heads, attn_drop=0., proj_drop=0., qkv_bias=False, proj_bias=False):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.h_c = dim // heads
        self.h = heads

        # Linear layers for query, key, and value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Rotary positional embedding for query and key
        self.apply_rotary_emb = RotaryEmb(heads, self.h_c)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_idx, mask, disp_map, attn_idx2, attn_inv, disp_before, u, v, h, w):
        b, n, _ = x.shape

        qkv = self.qkv(x).reshape(b, -1, 3, self.dim).permute(2, 0, 1, 3)

        query = qkv[0].view(b, n, self.h, self.h_c // 2, 2)  # b (u v h w) head hc/2 2
        key = qkv[1].view(b, n, self.h, self.h_c // 2, 2)
        value = qkv[2].view(b, n, self.h, self.h_c)

        if disp_before is not None:
            query, key = self.apply_rotary_emb(query, key, disp_before, h, w, u, v, b)

        query, key = query.flatten(-2).type_as(value), key.flatten(-2).type_as(value)
        disp_size = disp_map.shape[0] // (u * v)

        qkv = torch.stack((query, key, value), dim=0)
        uv = u * v
        attn_out, disp = DisparityAngularAttention.apply(qkv, attn_idx, attn_idx2, attn_inv, mask, disp_map, disp_size,
                                                         uv)

        #######################
        # This section of code represents the forward pass of the DotFunc.
        # It calculates the attention weights and the corresponding output.
        # The backward pass is handled by PyTorch's built-in autograd mechanism.
        # Enabling this code will consume additional GPU memory, so it should
        # only be activated when testing or profiling parameter sizes.

        # _, b, n, h, hc = qkv.shape
        # qkv_segment = qkv[:, :, attn_idx, ...]  # 3 b (u v h w) head hc
        # query_segment, key_segment, value_segment = qkv_segment
        #
        # # Rearrange tensors for matrix multiplication
        # query_segment = rearrange(query_segment, 'b (disp uv hw) head hc -> b disp hw head uv hc', disp=disp_size,
        #                           uv=uv)
        # key_segment = rearrange(key_segment, 'b (disp uv hw) head hc -> b disp hw head hc uv', disp=disp_size, uv=uv)
        #
        # # Compute attention weights
        # scale_factor = 1 / math.sqrt(query_segment.size(-1))
        # attn_weight = query_segment @ key_segment * scale_factor
        #
        # # Rearrange and apply mask to attention weights
        # attn_weight = rearrange(attn_weight, 'b disp hw head uv1 uv2 -> b (disp uv1 hw) head uv2')
        # attn_weight = attn_weight[:, attn_inv, ...]
        # attn_weight = rearrange(attn_weight, 'b (disp uv1hw) head uv2 -> b uv1hw head (uv2 disp)', disp=disp_size)
        #
        # attn_bias = torch.zeros_like(mask, dtype=attn_weight.dtype, device=attn_weight.device)
        # attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        # attn_weight = attn_weight + attn_bias[None, :, None, :]
        # if torch.is_autocast_enabled():
        #     attn_weight_max = torch.max(attn_weight, dim=-1, keepdim=True)[0]
        #     attn_weight = attn_weight - attn_weight_max
        # attn_weight = F.softmax(attn_weight, dim=-1, dtype=attn_weight.dtype)
        #
        # # Compute disparity
        # disp = torch.sum(attn_weight * disp_map[None, None, None, :], dim=-1, dtype=attn_weight.dtype)  # b uvhw head
        #
        # # Rearrange attention output
        # attn_weight = rearrange(attn_weight, 'b uv1hw head (uv2 disp) -> b (disp uv1hw) head uv2', uv2=uv)
        # attn_weight = attn_weight[:, attn_idx2, ...]
        # attn_weight = rearrange(attn_weight, 'b (disp uv1 hw) head uv2 -> b disp hw head uv1 uv2', disp=disp_size,
        #                         uv1=uv)
        #
        # # Apply attention to value tensor
        # value_segment = rearrange(value_segment, 'b (disp uv hw) head hc -> b disp hw head uv hc', disp=disp_size,
        #                           uv=uv)
        # _attn_out = attn_weight @ value_segment  # b disp hw head uv hc
        #
        # # Rearrange attention output
        # _attn_out = rearrange(_attn_out, 'b disp hw head uv hc -> b (disp uv hw) (head hc)')
        #
        # attn_out = torch.zeros((b, n, h * hc), device=qkv.device, dtype=qkv.dtype)
        # attn_out = torch.Tensor.index_add_(attn_out, dim=1, index=attn_idx, source=_attn_out)
        #######################

        attn_out = self.attn_drop(attn_out)

        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        return disp, attn_out


class DisparityAngularAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        qkv, attn_idx, attn_idx2, attn_inv, mask, disp_map, disp_size, uv = args
        _, b, n, h, hc = qkv.shape
        qkv_segment = qkv[:, :, attn_idx, ...]  # 3 b (u v h w) head hc
        query_segment, key_segment, value_segment = qkv_segment

        # Rearrange tensors for matrix multiplication
        query_segment = rearrange(query_segment, 'b (disp uv hw) head hc -> b disp hw head uv hc', disp=disp_size,
                                  uv=uv)
        key_segment = rearrange(key_segment, 'b (disp uv hw) head hc -> b disp hw head hc uv', disp=disp_size, uv=uv)

        # Compute attention weights
        scale_factor = 1 / math.sqrt(query_segment.size(-1))
        attn_weight = query_segment @ key_segment * scale_factor

        # Rearrange and apply mask to attention weights
        attn_weight = rearrange(attn_weight, 'b disp hw head uv1 uv2 -> b (disp uv1 hw) head uv2')
        attn_weight = attn_weight[:, attn_inv, ...]
        attn_weight = rearrange(attn_weight, 'b (disp uv1hw) head uv2 -> b uv1hw head (uv2 disp)', disp=disp_size)

        attn_bias = torch.zeros_like(mask, dtype=attn_weight.dtype, device=attn_weight.device)
        attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        attn_weight = attn_weight + attn_bias[None, :, None, :]
        if torch.is_autocast_enabled():
            attn_weight_max = torch.max(attn_weight, dim=-1, keepdim=True)[0]
            attn_weight = attn_weight - attn_weight_max
        attn_weight = attn_weight.softmax(dim=-1).to(qkv.dtype)
        # Compute disparity
        disp = torch.sum(attn_weight * disp_map[None, None, None, :], dim=-1)  # b uvhw head

        # Rearrange attention output
        attn_weight = rearrange(attn_weight, 'b uv1hw head (uv2 disp) -> b (disp uv1hw) head uv2', uv2=uv)
        attn_weight = attn_weight[:, attn_idx2, ...]
        attn_weight = rearrange(attn_weight, 'b (disp uv1 hw) head uv2 -> b disp hw head uv1 uv2', disp=disp_size,
                                uv1=uv)

        # Apply attention to value tensor
        value_segment = rearrange(value_segment, 'b (disp uv hw) head hc -> b disp hw head uv hc', disp=disp_size,
                                  uv=uv)
        _attn_out = attn_weight @ value_segment  # b disp hw head uv hc

        # Rearrange attention output
        _attn_out = rearrange(_attn_out, 'b disp hw head uv hc -> b (disp uv hw) (head hc)')

        attn_out = torch.zeros((b, n, h * hc), device=qkv.device, dtype=qkv.dtype)
        attn_out = torch.Tensor.index_add_(attn_out, dim=1, index=attn_idx, source=_attn_out)

        # Save context for backward pass
        ctx.save_for_backward(qkv, attn_weight)
        if not hasattr(ctx, 'disp_size'):
            ctx.disp_size = disp_size
            ctx.uv = uv
            ctx.attn_idx, ctx.attn_idx2, ctx.attn_inv, ctx.disp_map = attn_idx, attn_idx2, attn_inv, disp_map

        return attn_out, disp

    @staticmethod
    def backward(ctx, *grad_outputs):

        grad_attn_out, grad_disp = grad_outputs
        qkv, attn_weight = ctx.saved_tensors

        _, b, n, h, hc = qkv.shape
        qkv_segment = qkv[:, :, ctx.attn_idx, ...]  # 3 b (u v h w) head hc
        query_segment, key_segment, value_segment = qkv_segment

        # Rearrange grad_attn_out for processing
        grad_attn_out = grad_attn_out[:, ctx.attn_idx, ...]
        grad_attn_out = rearrange(grad_attn_out, 'b (disp uv hw) (head hc) -> b disp hw head uv hc',
                                  disp=ctx.disp_size, uv=ctx.uv, head=h)

        value_segment = rearrange(value_segment, 'b (disp uv hw) head hc -> b disp hw head hc uv',
                                  disp=ctx.disp_size, uv=ctx.uv)

        # Compute gradients for attention weights and value
        grad_attn_weight = grad_attn_out @ value_segment
        grad_value_segment = attn_weight.transpose(-2, -1) @ grad_attn_out

        grad_value_segment = rearrange(grad_value_segment, 'b disp hw head uv hc -> b (disp uv hw) head hc')

        _grad_attn_weight = torch.stack((grad_attn_weight, attn_weight), dim=0)
        _grad_attn_weight = rearrange(_grad_attn_weight, 'c b disp hw head uv1 uv2 -> c b (disp uv1 hw) head uv2')
        _grad_attn_weight = _grad_attn_weight[:, :, ctx.attn_inv, ...]
        _grad_attn_weight = rearrange(_grad_attn_weight, 'c b (disp uv1hw) head uv2 -> c b uv1hw head (uv2 disp)',
                                      disp=ctx.disp_size)
        grad_attn_weight, attn_weight = _grad_attn_weight

        grad_attn_weight += grad_disp[..., None] * ctx.disp_map[None, None, None, :]

        # Backpropagation through softmax
        grad_attn_weight = attn_weight * (
                torch.sum(-(attn_weight * grad_attn_weight), dim=-1, keepdim=True) + grad_attn_weight)

        grad_attn_weight = rearrange(grad_attn_weight, 'b uv1hw head (uv2 disp) -> b (disp uv1hw) head uv2', uv2=ctx.uv)
        grad_attn_weight = grad_attn_weight[:, ctx.attn_idx2, ...]
        grad_attn_weight = rearrange(grad_attn_weight, 'b (disp uv1 hw) head uv2 -> b disp hw head uv1 uv2',
                                     disp=ctx.disp_size, uv1=ctx.uv)

        scale_factor = 1 / math.sqrt(query_segment.size(-1))
        grad_attn_weight = grad_attn_weight * scale_factor

        query_segment = rearrange(query_segment, 'b (disp uv hw) head hc -> b disp hw head hc uv', disp=ctx.disp_size,
                                  uv=ctx.uv)
        key_segment = rearrange(key_segment, 'b (disp uv hw) head hc -> b disp hw head uv hc', disp=ctx.disp_size,
                                uv=ctx.uv)

        # Compute gradients for query and key
        grad_query_segment = grad_attn_weight @ key_segment
        grad_key_segment = query_segment @ grad_attn_weight

        grad_query_segment = rearrange(grad_query_segment, 'b disp hw head uv hc -> b (disp uv hw) head hc')
        grad_key_segment = rearrange(grad_key_segment, 'b disp hw head hc uv -> b (disp uv hw) head hc')
        grad_qkv_segment = torch.stack((grad_query_segment, grad_key_segment, grad_value_segment), dim=0)

        grad_qkv = torch.zeros_like(qkv)
        grad_qkv = torch.Tensor.index_add_(grad_qkv, dim=2, index=ctx.attn_idx, source=grad_qkv_segment)

        return grad_qkv, None, None, None, None, None, None, None


class RotaryEmb(nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.position = nn.Linear(heads * 2, heads * head_dim // 2, bias=False)

    def forward(self, query, key, disp_before, h, w, u, v, batches, theta=1000.0):
        hi, wi = torch.meshgrid(torch.linspace(1, h, h, device=key.device, dtype=query.dtype),
                                torch.linspace(1, w, w, device=key.device, dtype=query.dtype),
                                indexing='ij')

        ui, vi = torch.meshgrid(torch.linspace(1, u, u, device=key.device, dtype=query.dtype),
                                torch.linspace(1, v, v, device=key.device, dtype=query.dtype),
                                indexing='ij')

        n = self.head_dim // 2

        # Compute positional embeddings for height, width, u, v
        hwi = torch.stack((hi, wi), dim=-1)
        uvi = torch.stack((ui, vi), dim=-1)
        hwi = repeat(hwi, 'h w c -> b (u v h w) head c', b=batches, u=u, v=v, head=self.heads)
        uvi = repeat(uvi, 'u v c -> b (u v h w) head c', b=batches, h=h, w=w, head=self.heads)
        with torch.cuda.amp.autocast(enabled=False):
            pos = (hwi - uvi * (disp_before[..., None])).flatten(-2, -1)
            pos = self.position(pos)
            pos = rearrange(pos, 'b n (head hc2) -> b n head hc2', head=self.heads)
            pos_freqs = 1.0 / (theta ** (torch.arange(0, n, 2, device=pos.device) / n))
            pos_freqs = repeat(pos_freqs, 'hc4 -> b n head (hc4 2)', b=batches, n=pos.shape[1], head=self.heads)
            query, key = query.type_as(pos_freqs), key.type_as(pos_freqs)

            # Apply rotary embeddings to query and key
            freqs_cis = torch.polar(torch.ones_like(pos), pos * pos_freqs)

            query, key = torch.view_as_complex(query), torch.view_as_complex(key)
            query = torch.view_as_real(query * freqs_cis)
            key = torch.view_as_real(key * freqs_cis)

        return query, key


class S_SA(nn.Module):
    def __init__(self, dim, heads, attn_drop=0., proj_drop=0., sr_ratio=4, qkv_bias=False):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.sr_disp = nn.Conv2d(heads, dim * 2, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
        self.sr = nn.Conv2d(dim, dim * 2, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)
        self.sr2 = nn.Linear(dim * 2, dim)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, disp, uv, h):

        x = rearrange(x, 'b (uv hw) c -> (b uv) hw c', uv=uv)
        b, n, c = x.shape

        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        x_ = x.permute(0, 2, 1).reshape(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
        disp = rearrange(disp, 'b (uv h w) head -> (b uv) head h w', uv=25, h=32)
        disp = 1 - abs(self.sr_disp(disp))
        x_ = (self.sr(x_) * disp).reshape(b, c * 2, -1).permute(0, 2, 1)
        x_ = self.norm(self.sr2(x_))
        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if torch.is_autocast_enabled():
            attn_max = torch.max(attn, dim=-1, keepdim=True)[0]
            attn = attn - attn_max
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = rearrange(x, '(b uv) hw c -> b (uv hw) c', uv=uv)

        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, heads: int, mode='train', act_fn=nn.GELU,
                 drop=0., ffn_bias=False):
        super().__init__()
        self.mode = mode

        if self.mode == 'train':
            # In training mode, the module includes a linear layer and a convolution layer
            self.linear_in = nn.Linear(in_features, hidden_features, bias=ffn_bias)
            self.spatial_proj = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, bias=ffn_bias)
            self.equivalent_layer = None
        elif self.mode == 'eval':
            # In evaluation mode, load the pre-trained equivalent layer
            self.equivalent_layer = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1, bias=ffn_bias)

        self.disp_gate_conv = nn.Conv2d(heads, hidden_features, kernel_size=3, padding=1, bias=False)
        self.linear_out = nn.Linear(hidden_features, in_features, bias=ffn_bias)

        self.act_fn = act_fn()
        self.drop = nn.Dropout(drop)

    def forward(self, x, disp, h, w):

        b = x.shape[0]
        if self.mode == 'train':
            # In training, use the linear layer followed by the convolution layer
            x = self.linear_in(x)
            x = rearrange(x, 'b (uv h w) c -> (b uv) c h w', h=h, w=w)
            x = self.spatial_proj(x)
        elif self.mode == 'eval':
            # In evaluation, use the equivalent layer
            x = rearrange(x, 'b (uv h w) c -> (b uv) c h w', h=h, w=w)
            x = self.equivalent_layer(x)

        x = self.act_fn(x)
        disp = rearrange(disp, 'b (uv h w) head -> (b uv) head h w', h=h, w=w)
        x = x * (1 - abs(self.disp_gate_conv(disp)))
        x = rearrange(x, '(b uv) c h w -> b (uv h w) c', b=b)
        x = self.drop(x)

        x = self.linear_out(x)
        x = self.drop(x)

        return x


def create_equivalent_layers_recursively(module):
    """Recursively create equivalent layers for all MLP instances inside the network."""
    for name, sub_module in module.named_children():

        if isinstance(sub_module, FFN):
            create_equivalent_layer(sub_module)  # Call the equivalent layer creation logic
        else:
            # If it's a nested module, call recursively
            create_equivalent_layers_recursively(sub_module)


def create_equivalent_layer(self):
    """Create the equivalent layer by combining the parameters of the linear and conv layers"""
    if self.linear_in is None or self.spatial_proj is None:
        raise ValueError("Original layers must exist before creating equivalent layer.")

    with torch.no_grad():
        # Create the equivalent layer by combining linear and conv layer parameters
        in_channels = self.linear_in.in_features
        out_channels = self.spatial_proj.out_channels
        if self.linear_in.bias is not None:
            self.equivalent_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            # Compute the bias of the equivalent layer by combining the biases of the linear and conv layers
            linear_bias_expanded = self.linear_in.bias.view(1, -1, 1, 1)
            self.equivalent_layer.bias.copy_(
                self.spatial_proj.bias + (self.spatial_proj.weight * linear_bias_expanded).sum(dim=[1, 2, 3]))
        else:
            self.equivalent_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        # Compute the weights of the equivalent layer by combining the weights of the linear and conv layers
        self.equivalent_layer.weight.copy_(
            (self.spatial_proj.weight.permute(0, 2, 3, 1) @ self.linear_in.weight).permute(0, 3, 1, 2))

    del self.linear_in
    del self.spatial_proj

    self.mode = 'eval'


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR):
        loss = self.criterion_Loss(out, HR)

        return loss


def LF_interpolate(LF, scale_factor, mode):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_upscale = F.interpolate(LF, scale_factor=scale_factor, mode=mode, align_corners=False)
    LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_upscale


def weights_init(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        pass
    elif isinstance(m, nn.Conv3d):
        pass

def gen_disp_match_idx(u: int, v: int, h: int, w: int, scale_factor: int):
    """Generate disparity‑related lookup indices for multi‑view matching.

    Parameters
    ----------
    u, v : int
        Number of sub‑aperture images along the vertical (``u``) and horizontal
        (``v``) axes.
    h, w : int
        Height and width **per** sub‑aperture image in pixels.
    scale_factor : int {2, 4}
        A value of ``2`` yields *five* disparity planes (\[0, ±1, ±2\]) and
        ``4`` yields *three* planes (\[0, ±1\]).

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary with the following keys:

        * ``disp_idx_uvhw`` – Flat indices mapping (u, v, h, w) → 1‑D.
        * ``mask`` – Boolean visibility mask for valid disparities.
        * ``disp`` – 1‑D tensor of disparity values repeated for every view.
        * ``disp_idx_uvhwd`` – Lookup from (disp, u, v, h, w) → flat index.
        * ``inv_disp_idx_uvhwd`` – Inverse mapping of the above.
    """

    # ---------------------------------------------------------------------
    # Basic disparity configuration
    # ---------------------------------------------------------------------
    if scale_factor == 2:
        # Five disparity planes: 0, ±1, ±2
        disp_radius = 2
        disp_table = [0, 6, 12, 18, 24]
    elif scale_factor == 4:
        # Three disparity planes: 0, ±1
        disp_radius = 1
        disp_table = [0, 6, 12]
    else:
        raise ValueError("Unsupported 'scale_factor' – expected 2 or 4.")

    # Size helpers
    full_h = (u * 2 - 1) * h
    full_w = (v * 2 - 1) * w

    # Allocation of global disparity and mask maps
    mask_map = torch.zeros((full_h, full_w), dtype=torch.int64, device=device)
    disp_map = torch.zeros((full_h, full_w), dtype=torch.float32, device=device)

    # Centre coordinates of the reference view within the mosaic grid
    h_mid = h // 2 - 1
    w_mid = w // 2 - 1
    u_mid = u - 1
    v_mid = v - 1

    # ------------------------------------------------------------------
    # Construct dense disparity & mask mosaics
    # ------------------------------------------------------------------
    for iu in range(u * 2 - 1):
        for iv in range(v * 2 - 1):
            dh = iu - u_mid  # vertical disparity between current SA‑I and ref
            dw = iv - v_mid  # horizontal disparity between current SA‑I and ref

            ref_row = iu * h + h_mid
            ref_col = iv * w + w_mid

            if dh == 0 and dw == 0:
                # Reference view (zero disparity)
                mask_map[ref_row, ref_col] = 1
                disp_map[ref_row, ref_col] = 0.0
                continue

            # Walk along the epipolar line for the current (dh, dw)
            for d in range(-disp_radius, disp_radius + 1):
                row = ref_row + d * dh
                col = ref_col + d * dw
                mask_map[row, col] = 1
                disp_map[row, col] = float(d)

                # Fill intermediate integer grid points for oblique lines
                if d < disp_radius:
                    if abs(dh) > abs(dw):
                        # Predominantly vertical epipolar line
                        for k_h in range(1, abs(dh)):
                            k_w = k_h * dw / abs(dh)
                            if k_w.is_integer():
                                k_w = int(k_w)
                                k_h = int(math.copysign(k_h, dh))
                                mask_map[row + k_h, col + k_w] = 1
                                disp_map[row + k_h, col + k_w] = d + k_h / dh
                    elif abs(dw) > 0:
                        # Predominantly horizontal epipolar line
                        for k_w in range(1, abs(dw)):
                            k_h = k_w * dh / abs(dw)
                            if k_h.is_integer():
                                k_h = int(k_h)
                                k_w = int(math.copysign(k_w, dw))
                                mask_map[row + k_h, col + k_w] = 1
                                disp_map[row + k_h, col + k_w] = d + k_w / dw

    # Keep only entries defined by 'disp_table' and broadcast over views
    disp_unique = torch.unique(disp_map)[disp_table]
    disp_planes = disp_unique.shape[0]

    # ------------------------------------------------------------------
    # Per‑view mesh‑grid indices (hi, wi)
    # ------------------------------------------------------------------
    hi, wi = torch.meshgrid(
        torch.arange(-h_mid, h - h_mid, dtype=torch.int64, device=device),
        torch.arange(-w_mid, w - w_mid, dtype=torch.int64, device=device),
        indexing="ij",
    )

    mask = torch.zeros(
        h * w, u * v, u * v, disp_planes, dtype=torch.int64, device=device
    )
    disp_idx_uvhw = torch.zeros(
        h * w, u * v, disp_planes, dtype=torch.int64, device=device
    )

    # ------------------------------------------------------------------
    # Build lookup tables for every disparity plane
    # ------------------------------------------------------------------
    for d_idx in range(disp_planes):
        disp_val = disp_unique[d_idx]

        if disp_val == 0:
            flat_idx = torch.arange(u * v, device=device)
            flat_idx = (flat_idx * h * w).unsqueeze(0) + torch.arange(h * w, device=device)[:, None]

            mask[..., d_idx] = 1
            disp_idx_uvhw[..., d_idx] = flat_idx
            continue

        # Non‑zero disparity – build per‑view mask and index tensors
        for iu in range(u):
            for iv in range(v):
                # Extract disparity patch centred at current (iu, iv)
                disp_patch = disp_map[
                    (u_mid - iu) * h : (u_mid - iu + u) * h,
                    (v_mid - iv) * w : (v_mid - iv + v) * w,
                ]
                disp_patch = rearrange(disp_patch, "(u h) (v w) -> u v h w", u=u, v=v)

                coords = torch.nonzero(disp_patch == disp_val)
                temp_index = torch.zeros(u * v, 4, dtype=torch.int64, device=device)
                temp_mask = torch.zeros(u * v, dtype=torch.int64, device=device)

                if coords.numel() > 0:
                    temp_index[coords[:, 0] * v + coords[:, 1]] = coords
                    temp_mask[coords[:, 0] * v + coords[:, 1]] = 1

                # Broadcast (h, w) offsets
                temp_index = temp_index.expand(*hi.shape, -1, -1).contiguous()
                temp_mask = temp_mask.expand(*hi.shape, -1).contiguous()
                temp_index[..., 2] += hi[..., None]
                temp_index[..., 3] += wi[..., None]

                temp_index = rearrange(temp_index, "h w n c -> (h w n) c")
                temp_mask = rearrange(temp_mask, "h w n -> (h w n)")

                # Cull out‑of‑image indices
                invalid = (
                    (temp_index[:, 2] >= h)
                    | (temp_index[:, 2] < 0)
                    | (temp_index[:, 3] < 0)
                    | (temp_index[:, 3] >= w)
                )
                temp_mask[invalid] = 0

                # Flatten to single integer index
                flat = (
                    temp_index[:, 0] * v * h * w
                    + temp_index[:, 1] * h * w
                    + temp_index[:, 2] * w
                    + temp_index[:, 3]
                )
                flat *= temp_mask  # zero‑out invalid positions

                flat = rearrange(flat, "(h w n) -> (h w) n", h=h, w=w)
                msk = rearrange(temp_mask, "(h w n) -> (h w) n", h=h, w=w)

                view_id = iu * v + iv
                flat[:, view_id] = view_id * h * w + torch.arange(h * w, device=device)

                mask[:, view_id, :, d_idx] = msk
                if view_id == 0:
                    disp_idx_uvhw[:, :, d_idx] = flat
                else:
                    offset = view_id * h * w
                    missing = (
                        torch.arange(offset, offset + h * w, device=device)
                        .unsqueeze(1)
                        .repeat(1, 2)
                    )
                    unused = torch.tensor(
                        list(
                            set(missing[:, 0].tolist())
                            - set(torch.unique(disp_idx_uvhw[:, view_id, d_idx]).tolist())
                        ),
                        device=device,
                    ) - offset

                    disp_idx_uvhw[disp_idx_uvhw[:, view_id, d_idx] == 0, :, d_idx] += flat[unused, :]

    # ------------------------------------------------------------------
    # Final reshaping and auxiliary tables
    # ------------------------------------------------------------------
    disp_idx_uvhw = rearrange(disp_idx_uvhw, "hw uv d -> d (uv hw)")
    inv_disp_idx_uvhw = torch.sort(disp_idx_uvhw, dim=1)[1]

    disp_idx_uvhwd = torch.arange(u * v * h * w * disp_planes, device=device).view(
        disp_planes, u * v * h * w
    )
    inv_disp_idx_uvhwd = disp_idx_uvhwd.clone()

    disp_idx_uvhwd = torch.stack(
        [torch.index_select(disp_idx_uvhwd[i], 0, disp_idx_uvhw[i]) for i in range(disp_planes)],
        dim=0,
    )
    inv_disp_idx_uvhwd = torch.stack(
        [torch.index_select(inv_disp_idx_uvhwd[i], 0, inv_disp_idx_uvhw[i]) for i in range(disp_planes)],
        dim=0,
    )

    disp_idx_uvhwd = disp_idx_uvhwd.flatten()
    inv_disp_idx_uvhwd = inv_disp_idx_uvhwd.flatten()
    disp_idx_uvhw = disp_idx_uvhw.flatten()

    mask = rearrange(mask, "hw v1 v2 d -> (v1 hw) (v2 d)").bool()
    disp = disp_unique.expand(u * v, disp_planes).flatten()

    return {
        "disp_idx_uvhw": disp_idx_uvhw,
        "mask": mask,
        "disp": disp,
        "disp_idx_uvhwd": disp_idx_uvhwd,
        "inv_disp_idx_uvhwd": inv_disp_idx_uvhwd,
    }

