import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        # print(x.shape)
        # exit()
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 init_cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def channel_to_last(x):
    """
    Args:
        x: (B, C, H, W, D)

    Returns:
        x: (B, H, W, D, C)
    """
    return x.permute(0, 2, 3, 1)


def channel_to_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 3, 1, 2)










class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        初始化 ChannelReducer 模块

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数（减小后的通道数）
        """
        super(ChannelReducer, self).__init__()
        # 逐点卷积，用于减少通道数
        # print(in_channels)
        # print(out_channels)
        # exit()
        out_channels = int(out_channels)
        self.reduce_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        # 批归一化层
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        前向传播

        :param x: 输入特征图
        :return: 经过通道减少后的特征图
        """
        # 逐点卷积，减少通道数
        x = self.reduce_conv(x)
        # 批归一化
        x = self.bn(x)
        return x


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 ca_num_heads=4,
                 sa_num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ca_attention=1,
                 expand_ratio=2,
                 init_cfg=None):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)

        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x, H, W):

        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out, s_i], 2)

            s_out = s_out.reshape(B, C, H, W)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out)))).reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v
        if self.ca_attention == 2:
            # x1 = input.permute(0, 3, 1, 2)
            # B, C, H, W = x1.shape
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                          N).transpose(
                    1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class SC(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=3, expand=2):
        """
        Args:
            in_channels: 输入特征F的通道数
            out_channels: 输出特征F‘的通道数
            d_state, d_conv, expand: SS2D的参数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels  # 中间通道数，简化设计，可与out_channels相同

        # 1. 生成多尺度特征: F, F_s2, F_s4
        # 使用步长为2和4的卷积进行下采样，模拟区域聚合
        self.downsample_s2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.downsample_s4 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=4, padding=2, groups=in_channels, bias=False)
        self.norm_s2 = nn.BatchNorm2d(in_channels)
        self.norm_s4 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)

        # 2. 通道投影层 (论文中的1x1 convs)，将不同尺度特征投影到统一通道数
        self.proj_f = nn.Conv2d(in_channels * 16, self.mid_channels, kernel_size=1)  # PixelUnshuffle 后通道数*16
        self.proj_s2 = nn.Conv2d(in_channels * 4, self.mid_channels, kernel_size=1) # PixelUnshuffle 后通道数*4
        self.proj_s4 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1)

        # 3. 核心：SS2D 模块，用于多尺度上下文扫描
        # 注意：输入到SS2D的通道数是 3 * self.mid_channels
        self.ss2d = SS2D(
            d_model=3 * self.mid_channels,  # 因为拼接了三个尺度的特征
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 4. 跨尺度融合的1x1卷积
        self.fuse_conv = nn.Conv2d(3 * self.mid_channels, 3 * self.mid_channels, kernel_size=1)

        # 5. 上采样和最终投影到输出通道
        # 输出空间尺寸上采样回 H/8 * W/8
        self.up_sample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 从H/32上采样到H/8
        self.final_proj = nn.Conv2d(3 * self.mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: 输入特征 F, 形状为 (B, C, H/8, W/8)
        返回: 多尺度上下文特征 F‘, 形状为 (B, out_channels, H/8, W/8)
        """
        B, C, H, W = x.shape
        # 1. 创建多尺度特征
        f_s2 = self.act(self.norm_s2(self.downsample_s2(x)))   # (B, C, H/16, W/16)
        f_s4 = self.act(self.norm_s4(self.downsample_s4(x)))   # (B, C, H/32, W/32)

        # 2. 使用 PixelUnshuffle 进行无损下采样，统一空间尺寸到 (H/32, W/32)
        # 对 F: 从 (H/8, W/8) -> (H/32, W/32), 通道变为 C*16
        f_unshuffle = F.pixel_unshuffle(x, 4)  # (B, C*16, H/32, W/32)
        # 对 F_s2: 从 (H/16, W/16) -> (H/32, W/32), 通道变为 C*4
        f_s2_unshuffle = F.pixel_unshuffle(f_s2, 2)  # (B, C*4, H/32, W/32)
        # F_s4 已经是 (H/32, W/32)，保持不变

        # 3. 通道投影
        f_proj = self.proj_f(f_unshuffle)        # (B, mid_C, H/32, W/32)
        f_s2_proj = self.proj_s2(f_s2_unshuffle) # (B, mid_C, H/32, W/32)
        f_s4_proj = self.proj_s4(f_s4)           # (B, mid_C, H/32, W/32)

        # 4. 沿通道维度拼接
        multi_scale_feat = torch.cat([f_proj, f_s2_proj, f_s4_proj], dim=1)  # (B, 3*mid_C, H/32, W/32)

        # 5. 调整维度以适配 SS2D: (B, C, H, W) -> (B, H, W, C)
        multi_scale_feat = multi_scale_feat.permute(0, 2, 3, 1)

        # 6. SS2D 扫描
        context_feat = self.ss2d(multi_scale_feat)  # (B, H/32, W/32, 3*mid_C)

        # 7. 跨尺度融合 (1x1 conv)
        context_feat = context_feat.permute(0, 3, 1, 2)  # 改回 (B, C, H, W) 格式
        context_feat = self.fuse_conv(context_feat)

        # 8. 上采样回原始分辨率 (H/8, W/8) 并投影到目标通道
        context_feat = self.up_sample(context_feat)  # (B, 3*mid_C, H/8, W/8)
        out_feat = self.final_proj(context_feat)     # (B, out_channels, H/8, W/8)

        return out_feat

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            mlp_ratio: int = 0,
            ca_attetion: int = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            # i: int =0,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.ln_2 = norm_layer(hidden_dim)
        self.ln_3 = norm_layer(hidden_dim)
        self.attn = Attention(dim=hidden_dim, ca_attention=ca_attetion)
        # print(hidden_dim)
        # exit()
        self.ca = ca_attetion
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # self.mlp = Mlp(
        #    in_features=hidden_dim,
        #    hidden_features=mlp_hidden_dim,
        #    act_layer=nn.GELU,
        #    drop=[0,0],
        #    init_cfg=None)

    def forward(self, input: torch.Tensor):
        # print(input.shape)
        # exit()
        if self.ca == 1 or self.ca == 2:
            x1 = input.permute(0, 3, 1, 2)
            B, C, H1, W1 = x1.shape
            x1 = x1.flatten(2).transpose(1, 2)
            x = x1 + self.drop_path(self.attn(self.ln_2(x1), H1, W1, self.ca))
            x = x.transpose(1, 2).view(B, C, H1, W1)
            x = x.permute(0, 2, 3, 1)

        else:
            x = input + self.drop_path(self.self_attention(self.ln_1(input)))
            # x = 0
        # x = x + self.drop_path(self.mlp(self.ln_3(x), 1, W1))
        # x = x + self.drop_path(self.self_attention(self.ln_1(x)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            mlp_ratio=0,
            ca_attention=0,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                mlp_ratio=mlp_ratio,
                ca_attention=ca_attention,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        # print(drop_path[1])
        # exit()
        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            mlp_ratio=0,
            ca_attention=0,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                mlp_ratio=mlp_ratio,
                ca_attention=ca_attention,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)






class SegMAN_Decoder_4Branch(nn.Module):
    def __init__(self, encoder_channels, embed_dim=256, num_classes=150):
        """
        四分支SegMAN解码器
        Args:
            encoder_channels: 列表，编码器四个阶段输出特征的通道数 [C2, C3, C4, C5]
                              C2: (H/8, W/8)  - 第2阶段
                              C3: (H/16, W/16) - 第3阶段
                              C4: (H/32, W/32) - 第4阶段
                              C5: (H/32, W/32) - 第5阶段
            embed_dim: 解码器内部统一投影的维度
            num_classes: 分割类别数
        """
        super().__init__()
        c2, c3, c4, c5 = encoder_channels  # 四个阶段的通道数

        # 1. 特征聚合: 将四个特征投影到统一通道数
        self.reduce_conv_c2 = nn.Sequential(
            nn.Conv2d(c2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.reduce_conv_c3 = nn.Sequential(
            nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.reduce_conv_c4 = nn.Sequential(
            nn.Conv2d(c4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.reduce_conv_c5 = nn.Sequential(
            nn.Conv2d(c5, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 特征融合卷积 (生成F)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),  # 拼接了四个特征
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # 3.
        self.sc = SC(in_channels=embed_dim, out_channels=embed_dim)

        # 4. 多尺度融合后的预测头
        # 全局平均池化分支
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # 最终的MLP预测头
        # 输入: F' + (f2_enhanced) + (f3_enhanced) + (f4_enhanced) + (f5_enhanced) + gap_feat
        self.predict_head = nn.Sequential(
            nn.Conv2d(embed_dim * 6, embed_dim * 2, kernel_size=1),  # 6个特征拼接
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, num_classes, kernel_size=1)
        )

    def forward(self, features):
        """
        features: 列表，包含编码器四个阶段的特征 [f2, f3, f4, f5]
                  f2: (B, C2, H/8, W/8)
                  f3: (B, C3, H/16, W/16)
                  f4: (B, C4, H/32, W/32)
                  f5: (B, C5, H/32, W/32)
        返回: 分割logits (B, num_classes, H, W)
        """
        f2, f3, f4, f5 = features

        # 1. 通道投影
        f2_reduced = self.reduce_conv_c2(f2)  # (B, embed_dim, H/8, W/8)
        f3_reduced = self.reduce_conv_c3(f3)  # (B, embed_dim, H/16, W/16)
        f4_reduced = self.reduce_conv_c4(f4)  # (B, embed_dim, H/32, W/32)
        f5_reduced = self.reduce_conv_c5(f5)  # (B, embed_dim, H/32, W/32)

        # 2. 将所有特征上采样到f2的尺寸 (H/8, W/8) 进行融合
        f3_up = torch.nn.functional.interpolate(f3_reduced, size=f2_reduced.shape[2:],
                                                mode='bilinear', align_corners=True)
        f4_up = torch.nn.functional.interpolate(f4_reduced, size=f2_reduced.shape[2:],
                                                mode='bilinear', align_corners=True)
        f5_up = torch.nn.functional.interpolate(f5_reduced, size=f2_reduced.shape[2:],
                                                mode='bilinear', align_corners=True)

        # 3. 聚合生成 F (在H/8分辨率)
        F_cat = torch.cat([f2_reduced, f3_up, f4_up, f5_up], dim=1)
        F = self.fusion_conv(F_cat)  # (B, embed_dim, H/8, W/8)

        # 4.
        F_prime = self.sc(F)  # (B, embed_dim, H/8, W/8)

        # 5. 多尺度融合 (将 F' 注入到各个阶段特征)
        # 将F'上采样/下采样到各个特征的分辨率
        F_prime_up_to_f2 = F_prime  # 已经是H/8

        # 上采样到H/16用于f3
        F_prime_up_to_f3 = torch.nn.functional.interpolate(
            F_prime, size=f3_reduced.shape[2:], mode='bilinear', align_corners=True
        )

        # 上采样到H/32用于f4和f5
        F_prime_up_to_f4_f5 = torch.nn.functional.interpolate(
            F_prime, size=f4_reduced.shape[2:], mode='bilinear', align_corners=True
        )

        # 特征增强
        f2_enhanced = f2_reduced + F_prime_up_to_f2
        f3_enhanced = f3_reduced + F_prime_up_to_f3
        f4_enhanced = f4_reduced + F_prime_up_to_f4_f5
        f5_enhanced = f5_reduced + F_prime_up_to_f4_f5

        # 6. 准备最终分类器输入
        # 将所有增强特征上采样到H/8
        f3_enhanced_up = torch.nn.functional.interpolate(
            f3_enhanced, size=f2_reduced.shape[2:], mode='bilinear', align_corners=True
        )
        f4_enhanced_up = torch.nn.functional.interpolate(
            f4_enhanced, size=f2_reduced.shape[2:], mode='bilinear', align_corners=True
        )
        f5_enhanced_up = torch.nn.functional.interpolate(
            f5_enhanced, size=f2_reduced.shape[2:], mode='bilinear', align_corners=True
        )

        # 全局平均池化特征（在f5_enhanced上做）
        gap_feat = self.gap(f5_enhanced)
        gap_feat = self.gap_proj(gap_feat)
        gap_feat = torch.nn.functional.interpolate(
            gap_feat, size=f2_reduced.shape[2:], mode='bilinear', align_corners=True
        )

        # 拼接所有特征: [F', f2_enhanced, f3_enhanced_up, f4_enhanced_up, f5_enhanced_up, gap_feat]
        final_feat = torch.cat([
            F_prime_up_to_f2,
            f2_enhanced,
            f3_enhanced_up,
            f4_enhanced_up,
            f5_enhanced_up,
            gap_feat
        ], dim=1)

        # 7. 最终预测
        out = self.predict_head(final_feat)  # (B, num_classes, H/8, W/8)
        out = torch.nn.functional.interpolate(
            out, scale_factor=8, mode='bilinear', align_corners=True
        )  # 上采样回原图尺寸 (B, num_classes, H, W)

        return out


class MFU(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        # print(dims)
        # exit()
        dims2 = [384, 768, 1536, 3072]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # self.SMT = SMT(pretrain_img_size=224, num_classes=4)
        # self.SMT_UP = SMT_UP(num_classes=4)
        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        mlp_ratios = [8, 6, 4, 2]
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # self.layers_CNN = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #    layer = LocalAggadd3(channels=dims[i_layer])
        #    self.layers_CNN.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # 修改为四分支解码器
        # 假设编码器输出4个特征：stage1, stage2, stage3, stage4
        # 对应分辨率：H/4, H/8, H/16, H/32, H/32
        # 但我们的解码器需要H/8, H/16, H/32, H/32四个特征
        # 所以取dims[1], dims[2], dims[3], dims[3]
        encoder_channels_for_decoder = [dims[1], dims[2], dims[3], dims[3]]  # 四个阶段的通道数

        self.decoder = SegMAN_Decoder_4Branch(
            encoder_channels=encoder_channels_for_decoder,
            embed_dim=256,  # 可调整
            num_classes=num_classes
        )

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.ChannelReducerlist = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ChannelReducer(
                in_channels=dims2[i_layer], out_channels=dims2[i_layer] / 2
            )
            self.ChannelReducerlist.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
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

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(4):
            skip_list.append(x)
            # print(x.shape)
            x1 = self.layers[i](x)
            x = x1

        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):

            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        # skip_list = self.forward_FRM(skip_list)

        # 获取四个特征：f2(H/8), f3(H/16), f4(H/32), f5(H/32)
        # skip_list[0]: H/4
        # skip_list[1]: H/8
        # skip_list[2]: H/16
        # skip_list[3]: H/32
        # x: H/32 (最后一个阶段的输出)

        f2 = skip_list[1].permute(0, 3, 1, 2)  # (B, C2, H/8, W/8)
        f3 = skip_list[2].permute(0, 3, 1, 2)  # (B, C3, H/16, W/16)
        f4 = skip_list[3].permute(0, 3, 1, 2)  # (B, C4, H/32, W/32)
        f5 = x.permute(0, 3, 1, 2)  # (B, C5, H/32, W/32) - 最后一个阶段的输出
        #print(f1.shape)
        #exit()
        # 4. 解码器前向传播
        out = self.decoder([f2, f3, f4, f5])
        # x = self.forward_features_up(x, skip_list)
        # x = self.forward_final(x)

        return out, skip_list







