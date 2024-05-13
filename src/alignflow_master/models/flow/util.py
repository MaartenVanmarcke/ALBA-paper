import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def checkerboard(x, reverse=False):
    raise NotImplementedError("")
    """Split x in a checkerboard pattern. Collapse horizontally."""
    # Get dimensions
    if reverse:
        b, c, h, w = x[0].size()
        w *= 2
        device = x[0].device
    else:
        b, c, h, w = x.size()
        device = x.device

    # Get list of indices in alternating checkerboard pattern
    y_idx = []
    z_idx = []
    for i in range(h):
        for j in range(w):
            if (i % 2) == (j % 2):
                y_idx.append(i * w + j)
            else:
                z_idx.append(i * w + j)
    y_idx = torch.tensor(y_idx, dtype=torch.int64, device=device)
    z_idx = torch.tensor(z_idx, dtype=torch.int64, device=device)

    if reverse:
        y, z = (t.contiguous().view(b, c, h * w // 2) for t in x)
        x = torch.zeros(b, c, h * w, dtype=y.dtype, device=y.device)
        x[:, :, y_idx] += y
        x[:, :, z_idx] += z
        x = x.view(b, c, h, w)

        return x
    else:
        if w % 2 != 0:
            raise RuntimeError('Checkerboard got odd width input: {}'.format(w))

        x = x.view(b, c, h * w)
        y = x[:, :, y_idx].view(b, c, h, w // 2)
        z = x[:, :, z_idx].view(b, c, h, w // 2)

        return y, z


def channelwise(x, reverse=False):
    raise NotImplementedError("")
    """Split x channel-wise. If channel is specified, split out just that channel."""
    if reverse:
        x = torch.cat(x, dim=1)
    else:
        x = x.chunk(2, dim=1)

    return x


def squeeze(x):
    raise NotImplementedError("")
    """Trade spatial extent for channels. I.e., convert each
    1x4x4 volume of input into a 4x1x1 volume of output.

    Args:
        x (torch.Tensor): Input to squeeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * 2 * 2, h // 2, w // 2)

    return x


def unsqueeze(x):
    raise NotImplementedError("")
    """Trade channels channels for spatial extent. I.e., convert each
    4x1x1 volume of input into a 1x4x4 volume of output.

    Args:
        x (torch.Tensor): Input to unsqueeze.

    Returns:
        x (torch.Tensor): Unsqueezed tensor.
    """
    b, c, h, w = x.size()
    x = x.view(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(b, c // 4, h * 2, w * 2)

    return x


def alt_squeeze(x, reverse=False):
    raise NotImplementedError("")
    """Trade spatial extent for channels using an ordering that's different
    from the normal squeeze function's channel ordering.

    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.

    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    # Defines permutation of input channels (shape is (4, 1, 2, 2)).
    squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                   [[[0., 0.], [0., 1.]]],
                                   [[[0., 1.], [0., 0.]]],
                                   [[[0., 0.], [1., 0.]]]],
                                  dtype=x.dtype,
                                  device=x.device)
    n, c, h, w = x.size()
    if reverse:
        c //= 4
    perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
    for c_idx in range(c):
        slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
        slice_1 = slice(c_idx, c_idx + 1)
        perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
    shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                    + [c_idx * 4 + 1 for c_idx in range(c)]
                                    + [c_idx * 4 + 2 for c_idx in range(c)]
                                    + [c_idx * 4 + 3 for c_idx in range(c)])
    perm_weight = perm_weight[shuffle_channels, :, :, :]

    if reverse:
        x = F.conv_transpose2d(x, perm_weight, stride=2)
    else:
        x = F.conv2d(x, perm_weight, stride=2)

    return x


def concat_elu(x):
    """Concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU."""
    return F.elu(torch.cat((x, -x), dim=1))


def safe_log(x):
    return torch.log(x.clamp(min=1e-22))


def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.
    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def tanh_to_logits(x, sldj, reverse=False):
    if reverse:
        tanh_ldj = 2. * (math.log(2.) - x - F.softplus(-2. * x))
        x = torch.tanh(x)

        # Scale (-0.95, 0.95) -> (-1, 1)
        # x = x / 0.95
        # sldj = sldj - math.log(0.95)

        sldj = sldj + tanh_ldj.flatten(1).sum(-1)
    else:
        # Scale (-1, 1) -> (-0.95, 0.95)
        x = x * 0.95

        # Inverse tanh
        # https://en.wikipedia.org/wiki/Hyperbolic_function#Inverse_functions_as_logarithms
        artanh_ldj = -torch.log(1. - x ** 2)
        x = 0.5 * torch.log((1 + x) / (1 - x))

        # Inverse tanh
        sldj = sldj + math.log(0.95) + artanh_ldj.flatten(1).sum(-1)

    return x, sldj