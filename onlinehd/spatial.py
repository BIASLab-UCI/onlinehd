import math

import torch

def cos_cdist(x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    r'''
    Computes pairwise cosine similarity between samples in `x1` and `x2`,
    forcing each point l2-norm to be at least `eps`. This similarity between
    `(n?, f?)` samples described in :math:`x1` and the `(m?, f?)` samples
    described in :math:`x2` with scalar :math:`\varepsilon > 0` is the
    `(n?, m?)` matrix :math:`\delta` given by:

    .. math:: \delta_{ij} = \frac{x1_i \cdot x2_j}{\max\{\|x1_i\|, \varepsilon\} \max\{\|x2_j\|, \varepsilon\}}

    Args:
        x1 (:class:`torch.Tensor`): The `(n?, f?)` sized matrix of datapoints
            to score with `x2`.

        x2 (:class:`torch.Tensor`): The `(m?, f?)` sized matrix of datapoints
            to score with `x1`.

        eps (float, > 0): Scalar to prevent zero-norm vectors.

    Returns:
        :class:`torch.Tensor`: The `(n?, m?)` sized tensor `dist` where
        `dist[i,j] = cos(x1[i], x2[j])` given by the equation above.

    '''
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    cdist = x1 @ x2.T
    cdist.div_(norms1).div_(norms2)
    return cdist
