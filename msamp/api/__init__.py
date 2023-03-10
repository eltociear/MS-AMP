# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP api module."""

from msamp.api.amp import initialize
from msamp.api.clip_grad import clip_grad_norm_

__all__ = ['initialize', 'clip_grad_norm_']
