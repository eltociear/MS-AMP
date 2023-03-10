# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MS-AMP amp module."""

import torch
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig

from msamp.nn import LinearReplacer
from msamp.optim import LBAdam, LBAdamW

opt_levels = ['O1', 'O2', 'O3']
supported_optimizers = (torch.optim.AdamW, torch.optim.Adam)


def _get_deepspeed_config(**deepspeed_args):
    """Get deepspeed config.

    Args:
        deepspeed_args (dict): Arguments for deepspeed. Only used when opt_level is 'O3'.

    Return:
        A DeepSpeedConfig object.
    """
    config = deepspeed_args.get('config', None)
    config_params = deepspeed_args.get('config_params', None)
    if config is None:
        config = config_params
    args = deepspeed_args.get('args', None)
    if config is None:
        config = (args.deepspeed_config if hasattr(args, 'deepspeed_config') else None)
    return DeepSpeedConfig(config)


def initialize(model, optimizer=None, opt_level='O1', use_deep_speed=False, **deepspeed_args):  # noqa: C901
    """Initialize your model, optimizer according to the optimization level.

    msamp.initialize() should be called after you have finished constructing your model and optimizer.
    Currently, msamp.initialize() should be called only once.

    Args:
        model (torch.nn.Module): Model to cast.
        optimizer (torch.optim.Optimizer): Optimizer to cast.
        opt_level (str): Optimization level. Currently supports 'O1', 'O2', 'O3'. 'O3' is used for deepspeed
            zero optimizer. Here are details of the optimization level:
            opt_level || Gemm || Communication || Weight || Weight Gradient || Optimizer States || Master Weight
            'O1'      || fp8  || fp8           || fp16   || fp8             || fp32 + FP32      || -
            'O2'      || fp8  || fp8           || fp16   || fp8             || fp8 + fp16       || -
            'O3'      || fp8  || fp8           || fp16   || fp8             || fp8 + fp16       || FP16
        use_deep_speed (bool): If True, the model and optimizer will be casted to deepspeed.
        deepspeed_args (dict): Arguments for deepspeed except model and optimizer.

    Return:
        If use_deep_speed is False, return the casted model and optimizer. Otherwise, return a tuple of engine,
            optimizer, training_dataloader, lr_scheduler.
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Model must be an instance of torch.nn.Module')

    if opt_level not in opt_levels:
        raise ValueError('Invalid optimization level. Please choose from {}'.format(opt_levels))

    cast_model = None
    cast_optimizer = None

    if opt_level == 'O1' or opt_level == 'O2':
        cast_model = LinearReplacer.replace(model)
        if not optimizer:
            # default optimizer.
            optimizer = LBAdamW(cast_model.parameters())
            optimizer.set_model(model)
            return model, optimizer

        if not isinstance(optimizer, supported_optimizers):
            raise ValueError('Optimizer {} is not supported in optimization level {}'.format(optimizer, opt_level))

        default_args = optimizer.defaults

        exp_avg_dtype = torch.float32
        exp_avg_sq_dtype = torch.float32
        if opt_level == 'O2':
            exp_avg_dtype = torch.uint8
            exp_avg_sq_dtype = torch.float16

        default_args['exp_avg_dtype'] = exp_avg_dtype
        default_args['exp_avg_sq_dtype'] = exp_avg_sq_dtype

        # currently, we don't support foreach and capturable.
        if 'foreach' in default_args:
            del default_args['foreach']
        if 'capturable' in default_args:
            del default_args['capturable']

        parameters = cast_model.parameters()
        if isinstance(optimizer, torch.optim.Adam):
            cast_optimizer = LBAdam(parameters, **default_args)
        elif isinstance(optimizer, torch.optim.AdamW):
            cast_optimizer = LBAdamW(parameters, **default_args)
        cast_optimizer.set_model(cast_model)

        if use_deep_speed:
            return deepspeed.initialize(model=model, optimizer=optimizer, **deepspeed_args)

        return cast_model, cast_optimizer
    else:  # O3
        if not use_deep_speed:
            raise ValueError('use_deep_speed must be True when opt_level is O3')

        # check if zero_optimization is enabled.
        deepspeed_config = _get_deepspeed_config(**deepspeed_args)
        if not deepspeed_config.zero_optimization:
            raise ValueError('zero_optimization must be enabled when opt_level is O3')

        return deepspeed.initialize(model=model, optimizer=optimizer, **deepspeed_args)
