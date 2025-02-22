# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# this file is adapted from deepspeed (Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0. DeepSpeed Team)

"""DeepSpeedEngine in MS-AMP."""
import deepspeed
from deepspeed.runtime.engine import SparseTensor, ZERO_OPTIMIZATION, AMP, amp, \
                                     FP16, BFLOAT16, logger, DeepSpeedEngine, instrument_w_nvtx, log_dist, \
                                     see_memory_usage, DummyOptim, DeepSpeedZeroOptimizer, DeepSpeedZeRoOffload, \
                                     PipelineModule, ZeroStageEnum

from msamp import initialize as msamp_initialize
from msamp.common.tensor import ScalingTensor, TensorDist
from msamp.optim import LBOptimizer
from msamp.deepspeed.runtime.fp8.fused_optimizer import FP8Optimizer
from msamp.deepspeed.runtime.zero import utils    # noqa: F401
from msamp.deepspeed.runtime.zero.fp8_stage_1_and_2 import FP8DeepSpeedZeroOptimizer
from msamp.deepspeed.runtime.config import FP8


def split_half_float_double_sparse(tensors):
    """Split tensors into buckets of the same type.

    Args:
        tensors (list): list of tensors to be bucketed.

    Returns:
        list: list of buckets, each bucket is a tuple of (dtype, list of tensors).
    """
    supported_types = [
        'torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor', 'torch.cuda.BFloat16Tensor',
        'msamp.common.tensor.tensor.ScalingTensor',
        SparseTensor.type()
    ]

    for t in tensors:
        assert t.type() in supported_types, f'attempting to reduce an unsupported grad type: {t.type()}'

    buckets = []
    for _, dtype in enumerate(supported_types):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


deepspeed.runtime.engine.split_half_float_double_sparse = split_half_float_double_sparse


class MSAMPDeepSpeedEngine(DeepSpeedEngine):
    """DeepSpeed Engine with MS-AMP support."""
    def _configure_optimizer(self, client_optimizer, model_parameters):
        """Config basic optimizer and optimizer.

        Args:
            client_optimizer (torch.optim.Optimizer or callable): client optimizer.
            model_parameters (list): list of model parameters.
        """
        if client_optimizer is not None:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                client_optimizer.param_groups[:] = [
                    pg for pg in client_optimizer.param_groups if len(pg['params']) != 0
                ]
                log_dist("Removing param_group that has no 'params' in the client Optimizer", ranks=[0])

                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])
        else:
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(f'Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer', ranks=[0])

        if self.msamp_enabled():
            optlevel = self.msamp_optlevel()
            if optlevel == 'O3':
                # O3 is for ZeRO and need to cast to O2 for MS-AMP.
                optlevel = 'O2'
            model, basic_optimizer = msamp_initialize(self.module, basic_optimizer, optlevel)
            self._set_client_model(model)
            # We need to reset param names after msamp initialize.
            self.param_names = {param: name for name, param in model.named_parameters()}

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        log_dist('DeepSpeed Basic Optimizer = {}'.format(basic_optimizer.__class__.__name__), ranks=[0])

        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)
        if optimizer_wrapper == ZERO_OPTIMIZATION:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif optimizer_wrapper == FP8:
            self.optimizer = self._configure_fp8_optimizer(basic_optimizer, optimizer_wrapper)
        elif optimizer_wrapper == AMP:
            amp_params = self.amp_params()
            log_dist(f'Initializing AMP with these params: {amp_params}', ranks=[0])
            model, self.optimizer = amp.initialize(self.module, basic_optimizer, **amp_params)
            self._set_client_model(model)
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif optimizer_wrapper == FP16:
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        elif optimizer_wrapper == BFLOAT16:
            self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        log_dist('DeepSpeed Final Optimizer = {}'.format(self.optimizer_name()), ranks=[0])

        self.compression_scheduler = self._configure_compression_scheduler()
        self.quantizer = self._configure_quantization()

    def _do_optimizer_sanity_check(self, basic_optimizer):
        """Check if optimizer is supported and return the wrapper type."""
        if self.zero_optimization():
            return super()._do_optimizer_sanity_check(basic_optimizer)

        if isinstance(basic_optimizer, LBOptimizer):
            return FP8
        return super()._do_optimizer_sanity_check(basic_optimizer)

    def _configure_fp8_optimizer(self, optimizer, optimizer_wrapper):
        """Configure fp8 optimizer.

        Args:
            optimizer (torch.optim.Optimizer): basic optimizer.
            optimizer_wrapper (str): optimizer wrapper.

        Returns:
            FP8_Optimizer: fp8 optimizer.
        """
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()

        if self.dynamic_loss_scale():
            log_dist('Creating fp8 optimizer with dynamic loss scale', ranks=[0])
            timers = self.timers if self.wall_clock_breakdown() else None
            optimizer = FP8Optimizer(
                optimizer,
                deepspeed=self,
                dynamic_loss_scale=True,
                initial_dynamic_scale=initial_dynamic_scale,
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                timers=timers,
                has_moe_layers=self.has_moe_layers,
            )
        else:
            log_dist(
                'Creating fp8 optimizer with static loss scale: {}'.format(self.loss_scale()),
                ranks=[0],
            )
            loss_scale = self.loss_scale()
            if loss_scale == 0:
                loss_scale = 1
            optimizer = FP8Optimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=loss_scale,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_adam_legacy=self.optimizer_legacy_fusion(),
                has_moe_layers=self.has_moe_layers,
            )

        return optimizer

    def _configure_zero_optimizer(self, optimizer):
        """Config zero optimizer.

        Args:
            optimizer (torch.optim.Optimizer): basic optimizer.
            use_fp8 (bool, optional): whether to use fp8 optimizer. Defaults to False.

        Returns:
            ZeROOptimizer: zero optimizer.
        """
        zero_stage = self.zero_optimization_stage()
        timers = self.timers if self.wall_clock_breakdown() else None

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        if self.zero_legacy_stage1():
            raise Exception(
                'The deprecated version of ZeRO Stage 1 is not supported in deepspeed >= 0.5.9. '
                'Please downgrade to a version less than 0.5.9 if '
                'you need to use this deprecated version of ZeRO.'
            )

        if zero_stage <= ZeroStageEnum.gradients:
            overlap_comm = self.zero_overlap_comm()
            contiguous_gradients = self.zero_contiguous_gradients()
            round_robin_gradients = self.zero_round_robin_gradients()
            assert not isinstance(optimizer, DummyOptim), 'zero stage {} requires an optimizer'.format(zero_stage)

            log_dist('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage), ranks=[0])
            # Overlap and contiguous grads are meaningless in stage 1 and are ignored
            if zero_stage == ZeroStageEnum.optimizer_states:
                overlap_comm = False
                round_robin_gradients = False

            if isinstance(self.module, PipelineModule):
                if overlap_comm:
                    logger.warning('Pipeline parallelism does not support overlapped communication, will be disabled.')
                    overlap_comm = False
            zero_t = DeepSpeedZeroOptimizer
            if self.msamp_enabled() or isinstance(optimizer, LBOptimizer):
                zero_t = FP8DeepSpeedZeroOptimizer
            optimizer = zero_t(
                optimizer,
                self.param_names,
                timers=timers,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=self.dynamic_loss_scale_args(),
                clip_grad=self.gradient_clipping(),
                contiguous_gradients=contiguous_gradients,
                reduce_bucket_size=self.zero_reduce_bucket_size(),
                allgather_bucket_size=self.zero_allgather_bucket_size(),
                dp_process_group=self.data_parallel_group,
                expert_parallel_group=self.expert_parallel_group if self.has_moe_layers else None,
                expert_data_parallel_group=self.expert_data_parallel_group if self.has_moe_layers else None,
                reduce_scatter=self.zero_reduce_scatter(),
                overlap_comm=overlap_comm,
                cpu_offload=self.zero_cpu_offload(),
                mpu=self.mpu,
                postscale_gradients=self.postscale_gradients(),
                gradient_predivide_factor=self.gradient_predivide_factor(),
                gradient_accumulation_steps=self.gradient_accumulation_steps(),
                ignore_unused_parameters=self.zero_ignore_unused_parameters(),
                partition_grads=zero_stage == ZeroStageEnum.gradients,
                round_robin_gradients=round_robin_gradients,
                has_moe_layers=self.has_moe_layers,
                fp16_master_weights_and_gradients=self.fp16_master_weights_and_gradients(),
                communication_data_type=self.communication_data_type,
                elastic_checkpoint=self.zero_elastic_checkpoint()
            )
            # update_hp_grads and clear_lp_grads will be called in PipelineEngine when using pipeline+bf16+zero1,
            # so we just set them to None.
            zero_t.update_hp_grads = lambda instance, clear_lp_grads: None
            zero_t.clear_lp_grads = lambda instance: None

        elif zero_stage == ZeroStageEnum.weights:
            assert not self.has_moe_layers, 'MoE not supported with Stage 3'
            if isinstance(optimizer, DummyOptim):
                log_dist('Creating ZeRO Offload', ranks=[0])
                optimizer = DeepSpeedZeRoOffload(
                    self.module,
                    timers=timers,
                    ds_config=self.config,
                    overlap_comm=self.zero_overlap_comm(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    offload_param_config=self.zero_offload_param(),
                    mpu=self.mpu
                )
            else:
                log_dist('Creating fp16 ZeRO stage {} optimizer'.format(zero_stage), ranks=[0])
                from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
                optimizer = DeepSpeedZeroOptimizer_Stage3(
                    self.module,
                    optimizer,
                    timers=timers,
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.data_parallel_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps(),
                    aio_config=self.aio_config(),
                    communication_data_type=self.communication_data_type
                )

        else:
            raise NotImplementedError('ZeRO stage {} not implemented'.format(zero_stage))

        return optimizer

    @instrument_w_nvtx
    def backward(      # noqa: C901
        self,
        loss,
        allreduce_gradients=True,
        release_loss=False,
        retain_graph=False,
        scale_wrt_gas=True
    ):
        """Execute backward pass on the loss.

        Args:
            loss: Torch tensor on which to execute backward propagation.
            all_reduce_gradients (bool, optional): All reduce gradients in the backward pass.
            release_loss (bool, optional): Release the loss tensor after the backward pass.
            retain_graph (bool, optional): Retain the computation graph after backward pass.
            scale_wrt_gas (bool, optional): Scale the loss w.r.t. gradient accumulation steps.

        Returns:
            loss: The loss tensor.
        """
        see_memory_usage('Engine before backward', force=self.memory_breakdown())

        if self.scale_wrt_gas is not None:
            scale_wrt_gas = self.scale_wrt_gas

        if not allreduce_gradients:
            logger.warning('Argument `allreduce_gradients` is deprecated, ignored, and will soon be removed')

        # scale loss w.r.t. gradient accumulation if needed
        if self.gradient_accumulation_steps() > 1 and scale_wrt_gas:
            loss = self._scale_loss_by_gas(loss.float())

        # Log training Loss
        if self.monitor.enabled:
            if self.is_gradient_accumulation_boundary():
                if self.global_rank == 0:
                    self.summary_events = [
                        (
                            'Train/Samples/train_loss',
                            loss.mean().item() * self.gradient_accumulation_steps(),
                            self.global_samples,
                        )
                    ]
                    self.monitor.write_events(self.summary_events)

        self._start_timers(self.engine_timers.backward_timers)

        assert self.optimizer is not None and not isinstance(self.optimizer, DummyOptim), \
            'must provide optimizer during init in order to use backward'

        self._start_timers(self.engine_timers.backward_inner_timers)

        if self.zero_optimization():
            self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif isinstance(self.optimizer, FP8Optimizer):
            self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.amp_enabled():
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
            delay_unscale = not self.is_gradient_accumulation_boundary()
            with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        elif self.fp16_enabled():
            if self.eigenvalue_enabled():
                self.optimizer.backward(loss, create_graph=True, retain_graph=True)
            else:
                self.optimizer.backward(loss, retain_graph=retain_graph)
        elif self.bfloat16_enabled():
            self.optimizer.backward(loss)
        else:
            if self.eigenvalue_enabled():
                loss.backward(create_graph=True, retain_graph=True)
            else:
                loss.backward(retain_graph=retain_graph)

        self._stop_timers(self.engine_timers.backward_inner_timers)

        self._start_timers(self.engine_timers.backward_reduce_timers)

        if allreduce_gradients and self.enable_backward_allreduce:
            # Traditional code path that allreduces the module parameter grads
            self.allreduce_gradients()

        self._stop_timers(self.engine_timers.backward_reduce_timers)

        self._stop_timers(self.engine_timers.backward_timers)

        if release_loss:
            # loss.data = None
            pass

        see_memory_usage('Engine after backward', force=self.memory_breakdown())

        return loss

    def fp8_allreduce_bucket(self, bucket, dp_group):
        """All reduce bucket of ScalingTensor.

        Args:
            bucket (list of ScalingTensor): bucket of ScalingTensor.
            dp_group: data parallel group.
        """
        if self.gradient_average:
            TensorDist.all_reduce_avg(bucket)
        else:
            TensorDist.all_reduce_sum(bucket)

    def allreduce_and_copy(self, small_bucket, dp_group):
        """All reudce tensors after flatten and copy to original tensors.

        Args:
            small_bucket (list of torch.Tensor or ScalingTensor): bucket of tensors.
            dp_group: data parallel group.
        """
        if len(small_bucket) == 0:
            return
        if isinstance(small_bucket[0], ScalingTensor):
            # ScalingTensor all reduce
            self.fp8_allreduce_bucket(small_bucket, dp_group)
            return
        allreduced = self.allreduce_bucket(small_bucket, dp_group)
        for buf, synced in zip(small_bucket, self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def msamp_enabled(self):
        """Whether amp is enabled."""
        return self._config.msamp_enabled

    def msamp_optlevel(self):
        """Return the opt level of MS-AMP."""
        return self._config.msamp_optlevel
