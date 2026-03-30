# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    # 这里按 token 级别计算 KL 散度惩罚项。
    # 输入是当前策略 old_log_probs 与参考策略 ref_log_prob 的对数概率。
    # 对离散分布而言，KL 的核心形式可写为：KL(P||Q) = E_P[log P - log Q]。
    # 在实现上，core_algos.kl_penalty(...) 会根据 kl_penalty 配置返回逐 token 的惩罚值。
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    # 仅在 response 区间上保留 KL；prompt 部分由 response_mask 置零。
    kld = kld * response_mask
    # beta 是 KL 系数（自适应控制器当前值），用于控制“贴近参考策略”的强度。
    beta = kl_ctrl.value

    # 最终 token 奖励 = 原始 token 分数 - beta * KL。
    # KL 越大，奖励扣减越多，从而抑制策略偏离参考策略过快。
    token_level_rewards = token_level_scores - beta * kld

    # 先对每个样本在序列维度做 masked mean（只统计 response token），
    # 再在 batch 维度求平均，得到一个标量 current_kl 用于日志和自适应调参。
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # 自适应更新 KL 控制器：当 current_kl 偏大/偏小时，后续 beta 会动态调节。
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # 将带 KL 惩罚的 token 级奖励写回 batch，供后续 advantage 计算使用。
    data.batch["token_level_rewards"] = token_level_rewards

    # 记录监控指标：当前 KL 与当前 KL 系数。
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    # responses 的形状通常是 [batch_size, response_len]，
    # 这里只取第 2 维长度，得到每条样本的响应 token 数。
    responses = data.batch["responses"]
    response_length = responses.size(1)

    # attention_mask 覆盖的是“整段输入”（通常是 prompt + response）。
    attention_mask = data.batch["attention_mask"]

    # 约定：response 位于拼接序列的尾部，因此用切片 [:, -response_length:]
    # 直接截取最后 response_length 列，得到仅对应 response 区间的 mask。
    # 后续 KL、优势等计算只在该区域生效，避免把 prompt 区域混入训练信号。
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # 兼容旧路径：如果上游还没有提前构造 response_mask，
    # 这里补算一次，确保后续优势函数只在 response token 上工作。
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    # 根据配置选择不同的 advantage estimator。
    # 所有分支最终都会把结果写回：
    # - advantages: 策略梯度里真正使用的优势项
    # - returns: 通常供 critic/value 学习作为回归目标
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE 分支：需要 value 网络输出 values。
        # 它会结合 token 级 reward、价值估计、gamma、lam 计算：
        # 1) advantage
        # 2) return
        # 这是 PPO 中最经典的一条路径。
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

        # PF-PPO 的可选重加权逻辑。
        # 如果打开，会基于配置中的重加权策略对样本重新赋权。
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO 分支：同样只在 response 区间做计算。
        grpo_calculation_mask = data.batch["response_mask"]

        # GRPO 会按 uid 对同一组样本进行组内比较，
        # 因此这里需要 non_tensor_batch["uid"] 作为分组索引。
        # norm_adv_by_std_in_grpo 控制是否按标准差归一化优势。
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # 其他 estimator 统一走“函数分发”路径：
        # 先根据 adv_estimator 找到对应实现，再按需拼装参数。
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }

        # 某些 estimator 需要按 uid 对样本分组或聚合，存在时就传入。
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]

        # 某些算法需要基线奖励（例如 REMAX 一类路径），存在时就附带传入。
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            # GDPO 不只依赖聚合后的 token reward，
            # 还可能需要原始 batch / non_tensor_batch 做按维度奖励拆解。
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch

        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # 这两类 optimal baseline 需要 actor 额外提供 sum_pi_squared。
            # 如果没算这个量，就无法构造对应的 step-dependent baseline。
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]

            # 如果前面已经做过 rollout correction / importance sampling，
            # 这里会把预先计算好的 IS 权重也一起传进去。
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # 调用具体 estimator 实现，统一拿回 advantages 和 returns。
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # 返回写回后的 DataProto，后续 actor/critic 更新都会使用这些字段。
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy in {"fsdp", "fsdp2"}:
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # 对于每个资源池，创建一个 WorkerGroup，里面包含分配到该资源池的所有角色的 Worker 类。
            if not class_dict:
                # 如果池子里没有角色，就不创建 WorkerGroup 了
                continue
            # 创建worker类型
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # 绑定同种类型的worker和资源池
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            # 收集所有worker group
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: 先把训练主流程里的 DataProto 转成 tensordict。
            # 新版 critic worker 更偏向直接消费 tensordict，因此这里先做格式适配。
            batch_td = batch.to_tensordict()
            # step 2: 把带 padding 的表示压缩成 no-padding 形式。
            # 这样 critic 在做 value 前向时只处理有效 token，减少 pad token 带来的额外计算开销。
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: 给 critic 补充控制信息。
            # 这里 compute_loss=False 表示当前只做 value 推理，
            # 不是在这一阶段直接训练 critic。
            tu.assign_non_tensor(batch_td, compute_loss=False)
            # 调用 critic worker 对当前 batch 做一次 value 前向。
            # 返回的是各 token 位置对应的 value 估计，后面会用于 advantage/return 计算。
            output = self.critic_wg.infer_batch(batch_td)
            # Ray worker 返回的是异步结果句柄，这里显式取回真正输出。
            output = output.get()
            # 从输出中取出 critic 的核心结果 values。
            values = tu.get(output, "values")
            # 把 no-padding 的 values 还原回 padding 布局。
            # 因为后续 response_mask、old_log_probs、ref_log_prob、token_level_rewards
            # 都和原始 batch 的 padding 结构对齐，values 也必须保持同样布局。
            values = no_padding_2_padding(values, batch_td)
            # 用主流程约定的字段名 values 重新打包，
            # 再从 tensordict 转回 DataProto，方便后面直接 union 回 batch。
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            # 旧版 worker 可以直接接受 DataProto，并直接返回主流程能消费的 values 结构。
            values = self.critic_wg.compute_values(batch)
        # 返回 critic 的 value 估计结果。
        # 后面的 compute_advantage(...)，尤其是 GAE 分支，会直接依赖这个字段。
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: 先把训练主流程里的 DataProto 转成 tensordict。
            # 新版 worker 接口主要以 tensordict 为输入，因此这里先做格式适配。
            batch_td = batch.to_tensordict()
            # step 2: 把带 padding 的表示压缩成 no-padding 形式。
            # 这样 reference policy 在前向时只处理有效 token，减少 pad token 带来的额外开销。
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: 给这次 reference 前向补充控制信息。
            # 这里不需要 entropy，也不需要在这一阶段直接构造 loss，
            # 因此 calculate_entropy=False, compute_loss=False。
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                # 如果 reference policy 和 actor 共用同一个模型底座，
                # 那么这里显式要求“不要启用 LoRA adapter”，
                # 从而得到未加 adapter 的 reference policy 概率。
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                # ref_in_actor=True: 不单独调用 ref_policy_wg，
                # 而是复用 actor worker，用 no_lora_adapter 模式算 reference log_prob。
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                # ref_in_actor=False: 使用独立的 reference policy worker 计算 reference log_prob。
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # 从 worker 输出里取出 token 级 log_probs。
            # 这些值后面会写入 ref_log_prob，并用于 KL 惩罚或 KL 相关监控。
            log_probs = tu.get(output, "log_probs")
            # step 4: 把 no-padding 结果还原回 padding 布局。
            # 后面的 old_log_probs、response_mask、values、token_level_rewards 等字段
            # 都和原始 batch 的 padding 结构对齐，因此这里也要保持一致。
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: 用训练主流程约定的字段名 ref_log_prob 重新打包，
            # 再从 tensordict 转回 DataProto，方便后面直接 union 回 batch。
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            # 旧版实现可以直接接受 DataProto，并返回主流程能直接消费的 reference log_prob 结果。
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        # 返回 reference policy 的 token 级对数概率。
        # 后面 apply_kl_penalty(...) 会把它和 old_log_probs 一起用于计算 KL 惩罚。
        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: 先把训练主流程里的 DataProto 转成 tensordict。
            # 新版 worker 更偏向直接消费 tensordict，因此这里先做一次格式适配。
            batch_td = batch.to_tensordict()
            # step 2: 再把带 padding 的表示压缩成 no-padding 表示。
            # 这样 actor 在计算 log_prob / entropy 时只处理有效 token，避免 pad token 带来额外开销。
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: 给 actor 补充这次前向的控制信号。
            # calculate_entropy=True 表示除了 log_probs 还要返回 entropy；
            # compute_loss=False 表示这里只做前向评估，不在这里直接构造训练 loss。
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            # 用当前 actor 对这批样本重新计算 log probability。
            # 这里得到的是 PPO 更新前的“old policy”概率锚点，而不是 rollout 阶段顺手缓存的结果。
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # 从 worker 返回结果里取出需要的字段：
            # - entropy: 后面会聚合成策略熵指标
            # - log_probs: 后面会写入 old_log_probs
            # - routed_experts: MoE/路由相关附加信息（如果模型有）
            # - mfu: 推理阶段硬件利用率指标，仅用于日志
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            routed_experts = tu.get(output, "routed_experts")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4: 把 no-padding 结果重新还原回 padding 版本。
            # 因为后面的 response_mask、KL、advantages、loss 聚合等逻辑，
            # 都默认这些张量和原始 batch 的 padding 布局一一对齐。
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: 重新打包成训练主流程使用的字段名，
            # 再从 tensordict 转回 DataProto，便于后面直接 union 回原 batch。
            if routed_experts is not None:
                old_log_prob = tu.get_tensordict(
                    {"old_log_probs": log_probs.float(), "entropys": entropy.float(), "routed_experts": routed_experts}
                )
            else:
                old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            # 旧版 worker 直接接受 DataProto，并返回主流程已经能消费的 old_log_prob 结构。
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            # 旧实现没有单独暴露对应的 mfu 统计，这里用 0 占位。
            old_log_prob_mfu = 0
        # 返回两部分：
        # 1) old_log_prob: 真正参与后续 PPO/KL/rollout correction 计算的数据
        # 2) old_log_prob_mfu: 仅用于性能日志记录
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        # rollout 相关配置会决定 actor 更新时使用的多轮模式、温度等元信息。
        rollout_config = self.config.actor_rollout_ref.rollout
        # 把 multi-turn 信息写入 batch.meta_info，供 actor worker 内部更新逻辑使用。
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        # 这里把 rollout 温度也写进 meta_info，
        # 方便 actor loss 侧在需要时与生成阶段保持一致的温度设定。
        batch.meta_info["temperature"] = rollout_config.temperature
        # 真正执行 actor 参数更新。
        if self.use_legacy_worker_impl == "disable":
            # step 1: 先把训练主流程里的 DataProto 转成 tensordict。
            # 新版 actor worker 主要以 tensordict 为输入，因此这里先做格式适配。
            batch_td = batch.to_tensordict()
            # step 2: 把带 padding 的表示压缩成 no-padding 形式。
            # 这样 actor 在训练 mini-batch 时只处理有效 token，减少 pad token 的额外开销。
            batch_td = left_right_2_no_padding(batch_td)
            # 只有当 entropy_coeff 非 0 时，actor 更新阶段才需要显式计算 entropy。
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            # 读取 actor PPO 更新相关超参数。
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            # 这里乘 rollout.n，是因为每个 prompt 在 rollout 后会扩成 n 条 response，
            # actor 更新看到的是扩展后的样本总量。
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            # 把 actor 更新所需的训练控制信息写入 non-tensor metadata：
            # - 是否计算 entropy
            # - 全局 batch / mini-batch 大小
            # - PPO epoch 数
            # - 数据打乱种子与 dataloader 行为
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            # 调用 actor worker 执行真正的 PPO policy update。
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            # 从 worker 输出中取回训练指标。
            actor_output = tu.get(actor_output, "metrics")
            # 统一给 actor 指标加上 actor/ 前缀，方便日志系统分类展示。
            actor_output = rename_dict(actor_output, "actor/")
            # 对 MFU 指标做一次键名调整，归入统一的性能指标命名空间。
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            # 训练主流程只需要这些 metrics，因此把结果重新包装成一个只带 meta_info 的 DataProto。
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            # 旧版 worker 可以直接接受 DataProto，并在内部完成 actor 更新。
            actor_output = self.actor_rollout_wg.update_actor(batch)

        # 返回 actor 更新后的指标结果。
        # 主循环后面会对这些 metrics 做 reduce，并写入日志系统。
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: 先把训练主流程里的 DataProto 转成 tensordict。
            # 新版 critic worker 主要消费 tensordict，因此这里先做一次格式适配。
            batch_td = batch.to_tensordict()
            # step 2: 把带 padding 的表示压缩成 no-padding 形式。
            # 这样 critic 在做 value loss 训练时只处理有效 token，减少 pad token 的额外开销。
            batch_td = left_right_2_no_padding(batch_td)
            # 读取 critic PPO 更新相关超参数。
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            # 这里同样乘 rollout.n，因为一个 prompt 会在 rollout 后扩展成 n 条 response，
            # critic 更新实际看到的是扩展后的样本规模。
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            # 把 critic 训练所需的 mini-batch / epoch / seed / shuffle 等控制信息
            # 写入 non-tensor metadata，供 worker 内部 dataloader 和训练循环读取。
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            # 调用 critic worker 执行真正的 value model mini-batch 训练。
            output = self.critic_wg.train_mini_batch(batch_td)
            # Ray worker 返回的是异步结果句柄，这里显式取回真正输出。
            output = output.get()
            # 训练主流程这里只关心 critic 训练产生的指标。
            output = tu.get(output, "metrics")
            # 给 critic 指标统一加上 critic/ 前缀，方便日志分类。
            output = rename_dict(output, "critic/")
            # 对 MFU 指标做一次键名调整，归入统一的性能指标命名空间。
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            # 训练主流程只需要这些 metrics，因此重新包装成只带 meta_info 的 DataProto。
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            # 旧版 worker 可以直接接受 DataProto，并在内部完成 critic 更新。
            critic_output = self.critic_wg.update_critic(batch)
        # 返回 critic 更新后的指标结果。
        # 主循环后面会对这些 metrics 做 reduce，并写入日志系统。
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # 运行时按需导入，避免模块初始化阶段引入额外依赖成本。
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # 初始化训练日志跟踪器，把完整训练配置一并记录到后端。
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 先从 0 开始；如果后面成功恢复 checkpoint，会被 _load_checkpoint 改写成真实步数。
        self.global_steps = 0

        # 在开始任何训练逻辑前，优先恢复 checkpoint，并把 trainer 权重同步到 rollout 侧。
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        # 根据当前 global step 推断应该从哪个 epoch 继续。
        current_epoch = self.global_steps // len(self.train_dataloader)

        # 可选地在正式训练前先跑一次验证，当前主要依赖 reward_function 做评估。
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            # 如果当前任务只想做验证，这里直接结束。
            if self.config.trainer.get("val_only", False):
                return

        # 某些调试/实验模式下可以跳过真实 rollout，改成包装后的生成逻辑。
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # 初始化训练进度条，从恢复后的步数继续显示。
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # 主循环内部按“当前 step”工作，因此进入循环前先把 step 推进到下一个待执行步。
        self.global_steps += 1
        last_val_metrics = None
        # 记录历史最长 step 耗时，供 ESI 临近到期时判断是否需要抢救式存 checkpoint。
        self.max_steps_duration = 0

        # profiler 需要同时感知前一/当前/下一步，才能支持连续多步采样。
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # 外层按 epoch 遍历，内层按 dataloader 产出的 batch 推进训练。
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 某些异步 worker 有延迟执行的 finalize 回调，这里先尽量非阻塞地收尾上一轮残留工作。
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                # metrics 用于最终日志聚合；timing_raw 用于采样各阶段耗时。
                metrics = {}
                timing_raw = {}

                # 按当前 profiler 配置决定是否在这一步启动 profile。
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                # dataloader 给的是原始字典，这里统一转成训练主流程使用的 DataProto。
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # 温度元信息会在后续 rollout / actor 更新等环节复用。
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # 为 batch 中每条样本分配唯一 uid，后续重排、分组和 advantage 关联都依赖它。
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # 从完整 batch 中提取 rollout 所需的生成输入子集。
                gen_batch = self._get_gen_batch(batch)

                # 把当前 step 传给生成侧，方便 tracing、诊断和日志对齐。
                gen_batch.meta_info["global_steps"] = self.global_steps
                # 一个 prompt 通常要采样 n 条 response，因此先在生成前把输入按 n 倍展开。
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                # 标记当前是否已经到达训练最后一步，后面会影响验证、存档和退出逻辑。
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # step 1: 先执行 rollout，拿到模型生成的 response 及其附带元信息。
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        # rollout 结束后让副本进入休眠，便于后续 checkpoint / 权重同步管理。
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()

                        # 合并 rollout 阶段的耗时信息，避免 timing 嵌套丢失。
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # REMAX 需要一条贪心 baseline 序列来构造 reward baseline，这里额外再生成一次。
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile()
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # 如果主 batch 里还没有 RM 分数，这里补算一次，供 baseline reward 使用。
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                batch_reward = self._compute_reward_colocate(batch)
                                batch = batch.union(batch_reward)

                            # REMAX baseline 使用序列级 reward，因此把 token 级 rm_scores 聚合到序列维度。
                            reward_baseline_tensor = batch.batch["rm_scores"].sum(dim=-1)

                            # baseline 只需要 reward 结果，不需要把临时生成字段继续带入主训练 batch。
                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            # 把 baseline reward 写回 batch，供后续 advantage 计算使用。
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # step 2: 把原始样本也按 rollout.n 展开，并和生成结果按样本位对齐合并。
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # 如果 rollout 阶段没有显式给出 response_mask，这里根据序列结构补齐。
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # 可选地重排 batch，让不同 DP rank 上的有效 token 数更均衡，提升吞吐稳定性。
                    # 这通常会改变样本顺序，但 advantage 依赖 uid，不受影响。
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # 记录每条样本的总 token 数，供后面的监控统计使用。
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # 多模态样本额外记录 image token 的长度分布，便于性能分析。
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # step 3: 计算或补齐奖励模型分数。
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # 从 batch 中抽取训练真正使用的 reward tensor，以及附加诊断信息。
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # step 4: 决定 old_log_probs 的来源。
                    # bypass 模式直接复用 rollout_log_probs；
                    # decoupled 模式则重新前向计算一次 π_old，作为 PPO 更新期的稳定锚点。
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            # MFU: Multiply-accumulate FPU Utilization
                            # 衡量模型前向推理阶段的硬件利用率，数值越高表示越充分利用了计算资源。
                            # 旧版 worker 可能没有单独统计 old_log_prob 的 MFU，这里用 0 占位；
                            # 新版 worker 则直接返回这个指标。
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            # 记录 old policy 前向得到的熵和推理 MFU，便于观察策略分布和性能。
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            # 熵只用于统计，不需要继续保留在训练 batch 中。
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # 如果同时保留了 rollout_log_probs，这里补充一些调试型概率差异指标。
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    # actor 更新前，batch 中必须已经带上 old_log_probs。
                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # step 5: 如果启用了 reference policy，这里计算 reference log_prob，供 KL 项使用。
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # step 6: 如果启用了 critic，这里补充 value 预测。
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # step 7: 把 token 级 reward 写入 batch，并准备优势计算所需的附加信息。
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 如果配置了“把 KL 并入 reward”，这里先做 KL 惩罚，再得到最终 token_level_rewards。
                        # 一般情况下默认是单独计算 KL 指标，不直接影响 reward，但这个选项可以让算法更紧耦合地利用 KL 惩罚信号，可能会有更稳定的训练表现。
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # decoupled 模式下还可以进一步做 rollout correction，
                        # 包括 IS 权重、拒绝采样和对应的 off-policy 指标。
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # 最后在 driver 侧直接计算 advantage / return，避免把这类轻量逻辑分发到 worker。
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # step 8: 如果启用了 critic，先更新 value model，并收集其训练指标。
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # step 9: 只有过了 critic warmup，才开始真正更新 actor。
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 执行 actor 的 PPO 更新。
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # 判断当前训练实例是否临近到期，必要时优先抢救式保存 checkpoint。
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # 满足保存频率、最后一步或实例临期等条件时，就在 actor 更新后落盘 checkpoint。
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # actor 更新完成后，把最新权重同步给 rollout 副本，保证下一步采样使用新策略。
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # 如果配置了 rollout 数据落盘目录，这里把本步样本和奖励信息写出去。
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # step 10: 按测试频率或最后一步执行验证。
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 根据当前/下一步 profiler 状态决定是否在这里停止采样。
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # 维护历史最长 step 耗时，供后续 ESI 保存判断使用。
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # step 11: 组织训练基础指标。
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # 收集 batch 统计、奖励拆分、时延、吞吐和方差代理等诊断指标。
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                # curriculum sampler 如果存在，就在每个 batch 后根据最新训练结果更新采样策略。
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # step 12: 把本步所有指标统一打到日志后端。
                logger.log(data=metrics, step=self.global_steps)

                # 推进进度条和全局 step，准备下一轮。
                progress_bar.update(1)
                self.global_steps += 1

                # 某些 profile 模式下会在每步训练后额外导出显存快照。
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                # 最后一步需要确保异步收尾执行完成，并输出最终验证结果后退出。
                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # 某些数据集支持 batch 级别的在线更新，这里在每步结束后通知它。
                if hasattr(self.train_dataset, "on_batch_end"):
                    # 数据集可能会根据这一批训练结果调整自身状态。
                    self.train_dataset.on_batch_end(batch=batch)
