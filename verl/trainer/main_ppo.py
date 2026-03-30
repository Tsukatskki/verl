# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

# 这个文件是 PPO 训练流程的入口之一：
# - 负责读取 Hydra 配置
# - 初始化 Ray 运行时
# - 组装 Actor/Critic/Ref/Reward 相关 worker
# - 创建数据集与采样器
# - 启动 RayPPOTrainer 执行训练

# 标准库：操作系统相关接口（进程 ID、环境等）
import os

# 标准库：网络主机信息（用于打印当前 hostname）
import socket

# Hydra：用于配置管理和命令行参数覆盖
import hydra

# Ray：用于分布式执行和远程 actor 管理
import ray

# OmegaConf：Hydra 的配置对象工具，支持 merge/resolve/to_container
from omegaconf import OmegaConf

# 抽象采样器基类（用于 curriculum sampler 的类型检查）
from verl.experimental.dataset.sampler import AbstractSampler

# 兼容旧版 reward 配置字段，迁移到新版配置结构
from verl.experimental.reward_loop import migrate_legacy_reward_impl

# 获取 PPO 训练默认 Ray runtime_env（环境变量等）
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

# PPO 的核心训练器（Ray 版）
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# 根据配置判断是否需要 critic、是否需要 reference policy
from verl.trainer.ppo.utils import need_critic, need_reference_policy

# 配置校验函数，检查关键字段是否自洽
from verl.utils.config import validate_config

# 自动设置设备类型（如 Ascend 场景下自动设置 npu）与 CUDA 可用性标识
from verl.utils.device import auto_set_device, is_cuda_available

# 动态加载外部类/对象（字符串路径 -> Python 对象）
from verl.utils.import_utils import load_extern_object


# 将 main 函数注册为 Hydra 程序入口：
# - config_path="config" 表示配置目录
# - config_name="ppo_trainer" 表示默认配置名
# - version_base=None 表示不强制特定 Hydra 版本行为
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # 在 Ascend NPU 环境下，自动将 config.trainer.device 设置为 npu。
    auto_set_device(config)
    # 将旧版 reward 配置自动迁移到新版字段，避免历史配置直接报错。
    config = migrate_legacy_reward_impl(config)
    # 进入 PPO 主流程。
    run_ppo(config)


# 定义 PPO 训练主调度函数（可注入自定义 task_runner_class）。
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # 若当前进程尚未初始化 Ray，则先执行一次 Ray 初始化。
    if not ray.is_initialized():
        # 获取 PPO 默认 runtime_env（包含推荐环境变量等）。
        default_runtime_env = get_ppo_ray_runtime_env()
        # 从配置中读取 ray_init 参数（若无则用空字典）。
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        # 从 ray_init 中取 runtime_env（若无则用空字典）。
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # 如果启用了 transfer queue，则把对应环境变量塞进 runtime_env。
        if config.transfer_queue.enable:
            # 先取已有 env_vars，避免覆盖用户已有设置。
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            # 设置传输队列开关。
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            # 写回 runtime_env。
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # 显式透传当前进程的可见卡与常用 CUDA/NCCL 环境变量。
        # 否则 Ray worker 可能会回退到宿主机的全量 GPU 视图，导致进程落到错误的物理卡上。
        inherited_env_keys = [
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NCCL_P2P_DISABLE",
            "CUDA_DEVICE_MAX_CONNECTIONS",
        ]
        runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
        for env_key in inherited_env_keys:
            env_val = os.environ.get(env_key)
            if env_val:
                runtime_env_vars[env_key] = env_val
        runtime_env_kwargs["env_vars"] = runtime_env_vars

        # 将默认 runtime_env 与用户 runtime_env 进行 merge（用户配置优先覆盖）。
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # 回填到 ray_init_kwargs 并保持 OmegaConf 结构。
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # 打印实际生效的 Ray 初始化参数，便于排查。
        print(f"ray init kwargs: {ray_init_kwargs}")
        # 把 OmegaConf 转成原生 dict 后调用 ray.init。
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 若未显式传入 task runner，就默认把 TaskRunner 包成一个 Ray 远程 actor。
    if task_runner_class is None:
        # num_cpus=1: 确保这个控制 actor 有固定 CPU 资源，且尽量不占用 head 的关键资源。
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # 以下分支决定是否给控制 actor 注入 Nsight profiler 运行时配置。
    if (
        # CUDA 可用（当前条件里写的是函数/变量符号，保持原始行为不变）
        is_cuda_available
        # profiler 工具选择 nsys
        and config.global_profiler.tool == "nsys"
        # 配置中提供了 steps 字段
        and config.global_profiler.get("steps") is not None
        # steps 非空
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        # 仅在需要 Nsight 时才导入，减少无关依赖影响。
        from verl.utils.import_utils import is_nvtx_available

        # nsys + nvtx 需要 nvtx 包。
        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        # 读取 controller 侧 nsys 参数并转成原生容器。
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        # 创建远程 actor，并通过 runtime_env 注入 nsight 参数。
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # 默认路径：直接创建远程 actor。
        runner = task_runner_class.remote()
    # 异步发起 runner.run，再通过 ray.get 阻塞等待训练结束。
    ray.get(runner.run.remote(config))

    # 可选：导出 Ray timeline 追踪文件，用于后续性能分析。
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        # 生成 timeline json。
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        # 角色 -> worker 类（Ray remote class）映射。
        self.role_worker_mapping = {}
        # 角色 -> 资源池 ID 映射。
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        # RayWorkerGroup 负责组织一组同类 worker。
        from verl.single_controller.ray import RayWorkerGroup

        # Role 定义了训练中不同角色枚举（ActorRollout/Critic/Ref 等）。
        from verl.trainer.ppo.ray_trainer import Role

        # auto/enable/disable：用于选择 legacy/new worker 实现。
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # 新模型引擎实现：把 actor/rollout/ref 统一到 engine worker 体系。
        if use_legacy_worker_impl == "disable":
            # 新实现中的 ActorRolloutRefWorker。
            from verl.workers.engine_workers import ActorRolloutRefWorker

            # 具体 worker 类。
            actor_rollout_cls = ActorRolloutRefWorker
            # worker group 实现类。
            ray_worker_group_cls = RayWorkerGroup

            # 优先读取新版 lora.rank。
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            # 兼容旧版 lora_rank 字段。
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            # 如果有 LoRA（rank>0 或 adapter_path 存在），则 ref 逻辑可并入 actor。
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            # 新引擎里：ref policy 与 actor rollout 可以同 worker。
            # 旧引擎里：ref 通常是独立 worker。
            if need_reference_policy(config) and not ref_in_actor:
                # 需要 reference policy 且未并入 actor 时，角色记为 ActorRolloutRef。
                role = Role.ActorRolloutRef
            else:
                # 否则仅注册为 ActorRollout。
                role = Role.ActorRollout
            # 角色映射到远程 worker 类。
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            # actor 角色使用 global_pool。
            self.mapping[role] = "global_pool"
            # 返回给调用方用于后续 trainer 初始化。
            return actor_rollout_cls, ray_worker_group_cls

        # legacy 分支：同步模式已废弃，因此统一使用异步 worker。
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # FSDP/FSDP2 对应 fsdp 异步实现。
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron 对应 megatron 异步实现。
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif (
            config.actor_rollout_ref.actor.strategy == "veomni"
            or config.actor_rollout_ref.actor.strategy == "torchtitan"
        ):
            # veomni/torchtitan 不支持 legacy worker。
            raise NotImplementedError(
                f"{config.actor_rollout_ref.actor.strategy} does not support legacy worker implementation"
            )

        else:
            # 其他未知策略直接报未实现。
            raise NotImplementedError

        # 注册 ActorRollout 角色的 worker。
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        # 绑定到全局资源池。
        self.mapping[Role.ActorRollout] = "global_pool"
        # 返回选中的 worker 类和 group 类。
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        # 获取 worker 实现选择开关。
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        # 按 critic.strategy 选择实现。
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                # legacy 路径：使用 fsdp 的 CriticWorker。
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # 新 worker 路径：critic 直接复用通用 TrainingWorker。
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                # 非法配置值。
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            # 目前 megatron 仍使用专用 CriticWorker（未来可切通用 worker）。
            from verl.workers.megatron_workers import CriticWorker

        elif config.critic.strategy == "veomni" or config.critic.strategy == "torchtitan":
            if use_legacy_worker_impl == "disable":
                # veomni/torchtitan 仅在新 worker 模式下支持。
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
                print(f"Using new worker implementation for {config.critic.strategy}")
            else:
                # 若不是 disable，直接报错提示配置无效。
                raise ValueError(
                    f"Invalid use_legacy_worker_impl for {config.critic.strategy}: {use_legacy_worker_impl}"
                )

        else:
            # 未支持的 critic 策略。
            raise NotImplementedError

        # 导入角色枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 注册 Critic 角色远程 worker。
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        # Critic 绑定到全局资源池。
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        # 定义默认全局资源池 ID。
        global_pool_id = "global_pool"
        # 资源池规格：每个节点 GPU 数为 n_gpus_per_node，共 nnodes 个节点。
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # 如果 reward model 单独启用资源池，则创建 reward_pool。
        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                # 校验 reward pool 的每节点 GPU 数必须 > 0。
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                # 校验 reward pool 节点数必须 > 0。
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            # 构造 reward_pool 的规格列表。
            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            # 写入资源池配置。
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            # 若未启用独立 reward_pool，则 reward 复用 trainer 的资源规模。
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        print(
            "[init_resource_pool_mgr] "
            f"reward.enable={config.reward.reward_model.enable}, "
            f"reward.enable_resource_pool={config.reward.reward_model.enable_resource_pool}, "
            f"resource_pool_spec={resource_pool_spec}"
        )

        # 资源池管理器：负责按角色映射分配到对应 pool。
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        # 创建资源池管理器实例。
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        # 返回管理器给 trainer 使用。
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        # 导入角色枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 仅当 reward model enable 时才处理映射。
        if config.reward.reward_model.enable:
            # 当前实现不注册 reward model worker，仅注册其资源池映射。
            if config.reward.reward_model.enable_resource_pool:
                # 启用独立池时映射到 reward_pool。
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                # 否则映射到 global_pool。
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        # 导入角色枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 新 worker 模式下，ref policy 已融合进 ActorRolloutRefWorker，无需单独 worker。
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            # 直接返回。
            return

        # 仅当配置需要 reference policy 时才注册。
        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # 打印配置时用到的 pretty print。
        from pprint import pprint

        # 将远端模型/检查点路径复制到本地（如 HDFS -> local）。
        from verl.utils.fs import copy_to_local

        # 打印当前执行节点信息，便于分布式排障。
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # 打印解析后的配置字典。
        pprint(OmegaConf.to_container(config, resolve=True))
        # 原地 resolve 配置中的引用/插值。
        OmegaConf.resolve(config)

        # 根据策略选择并注册 actor/rollout worker。
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # 注册 critic worker。
        self.add_critic_worker(config)

        # 注册 reward model 的资源池映射。
        self.add_reward_model_resource_pool(config)

        # 若 KL loss/reward 需要 ref policy，则注册 ref worker（legacy 模式）。
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 进行配置一致性校验。
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # 将模型路径复制到本地；可选 use_shm 以利用共享内存加速加载。
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # 导入 tokenizer / processor 构建函数。
        from verl.utils import hf_processor, hf_tokenizer

        # 是否允许执行远端仓库自定义代码（transformers trust_remote_code）。
        trust_remote_code = config.data.get("trust_remote_code", False)
        # 基于本地模型路径创建 tokenizer。
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # 多模态场景可用的 processor（纯文本模型时可能为 None）。
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 初始化资源池管理器。
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # 批处理拼接函数（DataLoader collate_fn）。
        from verl.utils.dataset.rl_dataset import collate_fn

        # 创建训练集。
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        # 创建验证集。
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        # 创建训练采样器（支持 curriculum / 随机 / 顺序采样）。
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 初始化 PPO Trainer。
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # 初始化 trainer 内部各类 worker group。
        trainer.init_workers()

        # 正式进入训练循环。
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    # 当前函数签名保留 is_train 以兼容上层调用与未来扩展；当前实现暂未使用。
    _ = is_train

    # 按配置选择具体的数据集实现类。
    from verl.utils.dataset.rl_dataset import get_dataset_class

    # 获取数据集类（例如常规 RL 数据集、特殊格式数据集等）。
    dataset_cls = get_dataset_class(data_config)

    # 使用选定类实例化数据集对象。
    dataset = dataset_cls(
        # 数据文件路径列表。
        data_files=data_paths,
        # 分词器。
        tokenizer=tokenizer,
        # 多模态处理器（可为 None）。
        processor=processor,
        # 数据配置。
        config=data_config,
        # 最大采样条目数（-1 表示不截断）。
        max_samples=max_samples,
    )

    # 返回数据集实例。
    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    # 引入 torch 以创建随机数发生器。
    import torch

    # 顺序采样器（不打乱）。
    from torch.utils.data import SequentialSampler

    # 使用 torchdata 的 RandomSampler，便于与状态恢复（checkpoint resume）配合。
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # 若 data_config 指定了自定义 sampler 类路径，则优先使用 curriculum sampler。
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        # 动态加载外部 sampler 类。
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        # 实例化 sampler，并传入数据集和数据配置。
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        # 运行时类型校验：必须继承 AbstractSampler。
        assert isinstance(sampler, AbstractSampler)
        # 使用 curriculum 时，强制 num_workers=0，避免 dataloader 预缓存打乱课程调度顺序。
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # 未指定 curriculum 时，若配置要求 shuffle，则创建随机采样器。
    elif data_config.shuffle:
        # 创建一个独立的生成器，便于控制可复现随机性。
        train_dataloader_generator = torch.Generator()
        # 读取用户配置的随机种子。
        seed = data_config.get("seed")
        if seed is not None:
            # 设定采样器随机种子，确保可复现。
            train_dataloader_generator.manual_seed(seed)
        # 构建随机采样器。
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # 若不打乱，则按顺序采样。
        sampler = SequentialSampler(data_source=dataset)

    # 返回最终 sampler。
    return sampler


if __name__ == "__main__":
    # 作为脚本直接执行时，从 main 入口启动。
    main()
