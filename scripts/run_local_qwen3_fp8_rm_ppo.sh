#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" ]] && continue
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue
        if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            if [[ -z "${!key+x}" ]]; then
                export "${key}=${value}"
            fi
        fi
    done < "${ENV_FILE}"
fi

################################################################################
# ====================== 环境与运行时配置 ======================
# 以下变量控制 CUDA / NCCL / vLLM / HuggingFace 等底层运行环境，
# 作用于整个训练流程（训练、推理、奖励模型推理）的底层设备和通信层。
################################################################################

# CUDA 设备最大并发连接数；设为 1 可减少 GPU 间通信冲突，适用于 Megatron 等需要确定性通信顺序的场景
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# 禁用 NCCL 的 P2P（点对点）传输；在某些多 GPU 拓扑下 P2P 不稳定时设为 1 以回退到共享内存通信
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# 指定 PyTorch JIT 编译 CUDA kernel 时的目标 GPU 架构；8.9 对应 Ada Lovelace（RTX 4090 等）
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
# verl 框架整体日志级别，控制框架内部日志输出的详细程度（DEBUG/INFO/WARNING/ERROR）
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
# PPO 训练循环专用日志级别，可单独调高以调试 PPO 阶段的问题
export VERL_PPO_LOGGING_LEVEL="${VERL_PPO_LOGGING_LEVEL:-INFO}"
# HuggingFace tokenizers 是否使用多线程并行；设为 true 可加速分词，但在 fork 进程中可能死锁
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
# 是否启用 vLLM V1 引擎（新版推理引擎）；作用于 Rollout 阶段的推理后端选择
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# 是否使用 FlashInfer 采样器（替代默认采样器）；作用于 Rollout 阶段的 token 采样策略
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# vLLM 使用的注意力计算后端；FLASH_ATTN 使用 FlashAttention 加速注意力计算，降低显存占用
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
# 是否启用 HuggingFace Hub 离线模式；设为 1 阻止从网络下载模型/分词器，必须使用本地缓存
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# 是否启用 Transformers 库离线模式；与上面配合确保完全离线运行
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Hydra 配置框架报错时是否显示完整堆栈；设为 1 便于定位配置解析错误
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
# 禁用 Python 输出缓冲，确保日志实时输出（对调试和日志收集至关重要）
export PYTHONUNBUFFERED=1

PYTHON_RUNNER=()
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    PYTHON_RUNNER=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" env PYTHONUNBUFFERED=1)
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export WANDB_PROJECT
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    export WANDB_ENTITY
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
    export WANDB_NAME
fi

################################################################################
# ====================== GPU 与分布式配置 ======================
# 控制多 GPU 训练的基本拓扑，作用于 Actor/Critic 训练及奖励模型推理阶段。
################################################################################

# 总 GPU 数量；决定了训练进程的并行度，影响所有使用 GPU 的阶段
NUM_GPUS="${NUM_GPUS:-2}"
# 训练器使用的 GPU 数量（Actor/Critic 训练阶段）；默认等于 NUM_GPUS
TRAINER_NUM_GPUS="${TRAINER_NUM_GPUS:-${NUM_GPUS}}"
# 奖励模型是否启用独立资源池；为 True 时 RM 在独立的 GPU 资源池中运行，避免与训练争抢 GPU
RM_ENABLE_RESOURCE_POOL="${RM_ENABLE_RESOURCE_POOL:-False}"
# 奖励模型每个节点使用的 GPU 数量；作用于奖励评估阶段
RM_NUM_GPUS_PER_NODE="${RM_NUM_GPUS_PER_NODE:-1}"
# 奖励模型使用的节点数量；多节点时可扩展 RM 的推理能力
RM_NNODES="${RM_NNODES:-1}"
# 训练后端选择：fsdp（PyTorch 原生分布式）或 megatron（NVIDIA Megatron-LM）；
# 决定了 Actor/Critic 训练阶段使用哪种并行训练框架
TRAIN_BACKEND="${TRAIN_BACKEND:-fsdp}"
# FSDP 具体策略：fsdp2 是 PyTorch 的 FSDP2（DTensor 版本），更高效的参数分片方式
TRAIN_STRATEGY="${TRAIN_STRATEGY:-fsdp2}"
# Hydra 配置文件名；决定加载哪个训练配置模板（ppo_trainer 或 ppo_megatron_trainer）
CONFIG_NAME="${CONFIG_NAME:-ppo_trainer}"

################################################################################
# ====================== 模型路径配置 ======================
# 指定 Actor（策略模型）和 Reward Model（奖励模型）的 HuggingFace 模型路径，
# 作用于模型加载阶段（训练开始前的初始化）。
################################################################################

# Actor/Critic/Ref 模型的 HuggingFace 模型 ID（用于标识模型）
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-FP8}"
# Actor/Critic/Ref 模型的实际路径（可以是本地路径或 HF Hub ID）；加载模型权重时使用
MODEL_PATH="${MODEL_PATH:-${MODEL_ID}}"
# 奖励模型的 HuggingFace 模型 ID
RM_MODEL_ID="${RM_MODEL_ID:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
# 奖励模型的实际路径；作用于奖励评估阶段加载 RM 权重
RM_MODEL_PATH="${RM_MODEL_PATH:-${RM_MODEL_ID}}"

################################################################################
# ====================== 数据配置 ======================
# 控制训练数据的加载与预处理，作用于数据准备阶段（DataLoader）。
################################################################################

# 训练数据文件路径（parquet 格式）；提供 prompt 用于 Rollout 生成和策略优化
TRAIN_FILES="${TRAIN_FILES:-data/ultrafeedback/train.parquet}"
# 验证数据文件路径；用于训练过程中的效果评估
VAL_FILES="${VAL_FILES:-data/ultrafeedback/test.parquet}"
# 数据集中 prompt 字段的键名；告诉 DataLoader 从哪个列读取提示文本
PROMPT_KEY="${PROMPT_KEY:-prompt}"
# 训练批量大小（每步总样本数）；作用于每个 PPO 迭代步骤中用于 Rollout 和训练的样本总量
# 增大可提升训练稳定性但增加显存和计算量
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# 最大 prompt 长度（token 数）；超过此长度的 prompt 会被过滤或截断（取决于 truncation 设置）
# 作用于数据预处理阶段，通过限制输入长度来控制显存使用
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
# 最大回复长度（token 数）；限制 Rollout 阶段模型生成的最大 token 数，
# 直接影响 Rollout 的生成时间和显存占用
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
# 是否返回原始对话格式（而非经过模板处理的文本）；作用于数据预处理阶段
DATA_RETURN_RAW_CHAT="${DATA_RETURN_RAW_CHAT:-False}"
# DataLoader 的工作进程数；0 表示在主进程中加载数据，可避免多进程数据加载的兼容性问题
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
# 验证集最大样本数；-1 表示使用全部验证数据，设较小值可加速验证
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-100}"
# 验证集 batch 大小；null 使用全部验证集作为一个 batch
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-100}"

################################################################################
# ====================== PPO 训练超参数 ======================
# 控制 PPO（Proximal Policy Optimization）算法的核心训练行为，
# 作用于 Actor（策略模型）和 Critic（价值模型）的训练阶段。
################################################################################

# Actor（策略模型）的学习率；作用于 Actor 训练阶段，控制策略梯度更新的步长
# 过大会导致训练不稳定，过小会收敛过慢
ACTOR_LR="${ACTOR_LR:-1e-6}"
# Actor 学习率预热步数比例；用于 warmup + decay 调度
ACTOR_LR_WARMUP_STEPS_RATIO="${ACTOR_LR_WARMUP_STEPS_RATIO:-0.0}"
# Actor 学习率预热初始值；默认从 0 开始 warmup
ACTOR_LR_WARMUP_INIT="${ACTOR_LR_WARMUP_INIT:-0.0}"
# Actor 学习率衰减总步数；默认由后端使用 total_training_steps
ACTOR_LR_DECAY_STEPS="${ACTOR_LR_DECAY_STEPS:-null}"
# Actor 学习率衰减风格；Megatron 支持 constant/linear/cosine/inverse_square_root
ACTOR_LR_DECAY_STYLE="${ACTOR_LR_DECAY_STYLE:-constant}"
# Actor 最小学习率；用于 decay 终点
ACTOR_MIN_LR="${ACTOR_MIN_LR:-0.0}"
# Critic（价值模型）的学习率；作用于 Critic 训练阶段，通常比 Actor 大以加速价值函数拟合
CRITIC_LR="${CRITIC_LR:-5e-6}"
# KL 散度损失系数；作用于 Actor 训练阶段，通过在损失函数中添加 KL 惩罚项，
# 约束策略模型不偏离参考模型（Ref）太远，防止策略崩溃（reward hacking）
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
# KL 损失计算方式；low_var_kl 是低方差 KL 估计器，相比标准 KL 散度方差更小、训练更稳定
# 作用于 Actor 训练阶段的 KL 惩罚计算
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
# 每个训练步 Actor 对同一批数据重复训练的轮数；增大可更充分利用样本但可能过拟合
# 作用于 Actor 训练阶段，控制梯度更新的总次数
PPO_EPOCHS="${PPO_EPOCHS:-1}"
# PPO 小批量大小；在一个 PPO epoch 中将训练数据划分为若干 mini-batch 依次更新
# 作用于 Actor/Critic 训练阶段，控制每次梯度更新使用的样本数
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
# 每个 GPU 上的微批量大小；控制梯度累积粒度，通过减小该值可降低单卡显存峰值
# 作用于 Actor/Critic 训练阶段的前向/反向传播
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
# Actor 每个 GPU 上每个微批次允许的最大 token 总长度；
# 作用于 Actor 训练阶段，通过限制 token 数量来控制显存使用（动态批大小的上限）
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
# Critic 每个 GPU 上每个微批次允许的最大 token 总长度；
# 作用于 Critic 训练阶段，功能同上但独立控制 Critic 的显存
CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU="${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU:-2048}"
# 优势函数估计器类型：gae（Generalized Advantage Estimation）使用 Critic 预测的价值函数；
# grpo（Group Relative Policy Optimization）不需要 Critic，通过组内相对排名计算优势
# 作用于优势计算阶段，决定如何计算每个 token/序列的优势值
ADV_ESTIMATOR="${ADV_ESTIMATOR:-gae}"

################################################################################
# ====================== LoRA 配置 ======================
# 控制 LoRA（Low-Rank Adaptation）低秩适配参数，作用于 Actor/Critic 模型的训练阶段，
# 通过只训练低秩分解矩阵来大幅减少可训练参数量和显存占用。
################################################################################

# LoRA 秩（rank）；作用于 Actor 模型，决定低秩矩阵的维度。
# 秩越大表达能力越强但可训练参数越多、显存越大；0 表示禁用 LoRA（全参数训练）
LORA_RANK="${LORA_RANK:-16}"
# Critic 模型的 LoRA 秩；默认跟随 Actor 的 LORA_RANK
CRITIC_LORA_RANK="${CRITIC_LORA_RANK:-${LORA_RANK}}"
# LoRA 缩放系数（alpha）；实际缩放因子为 alpha/rank，控制 LoRA 权重对原始权重的影响强度
# alpha 越大 LoRA 分支的输出越强，需要和 rank 配合调整
LORA_ALPHA="${LORA_ALPHA:-32}"
# LoRA dropout 比例；作用于训练阶段，在 LoRA 层中随机丢弃部分激活以防止过拟合
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
# LoRA 应用的目标模块列表；指定哪些线性层使用 LoRA 适配
# linear_qkv: 注意力 QKV 投影层, linear_proj: 注意力输出投影层
# linear_fc1/fc2: FFN 前馈网络的两个线性层
MEGATRON_LORA_TARGET_MODULES_DEFAULT="['linear_qkv','linear_proj','linear_fc1','linear_fc2']"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-${MEGATRON_LORA_TARGET_MODULES_DEFAULT}}"
# Critic 模型的 LoRA 目标模块；默认与 Actor 相同
CRITIC_LORA_TARGET_MODULES="${CRITIC_LORA_TARGET_MODULES:-${LORA_TARGET_MODULES}}"
# 是否在保存时将 LoRA 权重合并到基础模型；False 表示单独保存 LoRA 增量权重
LORA_MERGE="${LORA_MERGE:-False}"
# 是否启用梯度检查点（gradient checkpointing）；作用于 Actor/Critic 训练阶段，
# 通过用计算换显存（反向传播时重新计算中间激活而非保存），大幅降低训练显存占用
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
# 是否使用 torch.compile 编译模型；作用于训练阶段，可通过静态图优化加速训练
# 但编译耗时且可能与某些动态特性不兼容
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"

################################################################################
# ====================== 并行策略配置（Rollout / RM / Megatron）======================
# 控制推理和训练的张量并行度，作用于 Rollout 生成、奖励模型推理、
# 以及 Megatron 后端训练中的模型并行切分方式。
################################################################################

# Rollout 阶段（vLLM 推理引擎）的张量并行度；将模型按张量维度切分到多 GPU 上
# 增大可推理更大模型，但增加通信开销。默认等于训练 GPU 数
ROLLOUT_TP="${ROLLOUT_TP:-${TRAINER_NUM_GPUS}}"
# 奖励模型推理的张量并行度；作用于 RM 评分阶段
RM_TP="${RM_TP:-1}"

# ===== Megatron 并行度配置（仅 TRAIN_BACKEND=megatron 时生效）=====

# Megatron 张量并行度（Tensor Parallel）；将每层的权重矩阵按列/行切分到多 GPU
# 作用于 Actor/Critic/Ref 的训练前向和反向传播，降低单卡显存
MEGATRON_TP="${MEGATRON_TP:-${TRAINER_NUM_GPUS}}"
# Megatron 流水线并行度（Pipeline Parallel）；将模型的不同层分配到不同 GPU 上
# 适合超深模型，通过流水线调度隐藏通信延迟
MEGATRON_PP="${MEGATRON_PP:-1}"
# Megatron 上下文并行度（Context Parallel）；将长序列沿 token 维度切分到多 GPU
# 用于处理超长上下文序列（如 128K+ tokens）
MEGATRON_CP="${MEGATRON_CP:-1}"
# Megatron 专家并行度（Expert Parallel）；用于 MoE（混合专家）模型，
# 将不同专家分配到不同 GPU 上并行计算
MEGATRON_EP="${MEGATRON_EP:-1}"
# Megatron 专家张量并行度（Expert Tensor Parallel）；在 EP 基础上对单个专家再做张量并行
MEGATRON_ETP="${MEGATRON_ETP:-null}"
# Megatron 虚拟流水线并行度（Virtual Pipeline Parallel）；
# 将每个流水线阶段的层拆分为多个虚拟阶段，减少流水线气泡（bubble）
MEGATRON_VPP="${MEGATRON_VPP:-null}"
# 是否使用 Megatron-Bridge（与 HuggingFace 模型格式的转换桥）；
# 启用后可直接加载 HF 格式的模型权重，LoRA 训练时必须启用
MEGATRON_USE_MBRIDGE="${MEGATRON_USE_MBRIDGE:-True}"
# 是否使用原始（vanilla）Megatron-Bridge；False 表示使用增强版 Bridge，支持 LoRA 等特性
MEGATRON_VANILLA_MBRIDGE="${MEGATRON_VANILLA_MBRIDGE:-False}"
# 是否启用序列并行（Sequence Parallel）；与张量并行配合，
# 将 LayerNorm 和 Dropout 的计算沿序列维度分片，进一步降低激活内存
MEGATRON_SEQUENCE_PARALLEL="${MEGATRON_SEQUENCE_PARALLEL:-True}"

################################################################################
# ====================== 显存卸载配置（CPU Offload）======================
# 控制将模型参数、优化器状态、梯度从 GPU 卸载到 CPU 内存，
# 通过牺牲速度来换取 GPU 显存空间。作用于 Actor/Critic/Ref 的训练阶段。
################################################################################

# Actor 模型参数卸载到 CPU；训练时按需将参数从 CPU 移回 GPU，降低常驻 GPU 显存
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
# Actor 优化器状态（如 Adam 的 m/v 矩阵）卸载到 CPU；这些状态通常占模型参数 2-3 倍显存
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
# Actor 梯度卸载到 CPU；反向传播中产生的梯度存到 CPU 以释放 GPU 显存
ACTOR_GRAD_OFFLOAD="${ACTOR_GRAD_OFFLOAD:-False}"
# Critic 模型参数卸载到 CPU
CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-False}"
# Critic 优化器状态卸载到 CPU
CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-False}"
# Critic 梯度卸载到 CPU
CRITIC_GRAD_OFFLOAD="${CRITIC_GRAD_OFFLOAD:-False}"
# 参考模型（Ref）参数卸载到 CPU；Ref 模型不训练只做前向推理，卸载到 CPU 可释放大量显存
# 默认 True 因为 Ref 模型不需要常驻 GPU
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"

################################################################################
# ====================== 数据类型与量化 ======================
# 控制模型的数值精度，作用于训练和推理阶段的显存占用和计算速度。
################################################################################

# 模型训练精度；bfloat16 是 16 位浮点数，相比 fp32 节省一半显存且范围足够大
# 作用于 Actor/Critic 训练阶段的权重和梯度精度
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
# Rollout 推理精度；作用于 vLLM 生成阶段
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
# Rollout 推理量化方式；null 表示不量化，可设为 fp8/awq/gptq 等以降低推理显存
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-null}"
# 模型权重加载格式；auto 表示自动检测，也可指定 safetensors/pt 等格式
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-auto}"

################################################################################
# ====================== Rollout（生成/推理）配置 ======================
# 控制 Actor 模型在 Rollout 阶段的推理行为（使用 vLLM 引擎生成回复）。
# Rollout 是 PPO 训练循环的第一阶段：用当前策略生成回复以收集经验。
################################################################################

# 推理引擎名称；vllm 使用 vLLM 高性能推理引擎，也可选 sglang 等
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
# Rollout 运行模式；async 表示异步推理（与训练重叠执行以提升吞吐），sync 表示同步
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
# 每个 prompt 生成的回复数量；n>1 时每个 prompt 采样多条回复，
# 增大可提高优势估计的准确性（尤其 GRPO 需要 n>1 做组内对比），但增加计算量
ROLLOUT_N="${ROLLOUT_N:-1}"
# 是否禁用 vLLM 的自定义 AllReduce 通信；True 使用 NCCL 标准 AllReduce，
# 在某些 GPU 拓扑/驱动下更稳定
ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-True}"
# vLLM 引擎的 GPU 显存利用率上限（0~1）；控制 vLLM KV Cache 占用的 GPU 显存比例
# 需要留够显存给训练阶段，所以通常设较小值（如 0.35）
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
# vLLM 支持的最大序列总长度（prompt+response）；决定 KV Cache 能容纳的最大上下文长度
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-1024}"
# vLLM 单次调度批中允许的最大 token 总数；控制推理时的最大并发 token 处理量
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}"
# vLLM 同时处理的最大序列数；控制推理时的最大并发请求数，影响吞吐和显存
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
# 计算 log prob 时每个 GPU 的微批量大小；作用于 log_prob 计算（Actor/Ref 的对数概率），
# 减小可降低显存但增加计算时间
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
# 是否强制使用 eager 模式（禁用 CUDA Graph）；True 避免 CUDA Graph 编译开销，
# 适合序列长度变化大的场景，但会略降推理速度
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
# 推理完成后是否释放 KV Cache 引擎显存；True 释放后训练阶段可使用更多显存，
# 但下一次 Rollout 需要重新分配。适合训练和推理交替执行的场景
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-True}"
# Rollout agent 的 worker 数量；作用于多轮对话等需要 agent 的场景
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"

################################################################################
# ====================== 奖励模型（Reward Model）配置 ======================
# 控制外部奖励模型的推理行为，作用于 PPO 训练循环的奖励评估阶段：
# 对 Rollout 生成的回复进行打分，提供训练信号。
################################################################################

# 是否启用奖励模型；False 时使用规则函数（如代码执行等）计算奖励
REWARD_MODEL_ENABLE="${REWARD_MODEL_ENABLE:-True}"
# 奖励模型推理 worker 数量；增大可并行评估更多回复，加速奖励计算
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-2}"
# 奖励模型是否禁用自定义 AllReduce（同 Rollout 同类参数）
RM_DISABLE_CUSTOM_ALL_REDUCE="${RM_DISABLE_CUSTOM_ALL_REDUCE:-True}"
# 奖励模型 vLLM 引擎的 GPU 显存利用率上限；控制 RM KV Cache 的显存占比
RM_GPU_MEMORY_UTILIZATION="${RM_GPU_MEMORY_UTILIZATION:-0.20}"
# 奖励模型支持的最大序列长度；需要能容纳 prompt + response
RM_MAX_MODEL_LEN="${RM_MAX_MODEL_LEN:-512}"
# 奖励模型单次批处理的最大 token 数
RM_MAX_NUM_BATCHED_TOKENS="${RM_MAX_NUM_BATCHED_TOKENS:-256}"
# 奖励模型同时处理的最大序列数
RM_MAX_NUM_SEQS="${RM_MAX_NUM_SEQS:-2}"
# 奖励模型权重加载格式
RM_ROLLOUT_LOAD_FORMAT="${RM_ROLLOUT_LOAD_FORMAT:-auto}"

################################################################################
# ====================== 训练器（Trainer）配置 ======================
# 控制整体训练流程的运行参数，作用于训练主循环的调度与日志记录。
################################################################################

# 项目名称；用于 W&B（Weights & Biases）等日志平台的项目归类
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-verl_local}"
# 实验名称；用于区分同一项目下的不同训练实验
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-qwen3_8b_fp8_rm_ppo}"
# 默认日志后端；console 输出到终端
DEFAULT_TRAINER_LOGGER='[console]'
TRAINER_LOGGER_WAS_SET=0
if [[ -n "${TRAINER_LOGGER+x}" ]]; then
    TRAINER_LOGGER_WAS_SET=1
fi
# 日志后端列表；可选 console（终端输出）、wandb（W&B 平台）、tensorboard 等
TRAINER_LOGGER="${TRAINER_LOGGER:-${DEFAULT_TRAINER_LOGGER}}"
# 模型保存频率（每隔多少步保存一次 checkpoint）；-1 表示不保存
SAVE_FREQ="${SAVE_FREQ:--1}"
# 测试/验证频率（每隔多少步执行一次验证）；-1 表示不验证
TEST_FREQ="${TEST_FREQ:--1}"
# 是否在训练开始前先执行一次验证；用于获取训练前的基线指标
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# 总训练步数；每步包含完整的 Rollout → 奖励计算 → 优势估计 → Actor/Critic 更新 流程
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5000}"
# 断点续训模式；disable 表示从头训练，auto 表示自动检测并恢复最新 checkpoint
RESUME_MODE="${RESUME_MODE:-disable}"
# 是否使用旧版 worker 实现；auto 表示自动选择
USE_LEGACY_WORKER_IMPL="${USE_LEGACY_WORKER_IMPL:-auto}"

# 训练输出目录；显式固定 Hydra run dir，并把 stdout/stderr 同步落盘，方便与 W&B 对照排查
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
RUN_TIME="${RUN_TIME:-$(date +%H-%M-%S)}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${ROOT_DIR}/outputs/${RUN_DATE}/${RUN_TIME}}"
RUN_STDOUT_LOG="${RUN_STDOUT_LOG:-${RUN_OUTPUT_DIR}/stdout.log}"

if [[ "${TRAINER_LOGGER_WAS_SET}" -eq 0 ]] && [[ -n "${WANDB_API_KEY:-}" ]] && [[ -n "${WANDB_PROJECT:-}" ]]; then
    TRAINER_LOGGER='[console,wandb]'
    TRAINER_PROJECT_NAME="${WANDB_PROJECT}"
fi

if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
    CONFIG_NAME="ppo_megatron_trainer.yaml"
    if [[ "${LORA_RANK}" != "0" ]] && [[ "${MEGATRON_USE_MBRIDGE}" != "True" ]]; then
        echo "Megatron LoRA requires Megatron-Bridge; forcing MEGATRON_USE_MBRIDGE=True" >&2
        MEGATRON_USE_MBRIDGE="True"
    fi
    if [[ "${LORA_RANK}" != "0" ]] && [[ "${MEGATRON_VANILLA_MBRIDGE}" != "False" ]]; then
        echo "Megatron LoRA requires the Megatron-Bridge backend; forcing MEGATRON_VANILLA_MBRIDGE=False" >&2
        MEGATRON_VANILLA_MBRIDGE="False"
    fi
    if [[ "${LORA_TARGET_MODULES}" == "all-linear" ]]; then
        echo "Megatron LoRA does not support target_modules=all-linear; using ${MEGATRON_LORA_TARGET_MODULES_DEFAULT} instead" >&2
        LORA_TARGET_MODULES="${MEGATRON_LORA_TARGET_MODULES_DEFAULT}"
    fi
    if [[ "${CRITIC_LORA_TARGET_MODULES}" == "all-linear" ]]; then
        echo "Megatron critic LoRA does not support target_modules=all-linear; using ${MEGATRON_LORA_TARGET_MODULES_DEFAULT} instead" >&2
        CRITIC_LORA_TARGET_MODULES="${MEGATRON_LORA_TARGET_MODULES_DEFAULT}"
    fi
else
    CONFIG_NAME="${CONFIG_NAME:-ppo_trainer}"
fi

################################################################################
# ====================== 构建训练命令 ======================
# 以下将上述超参数通过 Hydra 配置覆盖传递给 verl 训练主入口。
################################################################################
CMD=(
    "${PYTHON_RUNNER[@]}"
    python -m verl.trainer.main_ppo
    --config-name="${CONFIG_NAME}"
    "hydra.run.dir=${RUN_OUTPUT_DIR}"

    # ---- 算法配置 ----
    algorithm.adv_estimator="${ADV_ESTIMATOR}"         # 优势估计器类型（gae/grpo）

    # ---- 数据配置 ----
    data.train_files="${TRAIN_FILES}"                  # 训练数据路径
    data.val_files="${VAL_FILES}"                      # 验证数据路径
    data.prompt_key="${PROMPT_KEY}"                    # prompt 字段键名
    data.train_batch_size="${TRAIN_BATCH_SIZE}"        # 每步训练的样本总数
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"      # prompt 最大 token 长度
    data.max_response_length="${MAX_RESPONSE_LENGTH}"  # 生成回复的最大 token 长度
    data.return_raw_chat="${DATA_RETURN_RAW_CHAT}"     # 是否返回原始对话格式
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}"  # 数据加载并行进程数
    data.filter_overlong_prompts=True                  # 过滤超长 prompt（超过 max_prompt_length）
    data.truncation=error                              # 超长处理策略：error 表示报错而非截断
    data.val_max_samples="${VAL_MAX_SAMPLES}"           # 验证集最大样本数
    data.val_batch_size="${VAL_BATCH_SIZE}"             # 验证集 batch 大小

    # ---- Actor/Ref 模型配置 ----
    actor_rollout_ref.model.path="${MODEL_PATH}"       # 模型权重路径
    actor_rollout_ref.model.use_shm=False              # 是否使用共享内存加载模型
    actor_rollout_ref.model.use_remove_padding=True    # 是否移除 padding 以节省计算
    actor_rollout_ref.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}"  # 梯度检查点

    # ---- Actor 训练配置 ----
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"     # Actor 学习率
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_LR_WARMUP_STEPS_RATIO}"  # Actor 学习率 warmup 比例
    actor_rollout_ref.actor.optim.lr_warmup_init="${ACTOR_LR_WARMUP_INIT}"  # Actor warmup 初始学习率
    actor_rollout_ref.actor.optim.lr_decay_steps="${ACTOR_LR_DECAY_STEPS}"  # Actor 学习率衰减步数
    actor_rollout_ref.actor.optim.lr_decay_style="${ACTOR_LR_DECAY_STYLE}"  # Actor 学习率衰减策略
    actor_rollout_ref.actor.optim.min_lr="${ACTOR_MIN_LR}"  # Actor 最小学习率
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" # PPO 训练轮数
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"  # PPO 小批量大小
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"  # 每 GPU 微批量
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"  # 每 GPU 最大 token 数
    actor_rollout_ref.actor.use_dynamic_bsz=True       # 动态批大小：按 token 总量而非固定样本数组批
    actor_rollout_ref.actor.use_kl_loss=True           # 启用 KL 散度损失约束
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"  # KL 损失系数
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}"  # KL 损失类型

    # ---- Rollout（vLLM 推理）配置 ----
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"   # 推理引擎名称
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}"   # 推理模式（async/sync）
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"         # 每个 prompt 生成的回复数
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"  # 推理张量并行度
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"  # GPU 显存利用率
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"  # 最大序列长度
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"  # 批处理最大 token 数
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"  # 最大并发序列数
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"  # 禁用 CUDA Graph
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"  # 推理后释放 KV Cache
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"  # log_prob 微批量
    actor_rollout_ref.rollout.load_format="${ROLLOUT_LOAD_FORMAT}"  # 模型加载格式
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}"  # 禁用自定义 AllReduce
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}"  # agent worker 数

    # ---- Ref 模型配置 ----
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"  # Ref log_prob 微批量

    # ---- 奖励模型配置 ----
    reward.num_workers="${REWARD_NUM_WORKERS}"          # 奖励计算 worker 数
    reward.reward_model.enable="${REWARD_MODEL_ENABLE}" # 是否启用 RM
    reward.reward_model.model_path="${RM_MODEL_PATH}"   # RM 模型路径
    reward.reward_model.enable_resource_pool="${RM_ENABLE_RESOURCE_POOL}"  # RM 独立资源池
    reward.reward_model.n_gpus_per_node="${RM_NUM_GPUS_PER_NODE}"  # RM 每节点 GPU 数
    reward.reward_model.nnodes="${RM_NNODES}"           # RM 节点数
    reward.reward_model.rollout.name="${ROLLOUT_NAME}"  # RM 推理引擎
    reward.reward_model.rollout.gpu_memory_utilization="${RM_GPU_MEMORY_UTILIZATION}"  # RM GPU 显存利用率
    reward.reward_model.rollout.tensor_model_parallel_size="${RM_TP}"  # RM 张量并行度
    reward.reward_model.rollout.max_model_len="${RM_MAX_MODEL_LEN}"  # RM 最大序列长度
    reward.reward_model.rollout.max_num_batched_tokens="${RM_MAX_NUM_BATCHED_TOKENS}"  # RM 批处理最大 token 数
    reward.reward_model.rollout.max_num_seqs="${RM_MAX_NUM_SEQS}"  # RM 最大并发序列数
    reward.reward_model.rollout.skip_tokenizer_init=False  # RM 是否跳过分词器初始化
    reward.reward_model.rollout.enforce_eager=True      # RM 禁用 CUDA Graph
    reward.reward_model.rollout.load_format="${RM_ROLLOUT_LOAD_FORMAT}"  # RM 模型加载格式
    +reward.reward_model.rollout.engine_kwargs.vllm.disable_custom_all_reduce="${RM_DISABLE_CUSTOM_ALL_REDUCE}"  # RM 禁用自定义 AllReduce

    # ---- 算法与训练器配置 ----
    algorithm.use_kl_in_reward=False                    # 是否将 KL 散度加入奖励信号（此处在 loss 中使用 KL，不在 reward 中重复）
    trainer.critic_warmup=0                             # Critic 预热步数（先训 Critic 再联合训练）；0 表示不预热
    trainer.logger="${TRAINER_LOGGER}"                   # 日志后端
    trainer.project_name="${TRAINER_PROJECT_NAME}"       # 项目名
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" # 实验名
    trainer.nnodes=1                                    # 训练节点数
    trainer.n_gpus_per_node="${TRAINER_NUM_GPUS}"        # 每节点 GPU 数
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"       # 训练前是否验证
    trainer.test_freq="${TEST_FREQ}"                     # 验证频率
    trainer.save_freq="${SAVE_FREQ}"                     # 保存频率
    trainer.resume_mode="${RESUME_MODE}"                 # 断点续训模式
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"  # 总训练步数
    trainer.use_legacy_worker_impl="${USE_LEGACY_WORKER_IMPL}"  # 是否使用旧版 worker
)

if [[ "${TRAIN_BACKEND}" == "megatron" ]]; then
    # ======= Megatron 后端专用配置 =======
    CMD+=(
        # ---- Actor LoRA 配置（Megatron 格式）----
        actor_rollout_ref.model.lora.rank="${LORA_RANK}"            # Actor LoRA 秩
        actor_rollout_ref.model.lora.alpha="${LORA_ALPHA}"          # Actor LoRA 缩放系数
        actor_rollout_ref.model.lora.target_modules="${LORA_TARGET_MODULES}"  # Actor LoRA 目标模块

        # ---- Actor Megatron 并行与卸载配置 ----
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${MEGATRON_TP}"   # Actor 张量并行度
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${MEGATRON_PP}" # Actor 流水线并行度
        actor_rollout_ref.actor.megatron.context_parallel_size="${MEGATRON_CP}"        # Actor 上下文并行度
        actor_rollout_ref.actor.megatron.expert_model_parallel_size="${MEGATRON_EP}"   # Actor 专家并行度（MoE）
        actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${MEGATRON_ETP}" # Actor 专家张量并行度
        actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size="${MEGATRON_VPP}"  # Actor 虚拟流水线并行
        actor_rollout_ref.actor.megatron.sequence_parallel="${MEGATRON_SEQUENCE_PARALLEL}"  # Actor 序列并行
        actor_rollout_ref.actor.megatron.use_mbridge="${MEGATRON_USE_MBRIDGE}"   # 使用 Megatron-Bridge
        actor_rollout_ref.actor.megatron.vanilla_mbridge="${MEGATRON_VANILLA_MBRIDGE}"  # 使用原始 Bridge
        actor_rollout_ref.actor.megatron.param_offload="${ACTOR_PARAM_OFFLOAD}"  # Actor 参数卸载到 CPU
        actor_rollout_ref.actor.megatron.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"  # Actor 优化器卸载
        actor_rollout_ref.actor.megatron.grad_offload="${ACTOR_GRAD_OFFLOAD}"    # Actor 梯度卸载
        actor_rollout_ref.actor.megatron.dtype="${MODEL_DTYPE}"     # Actor 训练精度

        # ---- Ref Megatron 并行与卸载配置 ----
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${MEGATRON_TP}"     # Ref 张量并行度
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${MEGATRON_PP}"   # Ref 流水线并行度
        actor_rollout_ref.ref.megatron.context_parallel_size="${MEGATRON_CP}"          # Ref 上下文并行度
        actor_rollout_ref.ref.megatron.expert_model_parallel_size="${MEGATRON_EP}"     # Ref 专家并行度
        actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${MEGATRON_ETP}"   # Ref 专家张量并行度
        actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size="${MEGATRON_VPP}"  # Ref 虚拟流水线并行
        actor_rollout_ref.ref.megatron.sequence_parallel="${MEGATRON_SEQUENCE_PARALLEL}"  # Ref 序列并行
        actor_rollout_ref.ref.megatron.use_mbridge="${MEGATRON_USE_MBRIDGE}"     # Ref 使用 Bridge
        actor_rollout_ref.ref.megatron.vanilla_mbridge="${MEGATRON_VANILLA_MBRIDGE}"  # Ref 原始 Bridge
        actor_rollout_ref.ref.megatron.param_offload="${REF_PARAM_OFFLOAD}"      # Ref 参数卸载到 CPU
        actor_rollout_ref.ref.megatron.dtype="${MODEL_DTYPE}"       # Ref 模型精度

        # ---- Critic 模型和训练配置（Megatron）----
        critic.model.path="${MODEL_PATH}"               # Critic 模型权重路径
        critic.optim.lr="${CRITIC_LR}"                  # Critic 学习率
        critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"  # Critic 每 GPU 微批量
        critic.ppo_max_token_len_per_gpu="${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU}"  # Critic 每 GPU 最大 token 数
        critic.model.lora.rank="${CRITIC_LORA_RANK}"    # Critic LoRA 秩
        critic.model.lora.alpha="${LORA_ALPHA}"         # Critic LoRA 缩放系数
        critic.model.lora.target_modules="${CRITIC_LORA_TARGET_MODULES}"  # Critic LoRA 目标模块

        # ---- Critic Megatron 并行与卸载配置 ----
        critic.megatron.tensor_model_parallel_size="${MEGATRON_TP}"   # Critic 张量并行度
        critic.megatron.pipeline_model_parallel_size="${MEGATRON_PP}" # Critic 流水线并行度
        critic.megatron.context_parallel_size="${MEGATRON_CP}"        # Critic 上下文并行度
        critic.megatron.expert_model_parallel_size="${MEGATRON_EP}"   # Critic 专家并行度
        critic.megatron.expert_tensor_parallel_size="${MEGATRON_ETP}" # Critic 专家张量并行度
        critic.megatron.virtual_pipeline_model_parallel_size="${MEGATRON_VPP}"  # Critic 虚拟流水线并行
        critic.megatron.sequence_parallel="${MEGATRON_SEQUENCE_PARALLEL}"  # Critic 序列并行
        critic.megatron.use_mbridge="${MEGATRON_USE_MBRIDGE}"         # Critic 使用 Bridge
        critic.megatron.vanilla_mbridge="${MEGATRON_VANILLA_MBRIDGE}" # Critic 原始 Bridge
        critic.megatron.param_offload="${CRITIC_PARAM_OFFLOAD}"       # Critic 参数卸载
        critic.megatron.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD}"  # Critic 优化器卸载
        critic.megatron.grad_offload="${CRITIC_GRAD_OFFLOAD}"         # Critic 梯度卸载
        critic.megatron.dtype="${MODEL_DTYPE}"            # Critic 训练精度
    )
else
    # ======= FSDP 后端专用配置 =======
    CMD+=(
        actor_rollout_ref.rollout.layered_summon=True   # 分层召唤：按需加载 Rollout 引擎以优化显存

        # ---- Actor LoRA 配置（FSDP 格式）----
        actor_rollout_ref.model.lora_rank="${LORA_RANK}"   # Actor LoRA 秩
        actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"  # Actor LoRA 缩放系数
        actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"  # Actor LoRA 目标模块

        # ---- Actor FSDP 训练配置 ----
        actor_rollout_ref.actor.strategy="${TRAIN_STRATEGY}"  # 分布式策略（fsdp/fsdp2）
        actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}"  # torch.compile 加速
        actor_rollout_ref.actor.fsdp_config.strategy="${TRAIN_STRATEGY}"  # FSDP 分片策略
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1   # FSDP 分片组大小；-1 表示全局分片（跨所有 GPU）
        actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"  # Actor 参数卸载到 CPU
        actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"  # Actor 优化器卸载
        actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}"  # Actor 模型精度
        actor_rollout_ref.actor.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}"  # FSDP 层的 compile

        # ---- Ref FSDP 配置 ----
        actor_rollout_ref.ref.strategy="${TRAIN_STRATEGY}"    # Ref 分布式策略
        actor_rollout_ref.ref.use_torch_compile="${USE_TORCH_COMPILE}"  # Ref torch.compile
        actor_rollout_ref.ref.fsdp_config.strategy="${TRAIN_STRATEGY}"  # Ref FSDP 策略
        actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"  # Ref 参数卸载到 CPU
        actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}"  # Ref 模型精度
        actor_rollout_ref.ref.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}"  # Ref compile

        # ---- Critic FSDP 训练配置 ----
        critic.strategy="${TRAIN_STRATEGY}"                   # Critic 分布式策略
        critic.model.path="${MODEL_PATH}"                     # Critic 模型路径
        critic.model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}"  # Critic 梯度检查点
        critic.model.use_remove_padding=True                  # Critic 移除 padding 优化
        critic.model.lora_rank="${CRITIC_LORA_RANK}"          # Critic LoRA 秩
        critic.model.lora_alpha="${LORA_ALPHA}"               # Critic LoRA 缩放系数
        critic.model.target_modules="${CRITIC_LORA_TARGET_MODULES}"  # Critic LoRA 目标模块
        critic.optim.lr="${CRITIC_LR}"                        # Critic 学习率
        critic.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"  # Critic 每 GPU 微批量
        critic.ppo_max_token_len_per_gpu="${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU}"  # Critic 每 GPU 最大 token 数
        critic.model.fsdp_config.strategy="${TRAIN_STRATEGY}"  # Critic FSDP 策略
        critic.model.fsdp_config.fsdp_size=-1                 # Critic FSDP 分片组大小
        critic.model.fsdp_config.param_offload="${CRITIC_PARAM_OFFLOAD}"  # Critic 参数卸载
        critic.model.fsdp_config.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD}"  # Critic 优化器卸载
        critic.model.fsdp_config.model_dtype="${MODEL_DTYPE}"  # Critic 模型精度
        critic.model.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}"  # Critic compile
    )
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

cd "${ROOT_DIR}"
mkdir -p "${RUN_OUTPUT_DIR}"
echo "Training outputs will be written to: ${RUN_OUTPUT_DIR}"
echo "Training stdout/stderr will be tee'd to: ${RUN_STDOUT_LOG}"
"${CMD[@]}" "$@" 2>&1 | tee -a "${RUN_STDOUT_LOG}"
exit "${PIPESTATUS[0]}"
