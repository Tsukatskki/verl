#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

################################################################################
# ====================== Megatron + GRPO 训练脚本 ======================
# 本脚本是 run_local_qwen3_fp8_rm_ppo.sh 的上层封装，通过设置环境变量来定制
# Megatron 后端 + GRPO 算法 + vLLM 推理 + Skywork RM 的训练配置。
# 仅需修改以下值即可进行日常训练。
################################################################################

################################################################################
# ====================== 设备与模型路径 ======================
################################################################################

# 指定使用哪些 GPU（作用于所有阶段：训练/推理/RM）；通过 CUDA_VISIBLE_DEVICES 限制可见设备
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Actor/Critic/Ref 模型路径（HF 格式）；这里使用 Qwen3-8B 全精度模型
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
# 奖励模型路径（HF 格式）；Skywork-Reward 是基于 Llama-3.1-8B 的奖励模型，
# 作用于奖励评估阶段对 Rollout 回复进行打分
export RM_MODEL_PATH="${RM_MODEL_PATH:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"

################################################################################
# ====================== 数据配置 ======================
# 控制训练和验证数据的加载，作用于数据准备阶段。
################################################################################

# 训练数据文件（parquet 格式）；包含 prompt 样本，用于 Rollout 生成和策略优化
export TRAIN_FILES="${TRAIN_FILES:-data/ultrafeedback/train_80.parquet}"
# 验证数据文件；用于训练过程中的效果评估（从原始训练集中拆出 20%）
export VAL_FILES="${VAL_FILES:-data/ultrafeedback/val_20.parquet}"

################################################################################
# ====================== 核心训练超参数 ======================
# 控制训练规模和学习率等关键参数，作用于 Actor 训练阶段。
################################################################################

# 总训练步数；每步包含完整的 Rollout → RM 打分 → 优势估计 → Actor 更新 流程
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
# 每步训练的总样本数（batch size）；4 卡下先保守扩大到 8，提升吞吐同时避免一下子过激
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
# prompt 最大 token 长度；超长 prompt 会被过滤，通过限制输入长度控制显存
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-64}"
# 模型生成回复的最大 token 长度；限制 Rollout 阶段的生成长度，直接影响推理时间和显存
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"

# Actor 学习率；控制策略梯度更新的步长，过大导致训练不稳定，过小收敛慢
# 此处设为 1e-5，比默认多模型 PPO 的 1e-6 更激进，适合 GRPO 场景
export ACTOR_LR="${ACTOR_LR:-1e-5}"
# Actor 学习率预热步数比例；0.1 表示前 10% step 线性 warmup
export ACTOR_LR_WARMUP_STEPS_RATIO="${ACTOR_LR_WARMUP_STEPS_RATIO:-0.1}"
# Actor 学习率衰减策略；Megatron 支持 constant/linear/cosine/inverse_square_root
export ACTOR_LR_DECAY_STYLE="${ACTOR_LR_DECAY_STYLE:-cosine}"
# Actor 最小学习率；cosine decay 末端收敛到该值
export ACTOR_MIN_LR="${ACTOR_MIN_LR:-1e-6}"
# Actor 学习率衰减总步数；默认与总训练步数一致
export ACTOR_LR_DECAY_STEPS="${ACTOR_LR_DECAY_STEPS:-${TOTAL_TRAINING_STEPS}}"
# KL 散度损失系数；控制策略模型与参考模型的偏离惩罚强度
# 值越大约束越强，策略变化越保守；此处 0.0005 较小，允许更大策略更新
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.0005}"
# KL 损失类型；low_var_kl 是低方差 KL 估计器，比标准 KL 散度的训练更稳定
export KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
# LoRA 秩（rank）；通过低秩分解只训练少量参数（rank=16 约占全参数的 0.5%），
# 大幅降低显存和计算量。作用于 Actor/Critic 训练阶段
export LORA_RANK="${LORA_RANK:-16}"
# LoRA 缩放系数（alpha）；实际缩放因子 = alpha/rank = 16/16 = 1.0
# 与 rank 相等时 LoRA 分支权重与基础模型等权重混合
export LORA_ALPHA="${LORA_ALPHA:-16}"
# Megatron-Bridge 使用的是 Megatron 模块名，不支持 HF/PEFT 风格的 all-linear
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-['linear_qkv','linear_proj','linear_fc1','linear_fc2']}"
export CRITIC_LORA_TARGET_MODULES="${CRITIC_LORA_TARGET_MODULES:-${LORA_TARGET_MODULES}}"

################################################################################
# ====================== Rollout 与 RM 关键参数 ======================
# 控制推理阶段和奖励模型评估阶段的显存分配。
################################################################################

# 每个 prompt 生成的回复数量；GRPO 算法需要 n>1（此处 n=2）以在组内做相对排名计算优势
# 越多越能准确估计优势但增加生成和奖励计算量
export ROLLOUT_N="${ROLLOUT_N:-2}"
# vLLM 推理引擎的 GPU 显存利用率上限（0~1）；
# 4 卡方案 A 中训练主链路优先，先进一步保守到 0.18，给训练和通信留余量
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.18}"
# 奖励模型推理的 GPU 显存利用率上限；先下调到 0.35，降低与训练争抢显存的风险
export RM_GPU_MEMORY_UTILIZATION="${RM_GPU_MEMORY_UTILIZATION:-0.35}"

################################################################################
# ====================== 日志与实验跟踪 ======================
# 控制训练日志记录平台，作用于训练主循环。
################################################################################

# W&B 项目名称
export WANDB_PROJECT="${WANDB_PROJECT:-verl_local}"
# W&B 用户/团队名称
export WANDB_ENTITY="${WANDB_ENTITY:-yshdouble}"
# 实验名称；在 W&B 中区分不同的训练运行
export TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-2026_03_30_qwen3_8b_grpo}"

################################################################################
# ====================== 训练后端与算法选择 ======================
# 这些参数决定了训练框架和强化学习算法的核心架构。
################################################################################

# 训练后端；megatron 使用 NVIDIA Megatron-LM 框架进行分布式训练
# 相比 FSDP 更适合大规模多 GPU/多节点训练
export TRAIN_BACKEND=megatron
# 优势估计器；grpo（Group Relative Policy Optimization）不需要 Critic 模型，
# 通过同一 prompt 的多条回复（由 ROLLOUT_N 控制）在组内相对排名来计算优势
export ADV_ESTIMATOR=grpo
# 日志后端；同时输出到终端和 W&B 平台
export TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"

################################################################################
# ====================== GPU 与 Megatron 并行配置 ======================
# 控制分布式训练的并行拓扑，作用于 Actor/Critic/Ref 的训练阶段。
################################################################################

# 总 GPU 数量
export NUM_GPUS=4
# 训练器实际使用的 GPU 数量
export TRAINER_NUM_GPUS=4
# Megatron 张量并行度（TP=4）；训练主链路扩到 4 卡
export MEGATRON_TP=4
# Megatron 流水线并行度（PP=1）；不使用流水线并行（模型不按层切分到不同 GPU）
export MEGATRON_PP=1
# Megatron 上下文并行度（CP=1）；不使用上下文并行（不沿序列维度切分）
export MEGATRON_CP=1
# 是否使用 Megatron-Bridge；必须启用以支持 HF 模型格式和 LoRA 训练
export MEGATRON_USE_MBRIDGE=True
# 是否使用原始 Bridge；False 使用增强版支持 LoRA 等功能
export MEGATRON_VANILLA_MBRIDGE=False

################################################################################
# ====================== PPO/GRPO 训练循环超参数 ======================
# 控制策略优化循环的批大小和内存限制，作用于 Actor 训练阶段。
# 注意：虽然使用 GRPO 算法，但底层复用 PPO 的训练循环实现。
################################################################################

# 每步 Actor 训练的轮数（epoch）；1 表示每步只过一遍训练数据
export PPO_EPOCHS=1
# PPO 小批量大小；4 卡下适度增大，减少更新过碎的问题
export PPO_MINI_BATCH_SIZE=4
# 每个 GPU 的微批量大小；1 表示每次前向/反向传播只处理 1 个样本（最小显存占用）
export PPO_MICRO_BATCH_SIZE_PER_GPU=1
# Actor 每 GPU 每个微批次的最大 token 总长度；限制显存峰值
# 192 = MAX_PROMPT_LENGTH(64) + MAX_RESPONSE_LENGTH(128)
export PPO_MAX_TOKEN_LEN_PER_GPU=192
# Critic 每 GPU 每个微批次的最大 token 总长度（GRPO 中 Critic 可能不使用但保留配置）
export CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU=192

################################################################################
# ====================== 显存优化配置 ======================
# 通过梯度检查点和 CPU 卸载来降低 GPU 显存占用，
# 以牺牲计算速度换取在有限显存上训练大模型的能力。
################################################################################

# 梯度检查点；通过重新计算反向传播中间激活（而非保存），降低约 60-70% 的激活显存
export ENABLE_GRADIENT_CHECKPOINTING=True
# Actor 模型参数卸载到 CPU；非活跃层的参数存放在 CPU 内存中
export ACTOR_PARAM_OFFLOAD=True
# Actor 优化器状态（Adam 的 m/v）卸载到 CPU；这些状态通常占参数量 2 倍显存
export ACTOR_OPTIMIZER_OFFLOAD=True
# 参考模型（Ref）参数卸载到 CPU；Ref 不训练只计算 KL 散度，可安全卸载
export REF_PARAM_OFFLOAD=True
# Critic 模型参数卸载到 CPU
export CRITIC_PARAM_OFFLOAD=True
# Critic 优化器状态卸载到 CPU
export CRITIC_OPTIMIZER_OFFLOAD=True

################################################################################
# ====================== Rollout（vLLM 推理引擎）配置 ======================
# 控制 Actor 模型在 Rollout 阶段的推理行为：用当前策略为每个 prompt 生成回复。
################################################################################

# Rollout 张量并行度（TP=2）；方案 A 先保守维持 2，降低 rollout 与训练的资源竞争
export ROLLOUT_TP=2
# 推理引擎选择；vllm 是高性能 LLM 推理引擎
export ROLLOUT_NAME=vllm
# 推理模式；async 表示推理与训练异步执行以提高 GPU 利用率
export ROLLOUT_MODE=async
# vLLM 支持的最大序列长度（prompt+response）；必须 >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
export ROLLOUT_MAX_MODEL_LEN=192
# vLLM 单次调度批中最大 token 数；控制推理时内存峰值
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=192
# vLLM 并发最大序列数；适度上调到 8，但不一上来过激
export ROLLOUT_MAX_NUM_SEQS=8
# 禁用 CUDA Graph；序列长度变化大时 CUDA Graph 不适用
export ROLLOUT_ENFORCE_EAGER=True
# 推理完成后释放 KV Cache；释放显存给后续训练阶段使用
export ROLLOUT_FREE_CACHE_ENGINE=True
# 禁用自定义 AllReduce；使用标准 NCCL 通信以确保稳定性
export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE=True

################################################################################
# ====================== 奖励模型（Reward Model）配置 ======================
# 控制外部奖励模型的推理行为，对 Rollout 生成的回复打分，提供训练信号。
################################################################################

# 启用外部奖励模型
export REWARD_MODEL_ENABLE=True
# RM 推理 worker 数量；1 个 worker 即可处理小 batch
export REWARD_NUM_WORKERS=1
# RM 是否使用独立资源池；False 表示与其他组件共享 GPU
export RM_ENABLE_RESOURCE_POOL=False
# RM 每节点 GPU 数量
export RM_NUM_GPUS_PER_NODE=1
# RM 节点数
export RM_NNODES=1
# RM 推理张量并行度；方案 A 先保守维持 2
export RM_TP=2
# RM 支持的最大序列长度；保留 512，避免再次触发长度超限
export RM_MAX_MODEL_LEN=512
# RM 单次批处理最大 token 数
export RM_MAX_NUM_BATCHED_TOKENS=256
# RM 并发最大序列数
export RM_MAX_NUM_SEQS=1
# RM 禁用自定义 AllReduce
export RM_DISABLE_CUSTOM_ALL_REDUCE=True

################################################################################
# ====================== 训练器控制参数 ======================
# 控制整体训练流程的 checkpoint 保存、验证和断点续训。
################################################################################

# checkpoint 保存频率（步为单位）；-1 表示不保存
export SAVE_FREQ="${SAVE_FREQ:--1}"
# 验证频率（步为单位）；每 10 步执行一次验证
export TEST_FREQ="${TEST_FREQ:-10}"
# 训练前是否先验证一次（获取基线指标）
export VAL_BEFORE_TRAIN=True
# 断点续训模式；disable 表示从头开始训练
export RESUME_MODE="${RESUME_MODE:-disable}"

# 调用底层通用训练脚本，所有上述环境变量将作为超参数传入
exec bash scripts/run_local_qwen3_fp8_rm_ppo.sh
