#!/usr/bin/env python
import argparse
import os

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


def build_config(args):
    config_dir = os.path.join(os.path.dirname(__file__), "..", "verl", "trainer", "config")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    config.actor_rollout_ref.model.path = args.model_path
    config.reward.num_workers = 1
    config.reward.reward_model.enable = True
    config.reward.reward_model.enable_resource_pool = False
    config.reward.reward_model.model_path = args.rm_model_path
    config.reward.reward_model.n_gpus_per_node = 1
    config.reward.reward_model.nnodes = 1
    config.reward.reward_model.rollout.name = "vllm"
    config.reward.reward_model.rollout.tensor_model_parallel_size = 1
    config.reward.reward_model.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.reward.reward_model.rollout.max_model_len = args.max_model_len
    config.reward.reward_model.rollout.max_num_batched_tokens = args.max_num_batched_tokens
    config.reward.reward_model.rollout.max_num_seqs = 1
    config.reward.reward_model.rollout.enforce_eager = True
    config.reward.reward_model.rollout.skip_tokenizer_init = False
    config.reward.reward_model.rollout.load_format = args.load_format
    config.trainer.nnodes = 1
    config.trainer.n_gpus_per_node = 1
    return config


def build_dataproto(tokenizer, question: str, answer: str) -> DataProto:
    raw_prompt = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_ids = tokenizer.encode(answer, add_special_tokens=False)

    tensors = {
        "prompts": torch.tensor([prompt_ids], dtype=torch.long),
        "responses": torch.tensor([response_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(prompt_ids) + len(response_ids)), dtype=torch.long),
    }
    non_tensors = {
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "data_source": np.array(["debug_reward"], dtype=object),
        "reward_model": np.array([{"ground_truth": None}], dtype=object),
        "extra_info": np.array([{"question": question}], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def extract_score(scored: DataProto) -> float:
    scores = scored.batch["rm_scores"][0]
    nonzero = torch.nonzero(scores, as_tuple=False)
    if nonzero.numel() == 0:
        return 0.0
    return float(scores[nonzero[-1, 0]].item())


def main():
    parser = argparse.ArgumentParser(description="Debug Skywork reward scoring through verl reward loop.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"))
    parser.add_argument(
        "--rm-model-path",
        default=os.environ.get("RM_MODEL_PATH", "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"),
    )
    parser.add_argument("--question", default="Please summarize why regular exercise is healthy.")
    parser.add_argument(
        "--answer",
        default="Regular exercise improves cardiovascular health, supports mood, and helps maintain strength.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--load-format", default=os.environ.get("RM_ROLLOUT_LOAD_FORMAT", "auto"))
    args = parser.parse_args()

    print("Reward debug config:")
    print(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "model_path": args.model_path,
                    "rm_model_path": args.rm_model_path,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "max_model_len": args.max_model_len,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                }
            )
        )
    )

    ray.init(ignore_reinit_error=True)
    try:
        config = build_config(args)
        tokenizer = hf_tokenizer(copy_to_local(args.model_path), trust_remote_code=True)
        sample = build_dataproto(tokenizer, args.question, args.answer)
        manager = RewardLoopManager(config)
        print(f"Reward router: {manager.reward_router_address}")
        scored = manager.compute_rm_score(sample)
        score = extract_score(scored)
        print("Reward scoring succeeded.")
        print(f"Score: {score:.6f}")
        print(f"RM tensor: {scored.batch['rm_scores']}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
