#!/usr/bin/env python
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from transformers import AutoConfig, AutoModelForCausalLM

from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [rank={rank} local_rank={local_rank}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--disable-sync-module-states", action="store_true")
    parser.add_argument("--disable-meta-init", action="store_true")
    parser.add_argument("--wrap-disable", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2

    dtype = getattr(torch, args.dtype)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    log(f"torch={torch.__version__} cuda={torch.version.cuda}")
    log(f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')} count={torch.cuda.device_count()} current={torch.cuda.current_device()}")
    log("init_process_group start")
    dist.init_process_group(backend="nccl")
    log("init_process_group done")

    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", mesh_shape=(world_size,))
    log(f"device_mesh={mesh}")

    log("AutoConfig.from_pretrained start")
    config = AutoConfig.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    log(
        f"AutoConfig done model_type={getattr(config, 'model_type', None)} "
        f"tie_word_embeddings={getattr(config, 'tie_word_embeddings', None)} "
        f"architectures={getattr(config, 'architectures', None)}"
    )

    use_meta = not args.disable_meta_init and not getattr(config, "tie_word_embeddings", False)
    init_context = get_init_weight_context_manager(use_meta_tensor=use_meta, mesh=mesh)
    log(f"init_context use_meta={use_meta}")

    with init_context():
        log("AutoModelForCausalLM.from_pretrained start")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            config=config,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
        )
        log("AutoModelForCausalLM.from_pretrained done")
        model.to(dtype)
        log("model.to(dtype) done")
        if not args.disable_lora:
            log("LoRA apply start")
            model.enable_input_require_grads()
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=args.target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            log("LoRA apply done")

    dist.barrier(device_ids=[get_device_id()])
    log("barrier after model init done")

    mp = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    auto_wrap_policy = None if args.wrap_disable else get_fsdp_wrap_policy(module=model, config=None, is_lora=not args.disable_lora)
    log(f"auto_wrap_policy={auto_wrap_policy}")

    start = time.time()
    log(f"FSDP start sync_module_states={not args.disable_sync_module_states}")
    model = FSDP(
        model,
        param_init_fn=init_fn,
        auto_wrap_policy=auto_wrap_policy,
        device_id=get_device_id(),
        mixed_precision=mp,
        sync_module_states=not args.disable_sync_module_states,
        device_mesh=mesh,
        use_orig_params=False,
        forward_prefetch=False,
    )
    dist.barrier(device_ids=[get_device_id()])
    log(f"FSDP done elapsed={time.time() - start:.2f}s")

    del model
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
