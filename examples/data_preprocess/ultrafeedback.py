#!/usr/bin/env python3
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
Preprocess the openbmb/UltraFeedback dataset into verl RL parquet format.
"""

import argparse
import os
import shutil
from typing import Any

import datasets


def pick_best_completion(example: dict[str, Any]) -> dict[str, Any]:
    completions = example.get("completions") or []
    if not completions:
        return {"response": "", "model": None, "overall_score": None, "fine_grained_score": None}

    def score_key(item: dict[str, Any]) -> tuple[float, float]:
        overall = item.get("overall_score")
        fine_grained = item.get("fine-grained_score", item.get("fine_grained_score"))
        overall = float(overall) if overall is not None else float("-inf")
        fine_grained = float(fine_grained) if fine_grained is not None else float("-inf")
        return overall, fine_grained

    best = max(completions, key=score_key)
    return {
        "response": best.get("response", ""),
        "model": best.get("model"),
        "overall_score": best.get("overall_score"),
        "fine_grained_score": best.get("fine-grained_score", best.get("fine_grained_score")),
    }


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        instruction = example["instruction"]
        best = pick_best_completion(example)
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": instruction}],
            "ability": "alignment",
            "reward_model": {
                "style": "model",
                "ground_truth": best["response"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": instruction,
                "reference_answer": best["response"],
                "reference_model": best["model"],
                "reference_overall_score": best["overall_score"],
                "reference_fine_grained_score": best["fine_grained_score"],
                "source": example.get("source"),
                "correct_answers": example.get("correct_answers"),
                "incorrect_answers": example.get("incorrect_answers"),
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated. Use --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/ultrafeedback")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--dataset_name", default="openbmb/UltraFeedback")
    parser.add_argument("--local_dataset_path", default=None, help="Use a local dataset path if already mirrored.")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    local_save_dir = args.local_dir or args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    dataset_name = args.local_dataset_path or args.dataset_name
    dataset = datasets.load_dataset(dataset_name)
    train_dataset = dataset["train"]

    split_dataset = train_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
    train_split = split_dataset["train"].map(function=make_map_fn("train", args.dataset_name), with_indices=True)
    val_split = split_dataset["test"].map(function=make_map_fn("test", args.dataset_name), with_indices=True)

    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "test.parquet")
    train_split.to_parquet(train_path)
    val_split.to_parquet(val_path)

    if args.hdfs_dir is not None:
        os.makedirs(args.hdfs_dir, exist_ok=True)
        dst_dir = os.path.join(args.hdfs_dir, os.path.basename(local_save_dir.rstrip("/")))
        shutil.copytree(local_save_dir, dst_dir, dirs_exist_ok=True)
