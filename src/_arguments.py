import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--tokenizer", type=str)

    parser.add_argument("--datasets", nargs="+", type=str, required=True, help="delimited by space")

    parser.add_argument("--global_batch", required=True, type=int)      # 288
    parser.add_argument("--micro_batch", required=True, type=int)       # 1

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dtype", default="bf16", type=str)
    parser.add_argument("--num_cpus", default=32, type=int)
    parser.add_argument("--deepspeed_stage", default=1, type=int)
    parser.add_argument("--activation_recomputation", action="store_true")

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=float, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)

    parser.add_argument("--fix_mistral_regex", action="store_true")


    args = parser.parse_args()
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0")) 

    # Print args
    args_dict = vars(args)
    print()
    print("[==================== Arguments ====================]")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("[===================================================]\n")

    return args