import os
from trl import SFTConfig
import wandb
import omegaconf


def initialize_wandb(config, logger=None):
    outdir = config.save.output_dir

    args = config.wandb
    assert '/outputs/' in outdir, f"outdir should be like [project_dir]/outputs/..."
    run_name = outdir.split('/outputs/')[-1]

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project_name,
        name=run_name,
        config=args,            # 학습에 사용된 모든 하이퍼파라미터를 WandB 대시보드에 저장
        save_code=True,         # 이 학습을 실행한 메인 파이썬 스크립트 파일(.py)을 WandB 서버에 업로드하여 보관
        # dir=outdir,             # WandB가 실행되는 동안 생성하는 임시 로그 및 메타데이터 파일들이 저장될 로컬 디렉토리
    )

    if logger:
        logger.info("[PROGRESS] wandb initialized.")


def load_deepspeed_config(config):
    # SLURM_GPUS_ON_NODE = int(os.getenv("SLURM_GPUS_ON_NODE", "-1"))
    # assert SLURM_GPUS_ON_NODE > 0, f"env SLURM_GPUS_ON_NODE is not set or invalid."
    
    args = config.deepspeed
    
    ds_config = {
        # "fp16": {
        #     "enabled": True if args.dtype=="fp16" else False,
        #     "loss_scale": 0,  # 동적 스케일링 활성화
        #     "initial_scale_power": 16,  # 초기 스케일 2^16
        #     "loss_scale_window": 1000,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1
        # },
        # "bf16":{
        #     "enabled": True if args.dtype=="bf16" else False,
        # },

        "bf16":{
            "enabled": True,
        },

        "zero_optimization":{
            "stage":args.deepspeed_stage,
            # "zero_hpz": SLURM_GPUS_ON_NODE,
            # "overlap_comm": True,
            # "contiguous_gradients": True,
            # "reduce_bucket_size": "auto",
        },

        # "train_batch_size": args.global_batch,
        # "train_micro_batch_size_per_gpu": args.micro_batch,
        # "gradient_accumulation_steps": grad_accum,
        # "gradient_clipping": args.gradient_clipping,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,               # default: 10
        
        # "optimizer": {
        #     "type": "AdamW", 
        #         "params": {
        #             "lr": args.learning_rate,
        #             "betas": [0.9, 0.999],  # 0.9, 0.999
        #             "eps":1e-8, # 1e-8
        #             "weight_decay": 0.01,
        #             # "max_grad_norm ":1.0, # added
        #     }
        # },

        # "scheduler": {
        #     "type": "WarmupCosineLR",
        #     "params": {
        #         # "total_num_steps": ,      # packing 하므로 total_num_steps 가 애매함
        #         # "warmup_num_steps": ,
        #         # "warmup_min_ratio": ,
        #         "warmup_type":"linear", # log -> linear
        #     }
        # },

        # "activation_checkpointing": {
        #     "partition_activations": True if args.gradient_checkpointing else False,
        #     # "cpu_checkpointing": True if args.gradient_checkpointing else False,
        #     "cpu_checkpointing": False,
        #     "contiguous_memory_optimization": True if args.gradient_checkpointing else False,
        #     "number_checkpoints": None,
        #     "synchronize_checkpoint_boundary": False,
        #     "profile": False
        # }
    }
    return ds_config


def load_sft_config(config, logger=None):
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    if WORLD_SIZE < 0:
        print("[WARNING] env WORLD_SIZE is not set or invalid.", flush=True)
        WORLD_SIZE = 1

    grad_accum = config.deepspeed.global_batch // config.deepspeed.micro_batch // WORLD_SIZE
    assert config.deepspeed.global_batch == grad_accum * config.deepspeed.micro_batch * WORLD_SIZE, \
        f"grad_accum: {grad_accum}, global_batch: {config.deepspeed.global_batch}, micro_batch: {config.deepspeed.micro_batch}, world_size: {WORLD_SIZE}"
    
    # (save to args)
    with omegaconf.open_dict(config):
        config.deepspeed.gradient_accumulation_steps = grad_accum

    deepspeed_config = load_deepspeed_config(config)

    sft_config = SFTConfig(
        # seed
        seed=config.seed,
        data_seed=config.data_seed,

        # model, tokenizer
        chat_template_path=config.model.get("chat_template_path", None),

        # dataset
        dataset_num_proc=config.data.dataset_num_proc,
        max_length=config.data.max_length,
        packing=True,
        shuffle_dataset=False,      # data load 할 때 이미 shuffle 함.

        # train
        per_device_train_batch_size=config.deepspeed.micro_batch,
        gradient_accumulation_steps=config.deepspeed.gradient_accumulation_steps,
        max_grad_norm=config.deepspeed.gradient_clipping,

        bf16=True,
        assistant_only_loss=config.trainer.get("assistant_only_loss", False),
        num_train_epochs=config.trainer.num_train_epochs,

        # optimizer
        optim=config.optimizer.optim,
        weight_decay=config.optimizer.weight_decay,
        adam_beta1=config.optimizer.adam_beta1,
        adam_beta2=config.optimizer.adam_beta2,
        adam_epsilon=config.optimizer.adam_epsilon,

        # scheduler
        learning_rate=config.scheduler.learning_rate,
        lr_scheduler_type=config.scheduler.lr_scheduler_type,
        warmup_ratio=config.scheduler.warmup_ratio,

        # train: deepspeed
        deepspeed=deepspeed_config,

        # output
        output_dir=config.save.output_dir,
        save_strategy=config.save.save_strategy,
        save_total_limit=config.save.save_total_limit,

        # monitoring
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=10,
        log_on_each_node=False,
    )

    if config.save.save_strategy == "steps":
        sft_config.save_steps = config.save.save_steps

    if logger:
        logger.info("[PROGRESS] SFTConfig loaded.")
        logger.info(f"=================SFTConfig:\n{sft_config}\n================\n")
    return sft_config


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="configs", version_base=None)
    def test(config):
        initialize_wandb(config)
        sft_config = load_sft_config(config)
        print(sft_config)
    
    test()
