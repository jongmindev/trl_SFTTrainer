import os
import logging
from transformers.utils import logging as hf_logging
from trl import SFTTrainer
import hydra

from src.data import load_dataset_sft
from src.model import load_model_and_tokenizer
from src.setup import initialize_wandb, load_sft_config


@hydra.main(config_path="src/configs", version_base=None)
def main(config):
    global_rank = int(os.getenv("RANK", "0"))

    if global_rank == 0:
        logger = logging.getLogger(__name__)
        hf_logging.set_verbosity_info()
    else:
        logger = None
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

    train_dataset = load_dataset_sft(config, logger=logger)
    model, tokenizer = load_model_and_tokenizer(config, logger=logger)

    if global_rank == 0:
        initialize_wandb(config, logger=logger)

    sft_config = load_sft_config(config, logger=logger)
    if global_rank != 0:
        sft_config.disable_tqdm = True

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    output_dir = config.save.output_dir
    resume_from_checkpoint = None
    if os.path.exists(output_dir) and any("checkpoint" in s for s in os.listdir(output_dir)):
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    main()
