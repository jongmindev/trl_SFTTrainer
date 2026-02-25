import os
import logging
from transformers.utils import logging as hf_logging
from trl import SFTTrainer
# from trl.trainer import DataCollatorForCompletionOnlyLM   # deprecated
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

    config.data.datasets = config.data.datasets[:1]

    train_dataset = load_dataset_sft(config, logger=logger)
    model, tokenizer = load_model_and_tokenizer(config, logger=logger)
    # collator = DataCollatorForCompletionOnlyLM(           # deprecated
    #     tokenizer=tokenizer,
    #     response_template=config.model.response_template,
    #     instruction_template=config.model.instruction_template,
    # )

    # if global_rank == 0:
    #     initialize_wandb(config, logger=logger)

    sft_config = load_sft_config(config, logger=logger)
    if global_rank != 0:
        sft_config.disable_tqdm = True
    sft_config.report_to = None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        # data_collator=collator,                           # deprecated
        train_dataset=train_dataset,
        args=sft_config,
    )

    output_dir = config.save.output_dir
    resume_from_checkpoint = None
    if os.path.exists(output_dir) and any("checkpoint" in s for s in os.listdir(output_dir)):
        resume_from_checkpoint = True

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    import torch
    model.to("cuda:0")
    model.eval()
    for i, batch in enumerate(trainer.get_train_dataloader()):
        
        # # profiler 출력
        # with torch.no_grad():
        #     with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=False,
        #         with_stack=False,
        #     ) as prof:
        #         _ = model(**batch)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        
        # print("\n\n\n" + "=" * 80 + "\n\n\n")

        # with torch.no_grad():
        #     # ---- A+B (원본 packed) ----
        #     out_ab = model(
        #         input_ids=batch["input_ids"],
        #         position_ids=batch["position_ids"],
        #         use_cache=False,
        #     )
        #     logits_ab = out_ab.logits  # (B, L, V)

        #     # 2) segment boundary 찾기: position_ids == 0 인 지점들이 segment start
        #     #    (각 배치 아이템별로 따로)
        #     pos = batch["position_ids"]  # (B, L)
        #     B, L = pos.shape

        #     # 우리는 "두 번째 segment의 시작"을 B 시작점으로 잡는다.
        #     # (첫 번째는 비교 불가)
        #     # starts[b] = [idx0, idx1, idx2, ...]
        #     leaked_any = False

        #     for b in range(B):
        #         starts = (pos[b] == 0).nonzero(as_tuple=False).flatten().tolist()

        #         # 보통 starts[0]==0 (맨 앞 segment 시작)
        #         # packing이 실제로 됐으면 starts 길이가 2 이상일 가능성이 큼
        #         if len(starts) < 2:
        #             # 이 샘플은 segment가 1개뿐이라 누출 테스트 불가
        #             continue

        #         start_B = starts[1]  # 두 번째 segment 시작 위치 (AB에서의 인덱스)

        #         # 3) AB에서의 "B 첫 토큰" logits
        #         #    (b번째 배치, start_B 위치의 logits)
        #         v_ab = logits_ab[b, start_B, :].float().detach().cpu()

        #         # ---- B_only (두 번째 segment만 잘라내기) ----
        #         input_b = batch["input_ids"][b, start_B:].unsqueeze(0)         # (1, L2)
        #         pos_b   = batch["position_ids"][b, start_B:].unsqueeze(0)      # (1, L2)
        #         # 여기서 pos_b의 첫 값은 보통 0이어야 함 (segment reset)

        #         out_b = model(
        #             input_ids=input_b,
        #             position_ids=pos_b,
        #             use_cache=False,
        #         )
        #         logits_b = out_b.logits  # (1, L2, V)
        #         v_b = logits_b[0, 0, :].float().detach().cpu()

        #         # 4) 비교: max abs diff / cosine similarity 등
        #         max_abs = (v_ab - v_b).abs().max().item()
        #         cos = torch.nn.functional.cosine_similarity(v_ab, v_b, dim=0).item()

        #         print(f"[leak-test] batch_item={b} start_B={start_B}  max_abs={max_abs:.6g}  cosine={cos:.8f}")

        #         # 보수적으로 기준 잡기 (환경/정밀도 따라 조금 달라질 수 있음)
        #         # - cosine이 0.999999 이상에 가깝고
        #         # - max_abs가 아주 작으면 (예: 1e-3 ~ 1e-2 이하) 누출 없다고 볼 수 있음
        #         if cos < 0.9999 or max_abs > 1e-2:
        #             leaked_any = True

        #     print("[leak-test] RESULT:", "POSSIBLE LEAK" if leaked_any else "looks OK (no obvious leak)")

        breakpoint()
        break


if __name__ == "__main__":
    main()

    # import os
    # local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    # print(f"local_rank: {local_rank}\n", flush=True)
    # global_rank = int(os.getenv("RANK", "-1"))
    # print(f"global_rank: {global_rank}\n", flush=True)
    # SLURM_GPUS_ON_NODE = int(os.getenv("SLURM_GPUS_ON_NODE", "-1"))
    # print(f"SLURM_GPUS_ON_NODE: {SLURM_GPUS_ON_NODE}\n", flush=True)
