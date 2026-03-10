# Set env

- uv add
    ```bash
    uv add datasets deepspeed hydra-core torch transformers trl wandb
    
    # gpu 할당 받은 계산 노드에서 아래 실행 (nvidia driver 인식되어야 함)
    thunder-shell -g 1 -t 180 -m 200 -c 32              # 충분한 cpu core / system memory 필요. (병렬 컴파일) 
    MAX_JOBS=16 uv add flash-attn --no-build-isolation  # 다소 시간 걸릴 수 있음 (30분 이상)
    ```
- flash-attn 설치 확인
    ```bash
    uv run check_fa2.py
    ```
