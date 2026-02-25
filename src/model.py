import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(config, logger=None):
    args = config.model

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",    # 성능 극대화
        # attn_implementation="sdpa",  # <--- "flash_attention_2" 대신 "sdpa" 사용
        device_map=None                             # DeepSpeed 사용 시 None 필수
        )
    if config.deepspeed.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if logger:
        logger.info("[PROGRESS] Model loaded.")

    # tokenizer
    tokenizer_id = args.tokenizer if args.tokenizer else args.model
    # fix_mistral_regex = True if args.fix_mistral_regex else False
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        # fix_mistral_regex=fix_mistral_regex,      # 옵션 안 주면 주라고 warning 띄우지만, 정작 주면 알아서 한다고 error 냄.
    )

    assert tokenizer.pad_token_id is not None, f"tokenizer.pad_token_id is None"
    assert tokenizer.eos_token_id is not None, f"tokenizer.eos_token_id is None"
    # assert tokenizer.chat_template is not None, f"tokenizer.chat_template is None"

    if logger:
        logger.info("[PROGRESS] Tokenizer loaded.")

    return model, tokenizer


if __name__ == "__main__":
    import hydra
    
    @hydra.main(config_path="configs", version_base=None)
    def test(config):
        model, tokenizer = load_model_and_tokenizer(config)

    test()
