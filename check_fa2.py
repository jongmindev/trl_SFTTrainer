import torch
from flash_attn import flash_attn_func

def check_flash_attention():
    # 1. GPU 가용성 확인
    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없는 환경입니다.")
        return

    # 2. FlashAttention은 bfloat16 또는 float16에서 작동합니다.
    device = "cuda"
    dtype = torch.bfloat16
    
    # 임의의 텐서 생성 (Batch, SeqLen, Heads, Dim)
    # FA2는 Dim이 128 이하일 때 최적의 성능을 냅니다.
    q = torch.randn(1, 128, 8, 64, device=device, dtype=dtype)
    k = torch.randn(1, 128, 8, 64, device=device, dtype=dtype)
    v = torch.randn(1, 128, 8, 64, device=device, dtype=dtype)

    try:
        # 실제 FlashAttention 커널 호출
        output = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=True)
        
        if output is not None:
            print(f"✅ FlashAttention-2 설치 성공!")
            print(f"✅ 연산 결과 텐서 모양: {output.shape}")
            print("✨ FlashAttention-2 is working perfectly!")
            
    except Exception as e:
        print(f"❌ 설치는 된 것 같으나 실행 중 오류가 발생했습니다:")
        print(f"오류 내용: {e}")

if __name__ == "__main__":
    check_flash_attention()
