import os, sys
import torch
from transformers import AutoModelForCausalLM

def main():
    model_name = sys.argv[1]
    prompt_len = int(sys.argv[2])   #prefill: 4096, decode: 1
    out_len = int(sys.argv[3])      #prefill: 1, decode: 1024

    # trust_remote_code=True: MLA checkpoints (TransMLA output) ship their own
    # modeling_*.py files alongside the weights and need this to load.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Pytorch optimized kernel for attention
        trust_remote_code=True,
    ).to("cuda")
    model.eval()

    input_ids = torch.full((1, prompt_len), 1000, dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)

    gen_kwargs = dict(
        max_new_tokens=out_len,
        min_new_tokens=out_len,
        do_sample=False,
        use_cache=True,
    )

    with torch.no_grad():
        for _ in range(3):        # 3 warm-up
            model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        torch.cuda.profiler.start()
        model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        torch.cuda.profiler.stop()

if __name__ == "__main__":
    main()
