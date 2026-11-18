import sys
import torch
import lightning as L
from pathlib import Path

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.generate.base import generate
from litgpt.utils import check_valid_checkpoint_dir, load_checkpoint, get_default_supported_precision


def main():
    checkpoint_dir = Path(sys.argv[1])
    prompt_len = int(sys.argv[2])   # prefill: 4096, decode: 1
    out_len = int(sys.argv[3])      # prefill: 1, decode: 1024

    check_valid_checkpoint_dir(checkpoint_dir) # check the yaml and jason files exist
    config = Config.from_file(checkpoint_dir / "model_config.yaml") # read the arch info e.g. layer count, head count, embed size.
    checkpoint_path = checkpoint_dir / "lit_model.pth"

    max_returned_tokens = prompt_len + out_len # max token numbers that model holds (prompt + generated tokens)- required for kv cache
    precision = get_default_supported_precision(training=False)  # bf16-true on H100

    fabric = L.Fabric(devices=1, precision=precision)

    with fabric.init_module(empty_init=True):
        model = GPT(config)
    with fabric.init_tensor():
        model.max_seq_length = max_returned_tokens
        model.set_kv_cache(batch_size=1)
    model.eval()
    model = fabric.setup_module(model)

    load_checkpoint(fabric, model, checkpoint_path)

    # fake prompt
    prompt = torch.full((prompt_len,), 1000, dtype=torch.long, device=fabric.device)

    gen_kwargs = dict(temperature=0.0, top_k=None, top_p=1.0, eos_id=None, include_prompt=False)
    # temperature=0.0 -> greedy/argmax decode (same as do_sample=False at HF)
    # eos_id=None -> do not stop at any token, generate until max_returned_tokens is reached

    def run_once():
        with torch.no_grad():
            generate(model, prompt, max_returned_tokens, **gen_kwargs)
        for block in model.transformer.h:
            block.attn.kv_cache.reset_parameters()  # reset kv cache for next run

    for _ in range(3):        # 3 warm-up
        run_once()

    torch.cuda.profiler.start()
    run_once()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
