import argparse
import csv
import os
import threading
import time

import torch
import torch.nn.functional as F
import lightning as L
import pynvml

from litgpt.config import Config
from litgpt.model import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3-0.6B")   # a LitGPT config name, e.g. Qwen3-0.6B, Qwen3-4B
    p.add_argument("--seq-len", type=int, default=2048) # sequence lengh for each training step.
    p.add_argument("--micro-batch", type=int, default=1) # batch size for each training step.
    p.add_argument("--warmup", type=int, default=3) # number of warmup steps to run before measuring throughput.
    p.add_argument("--steps", type=int, default=10) # number of training steps that will be measured.
    p.add_argument("--precision", default="bf16-mixed")
    args = p.parse_args()

    fabric = L.Fabric(devices=1, precision=args.precision) # for now only ONE GPU (no fsdp),
    fabric.launch()

    config = Config.from_name(args.model)  # architecture only(layer number, head count, embed size etc.), no weights are downloaded
    fabric.print(f"model: {args.model}, precision: {args.precision}, "
                 f"seq_len: {args.seq_len}, micro_batch: {args.micro_batch}")

    with fabric.init_module(empty_init=False):  #model params are init in GPU directly. weights are initialited randomly.
        model = GPT(config) # build a model with the given config (weights are initialized randomly).
    model.max_seq_length = args.seq_len # set max sequence length that model can handle.

    n_params = sum(p.numel() for p in model.parameters()) 
    fabric.print(f"parameters: {n_params / 1e9:.3f} B") # print the total param number for verification.

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=True) # optimizer setup with learning rate and weight decay. fused=True means using fused AdamW implementation for better performance.
    model, optimizer = fabric.setup(model, optimizer) # setup the model and optimizer 
    model.train() # model is set to training mode.

    vocab = config.padded_vocab_size # vocab size suitable for gpu.

    def make_batch(): # genertaes random token between zero and vocab.
        # fake data: random token ids, target = input shifted by one token
        x = torch.randint(0, vocab, (args.micro_batch, args.seq_len + 1), device=fabric.device)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def train_step():
        input_ids, targets = make_batch()
        logits = model(input_ids)                                        # forward pass
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        fabric.backward(loss)                                            # backward pass
        optimizer.step()                                                 # update weights
        optimizer.zero_grad(set_to_none=True)                            # down zero gradients for next step
        return loss

    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # --- energy measurement setup via nvml ---
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    samples = []  # (timestamp, power_w, sm_clock_mhz)
    stop_flag = threading.Event()

    def sample_power():
        while not stop_flag.is_set():
            t = time.perf_counter()
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            samples.append((t, power_w, clock_mhz))
            time.sleep(0.05)

    thread = threading.Thread(target=sample_power)
    thread.start()

    t0 = time.perf_counter()
    for i in range(args.steps):
        loss = train_step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    stop_flag.set()
    thread.join()
    pynvml.nvmlShutdown()

    elapsed = t1 - t0

    # trapezoidal integration to calculate energy in millijoules.
    # energy=power x time  = joules
    # j x 1000 = millijoules
    # samples[1:]= the first value is removed, later with zip, we would have pairs of (t_a, p_a) and (t_b, p_b) for each consecutive sample.
    # p_a = start power, p_b = end power, t_a = start time, t_b = end time. (t_b - t_a) is the time interval between two samples.
    # avarage power = 0.5 * (p_a + p_b)
    # energy = average power * time interval = 0.5 * (p_a + p_b) * (t_b - t_a)
    energy_mJ = 0.0
    for (t_a, p_a, _), (t_b, p_b, _) in zip(samples, samples[1:]):
        energy_mJ += 0.5 * (p_a + p_b) * (t_b - t_a) * 1000.0

    mean_power_w = sum(p for _, p, _ in samples) / len(samples)
    mean_clock_mhz = sum(c for _, _, c in samples) / len(samples)

    tokens = args.steps * args.micro_batch * args.seq_len
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"last loss:       {loss.item():.3f}")
    print(f"time per step:   {elapsed / args.steps * 1000:.1f} ms")
    print(f"throughput:      {tokens / elapsed:,.0f} tokens/s")
    print(f"peak memory:     {peak_mem_gb:.1f} GB")
    print(f"energy:          {energy_mJ:.1f} mJ")
    print(f"energy per step: {energy_mJ / args.steps:.1f} mJ/step")
    print(f"energy per tok:  {energy_mJ / tokens:.2f} mJ/tok")
    print(f"mean power:      {mean_power_w:.1f} W")
    print(f"mean sm clock:   {mean_clock_mhz:.0f} MHz")
    print(f"power samples:   {len(samples)}")

    csv_path = "training_energy_results.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "seq_len", "micro_batch", "steps",
                "time_per_step_ms", "throughput_tok_s", "peak_mem_GB",
                "energy_mJ", "mJ_per_step", "mJ_per_tok",
                "mean_power_w", "mean_clock_mhz", "num_samples",
            ])
        writer.writerow([
            args.model, args.seq_len, args.micro_batch, args.steps,
            f"{elapsed / args.steps * 1000:.1f}", f"{tokens / elapsed:.2f}", f"{peak_mem_gb:.1f}",
            f"{energy_mJ:.1f}", f"{energy_mJ / args.steps:.1f}", f"{energy_mJ / tokens:.2f}",
            f"{mean_power_w:.1f}", f"{mean_clock_mhz:.0f}", len(samples),
        ])


if __name__ == "__main__":
    main()