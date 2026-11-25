import sys
import time
import csv
import os
import threading


import torch
import pynvml
from transformers import AutoModelForCausalLM


def main():
    model_name = sys.argv[1]
    prompt_len = int(sys.argv[2])   # prefill: 4096, decode: 1
    out_len = int(sys.argv[3])      # prefill: 1, decode: 1024
    tag = sys.argv[4] if len(sys.argv) > 4 else ""   # e.g.: cap_280, clock_780

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
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

    pynvml.nvmlInit() # connect to nvml library
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # get the handle for GPU 0

    samples = [] # the list we will use to store power samples and clock.
    stop_flag = threading.Event() # the flag to stup the backforund thread.

    def sample_power(): # the loop that will run in the backforund continuesly until the stop_flag sit set.
        while not stop_flag.is_set():
            t = time.perf_counter() # get the current time.
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # read the current power usage in watts (normally milliwatt but we divided by 1000).
            clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            samples.append((t, power_w, clock_mhz)) # add the time power pair to the list.
            time.sleep(0.05) # wait 50 miliseconds before the next sample.

    with torch.no_grad(): # close gradient calculation for inference.
        for _ in range(3):   # warmup 3 times
            model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        torch.cuda.synchronize() # waits to finish 3 warmups.
        t0 = time.perf_counter() # save the actual start time.

        thread = threading.Thread(target=sample_power)# start he sample_power func in different thread. (necessary to run in the background)
        thread.start()

        model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        torch.cuda.synchronize()# when generate is done, wait for all the GPU work to finish.
        stop_flag.set() # stop the background thread.
        thread.join() # wait to finish the background thread.
        t1 = time.perf_counter() # t1 is the time when the measurement is done.

    latency_s = t1 - t0 # time taken for the measured run. time spent for generating 1024 tokens in our case.

    energy_mJ = 0.0
    # trapezoidal integration to calculate energy in millijoules.
    for (t_a, p_a, _), (t_b, p_b, _) in zip(samples, samples[1:]):
        energy_mJ += 0.5 * (p_a + p_b) * (t_b - t_a) * 1000.0
    
    mean_power_w = sum(p for _, p, _ in samples) / len(samples)
    mean_clock_mhz = sum(c for _, _, c in samples) / len(samples)

    print(f"tag: {tag}")
    print(f"latency: {latency_s:.3f} s")
    print(f"tokens: {out_len}")
    print(f"throughput: {out_len / latency_s:.2f} tok/s")
    print(f"energy: {energy_mJ:.1f} mJ")
    print(f"energy per token: {energy_mJ / out_len:.2f} mJ/tok")
    print(f"mean power: {mean_power_w:.1f} W")
    print(f"mean sm clock: {mean_clock_mhz:.0f} MHz")
    print(f"power samples collected: {len(samples)}")

    csv_path = "energy_profile_results.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "tag", "model_name", "prompt_len", "out_len",
                "latency_s", "throughput_tok_s", "energy_mJ",
                "mJ_per_tok", "mean_power_w", "mean_clock_mhz", "num_samples",
            ])
        writer.writerow([
            tag, model_name, prompt_len, out_len,
            f"{latency_s:.3f}", f"{out_len / latency_s:.2f}", f"{energy_mJ:.1f}",
            f"{energy_mJ / out_len:.2f}", f"{mean_power_w:.1f}", f"{mean_clock_mhz:.0f}",
            len(samples),
        ])


if __name__ == "__main__":
    main()