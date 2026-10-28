import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# this specifies that vllm execution will not be as subprocess but in tame python process. because cuda.profile/ncu tracks this process.  
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt



def main():
    model= sys.argv[1]
    prompt_length=int(sys.argv[2])
    output_length= int(sys.argv[3])

    llm = LLM(model=model, dtype="bfloat16")
    prompt= TokensPrompt(prompt_token_ids=[1000]* prompt_length) # produce a fake prompt
    params = SamplingParams(max_tokens=output_length, ignore_eos=True) # producing settings, produces untol max_tokens.

    for _ in range(3):
        llm.generate(prompt,params)
    
    torch.cuda.profiler.start() # ncu profiling starts
    llm.generate(prompt, params)
    torch.cuda.profiler.stop()

if __name__ == "__main__":
    main()

