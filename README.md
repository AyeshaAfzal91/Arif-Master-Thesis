## Administrative tasks
- GitHub: https://github.com/AyeshaAfzal91/Arif-Master-Thesis/tree/main 
  -- Move all results of Starting-task in “Starting-task” folder
  -- Document “summary of changes” for each “existing task“ or “summary of steps” for each “new task” in “README”
  -- Always commit current results at Github
- Student Email
  -- FAU: mehmet.a.bagc@fau.de
  -- Gmail: arif.bagci71@gmail.com
  -- GitHub: https://github.com/mrfbgc
- Thesis listed on website ⇒ TODO (Thesis Initial title: “??”?
- GitHub access ⇒ Done 
- Student listed on mailing list: hpcplus@fau.de ⇒ Done 
- Thesis duration: Nov – April (flexible) ??
- Thesis Requirements: Two talks and a comprehensive written thesis
- Thesis Registration: Nov (flexible)??
- First talk: Nov/Dec (flexible)??
- Second talk/Thesis: End-April (flexible)??
- Weekly meetings dates (THU, 10 a.m.): calendar invite set-up DONE 
- Access to HPC systems at NHR@FAU
  -- Alex (Done)
https://portal.hpc.fau.de 
ssh -Y ihpc171h@alex.nhr.fau.de
  -- Helma (TODO)
Arif filled the form and I’ll inform the admin ⇒ TODO
- Matrix Room for chat: Please set up an account at matrix.org (should look something like “@m.mouse:matrix.org”) and send it to me! => @mrfbgc:matrix.org ⇒ Done 
- Thesis Context (Integration into Wattlytics)
  -- Hardware: H100/H200 Helma or Testcluster GPUs
  -- Applications (compute-bound and/or memory-bound)
--- Three LLMs (focus on dense)
Mamba2 hybrid (state space models)
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
MLA (multi-head-latent attention from deepseek)
TransMLA this model: https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base 
GQA (group query attention)
https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base 
--- AI benchmarks from computer vision
FLUX1 dev: https://huggingface.co/black-forest-labs/FLUX.1-dev 
ResNet-50: https://huggingface.co/microsoft/resnet-50 
- Reference papers (running knobs: freq scaling and power capping)
-- LLM: https://arxiv.org/abs/2605.11999
  --- Hardware: NVIDIA H200
  --- LLM models: GQA, MLA, Gated DeltaNet, and Mamba2)
-- AI: https://arxiv.org/abs/2603.16164
  --- NVIDIA H100, NVIDIA H200, and AMD MI300X GPUs
  --- AI model: a benchmarking framework with popular deep learning applications from computer vision (image classification and generation) and large language models (continued pre-training and inference) implementing modern methods.
-- Experimental setting general: always use the updated software version available.  

## Starting Task
Please complete a jump-start task focused on GPU energy profiling and performance analysis of LLM inference workloads.

## 1. Environment Setup
Identify and access a test cluster <https://doc.nhr.fau.de/clusters/testcluster> with Nvidia H200 GPU or use A100 on Alex, and set up the required stack using apptainer or a virtual env:
PyTorch
Hugging Face Transformers
Mamba2 (state-spaces/mamba2-130m)
Transformer baselines (EleutherAI/pythia-160m, EleutherAI/pythia-70m)
Hints: If you hit the Mamba-SSM kernel compilation issue, the following combination on cip7b2 is known to resolve it: CUDA 13.3 driver + gcc 14.2, a conda env with Python 3.11, torch 2.12.1+cu130, transformers 5.12.1, mamba-ssm with causal-conv1d 1.6.2 (build this from source), cuDNN 9.20, and triton 3.7.1. The ciptmp directory should have enough space for the build as long as you have not enabled the pip cache.

## 2. Model Execution & Benchmarking
Run inference workloads with the selected models and measure/log:
GPU power consumption (via NVML APIs)
Throughput (tokens per second)
Energy per token
Hints: You may initially treat inference as a single stage (i.e., without separating prefill and decode). A few measurement points that matter for getting trustworthy numbers: make sure you have a proper warm-up phase before recording, and that your benchmark iterations are long enough for NVML to capture meaningful readings. For very short sequences and small models, a single forward pass can complete in under 100 ms, so coarse sampling intervals will miss it entirely. You can consider using nvidia-ml-py (version 13.610) with explicit polling around the forward pass.

## 3. Parameter Sensitivity Analysis
Analyze how the following factors affect energy consumption and throughput:
Sequence length
Batch size
Model architecture (Transformer vs. Mamba-style) and total parameter count
Hints: The core goal of this benchmark is to find the crossover point where Mamba2 begins to outperform the Transformer baselines. The three models were chosen with comparable parameter counts precisely to isolate architectural differences. Since Mamba2's key advantage is linear scaling of compute and memory with sequence length (versus quadratic for standard softmax attention), the interesting regime starts well above 512 tokens — a range of 512 or below tells no story.
Concretely:
Re-run with sequence lengths of 1K, 2K, 4K, … all the way up to 32K (fix batch size at 1), logging throughput and energy per token across that range. You should see clear inflection points.
Then scale sequence length until memory is exhausted, and visualize energy per token and tokens per second as heatmaps across batch size and sequence length. That will make the trade-offs immediately visible.
Identify and discuss the trade-offs between energy, batch size, and sequence length, including any observed sweet spots or intersection points.

## 4. Reporting
Give a presentation and hand in a README report on GitHub together with your benchmarking code, summarizing:
Experimental setup
Measurement methodology
Key findings and insights
Visualizations where appropriate (energy vs. sequence length, throughput vs. batch size, the heatmaps above)
Assumptions and limitations

For the analysis, what I want to see is your interpretation of the results: what the trends mean, where the crossover is, whether it matches the theoretical expectation, and why or why not. Plots without discussion are not sufficient to demonstrate your understanding of the core trade-offs.
