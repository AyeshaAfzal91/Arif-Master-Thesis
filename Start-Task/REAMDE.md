## Start-task

### First Step: Environment setup:
- vLLM <https://vllm.ai/>, pytorch and cuda libraries might be GBs. Therefore, first proceed with creating a directory:

- `ws_allocate pwr_repro 60` = creates workspace for 60 days named pwr_repro. 
- `WS=$(ws_find pwr_repro)` = ws_find gives the path of this workspace, later this path is assocated to WS variable to easier use.
- `echo $WS`

### Second step: necessary modules and downloads:
- `module load python/3.12-conda`
- `export PIP_CACHE_DIR=$WS/pip_cache`
- `export UV_CACHE_DIR=$WS/uv_cache` 

These 2 exports routes the pip and uv caches to workspace instead of $HOME.

- `python -m venv $WS/venv`: creates an virtual python environment in workspace.
- `source $WS/venv/bin/activate` : activates the environment.

- `pip install --upgrade pip uv` : updates the pip and install uv
- `uv pip install vllm` : downloads and installs vllm and its' dependencies (cuda, pytorch).

### Third Step: downloading the models
- `export HF_HOME=$WS/hf_cache` : indicats where the huggingface models will be downloaded
- `which hf` : tells whether the hf command is downloaded.

#### Models:
- GQA(Qwen3-4B) : <https://huggingface.co/Qwen/Qwen3-4B> - not sure
- GQA-ctrl(Mintron-4B) : <https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base>
- MLA (Mintron-4B): <https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base>
- GDN(Qwen3.5-4b) : <https://huggingface.co/Qwen/Qwen3.5-4B> - not sure
- Mamba2(nemotron-nano): <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16>

- `hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- `hf download nvidia/Llama-3.1-Minitron-4B-Depth-Base`
- `hf download Qwen/Qwen3-4B`
- `hf download Qwen/Qwen3.5-4B`


### installed version:
- vllm= 0.31.0
- pytorch = 2.13.0
- cuda= 13.0 
- python= 3.12.7
- transformers = 5.17.0

### Fourth step: allocate a node and check the environment
- `source /hnvme/workspace/ihpc170h-pwr_repro/venv/bin/activate` = activate the venv in the node
- `export HF_HOME=/hnvme/workspace/ihpc170h-pwr_repro/hf_cache` = find the model cache directory

- `module load cuda/13.0.2` = load the  latest cuda module

### Fifth step: run the model

`ncu --profile-from-start off --clock-control base --section SpeedOfLight_RooflineChart --section SpeedOfLight_HierarchicalTensorRooflineChart --export prefill.ncu-rep python run_model.py nvidia/Llama-3.1-Minitron-4B-Depth-Base 4096 1`

or run the one of the .sh scripts, e.g. `sbatch profile_prefill.sh`

