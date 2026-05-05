# GPU Container

Reusable CUDA + `uv` image for Prime Intellect, Vast.ai, or any H100/A100 host.

Build from the repo root:

```bash
docker build -f docker/Dockerfile.gpu -t em-exp:gpu .
```

Run with GPU access:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN \
  -e WANDB_API_KEY \
  -v "$PWD/data:/workspace/emergent_misalignment_experiments/data" \
  -v "$PWD/outputs:/workspace/emergent_misalignment_experiments/outputs" \
  -it em-exp:gpu
```

Inside the container:

```bash
uv run python scripts/data/materialize_sources.py --config configs/experiment_1.yaml
```
