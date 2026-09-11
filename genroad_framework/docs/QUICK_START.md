# Quick Start

## 1. Install

```bash
cd genroad_framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-compatible PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
```

Set the CosXL Edit checkpoint path in `configs/config.yaml`.

## 2. Start

```bash
python -m app.main
```

Open `http://127.0.0.1:7860` in a browser.

## 3. Generate an image

1. Load a folder and select an image.
2. Click two corners to define the object bounding box.
3. Select an object type and run inpainting.
4. In the Harmonization tab, click the object and accept its SAM mask.
5. Choose a pipeline variation and accept it.
6. Generate one or more weather transformations.
7. Review annotations and save the result.

The first model invocation downloads Hugging Face dependencies and can take
several minutes. The CosXL Edit checkpoint must be provided manually.
