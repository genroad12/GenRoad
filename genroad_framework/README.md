# GenRoad Framework

GenRoad Framework is the interactive image-generation tool used to create the
synthetic anomaly images released with the [GenRoad paper and
dataset](https://github.com/sahibinden/GenRoad). It inserts a text-guided
object into a road scene, optionally harmonizes the object with the scene,
allows its annotation to be reviewed, and applies a global weather or scene
transformation.

The supported public interface is the Gradio web application.

## Pipeline

1. Load an image and draw an object bounding box.
2. Generate the object with Stable Diffusion 2 Inpainting.
3. Refine the object boundary with Segment Anything (SAM).
4. Apply image harmonization and review the annotation.
5. Transform the complete scene with CosXL Edit.
6. Save the image and reproducibility metadata.

The SAM and harmonization ablation variants used for the paper are research
experiments and are intentionally not part of the public production UI.

## Requirements

- Python 3.10 or newer
- NVIDIA GPU with CUDA support recommended
- At least 16 GB VRAM for the complete workflow; larger images may require more
- A local CosXL Edit checkpoint supplied under its model license

Stable Diffusion, SAM, and supporting Hugging Face models are downloaded on
first use. Model checkpoints are not included in this repository.

## Installation

```bash
cd genroad_framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch wheel matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/), then install the
remaining dependencies:

```bash
python -m pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
```

Edit `configs/config.yaml` and set `models.scene_transform.model_path` to the
CosXL Edit checkpoint. Keep this file local; it is ignored by Git.

## Launch

From the `genroad_framework` directory:

```bash
python -m app.main
```

The application listens on `http://127.0.0.1:7860` by default. Useful options:

```bash
python -m app.main --port 8080
python -m app.main --config /path/to/config.yaml
python -m app.main --share
```

`--share` creates a temporary public tunnel. Do not use it with private images
or an untrusted multi-user deployment. The application is intended for
research and local/demo use; production deployments should add authentication,
resource limits, and persistent per-user isolation.

The equivalent launcher is:

```bash
./scripts/start_gui.sh --local
```

## Outputs

The Save tab writes PNG images and JSON metadata. Output filenames include the
selected pipeline mode, for example:

```text
scene01_bboxHarm_snow.png
scene01_bboxHarm_snow.json
```

Metadata records the pipeline mode, seed, annotation, weather prompt, and
model configuration without exposing the workstation's absolute filesystem
paths.

## Configuration

The example configuration is intentionally portable. Relative paths are
resolved from the `genroad_framework` directory. The most important sections
are:

- `models.inpainting`: Stable Diffusion inpainting model and sampling settings
- `models.scene_transform`: CosXL Edit checkpoint
- `models.sam`: SAM variant and mask settings
- `gui.defaults`: initial GUI values
- `scene_transform.weather_configs`: weather prompts

## Model and dataset licensing

This tool is released under Apache-2.0. Third-party model weights have their
own licenses and terms. Review the licenses of Stable Diffusion, SAM, CosXL
Edit, and every downloaded Hugging Face model before redistribution or
commercial use. The GenRoad dataset and paper are documented in the
[main repository](https://github.com/sahibinden/GenRoad).

## Limitations

The genroad framework is a research tool and does not guarantee physical correctness,
annotation correctness, or safety-critical performance. Generated images must
be reviewed by a human before being used in a dataset or evaluation.

## Documentation

- [Quick start](docs/QUICK_START.md)
- [GUI guide](docs/GUI_GUIDE.md)
- [Configuration template](configs/config.example.yaml)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
