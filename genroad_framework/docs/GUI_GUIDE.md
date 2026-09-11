# Web GUI Guide

## Load and select an image

In **Load & BBox**, enter an image directory, load it, and select an image.
Click the top-left and bottom-right corners of the object region. The red
rectangle is the inpainting region, not necessarily the final object outline.

## Inpainting

Choose an object type, seed, inference steps, and guidance scale. Run
inpainting to create a raw candidate. Rejecting a candidate restores the
previous accepted state; retrying with the same seed is useful for debugging.

## SAM and harmonization

In **Harmonization**:

1. Click the generated object.
2. Inspect the green SAM mask.
3. Accept the mask, or use Expand Mode for disconnected object parts.
4. Choose a harmonization method and preview settings.

Accept the harmonized result to continue to annotation and weather
transformation. The SAM and harmonization ablation variants used during paper
experiments are not part of the public production workflow.

## Annotation review

The Annotations tab supports two-click bounding-box editing and optional SAM
assistance. Review labels and boxes before saving.

## Weather transformation

Select weather effects and generate them independently. Each weather result
can be regenerated with its own seed and accepted or rejected. The selected
pipeline variation is included in saved filenames and metadata.

## Saving

The Save tab writes the accepted inpainting and weather results. Use a unique
prefix for each source image. Do not write output into the source repository
when preparing a pull request; use a local output directory that is ignored by
Git.

## Troubleshooting

- **Model not found:** verify `configs/config.yaml` and the CosXL checkpoint path.
- **CUDA out of memory:** reduce image size, inference steps, or model precision.
- **Port in use:** launch with `--port 8080`.
- **Share link concerns:** use the default localhost mode for private images.
- **SAM is slow:** reduce `models.sam.crop_size` in the local config.
