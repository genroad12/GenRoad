# Contributing

Thank you for contributing to GenRoad.

## Development setup

1. Create a Python virtual environment.
2. Install the dependencies in `genroad_framework/requirements.txt`.
3. Copy `genroad_framework/configs/config.example.yaml` to a local
   `config.yaml`.
4. Run the model-free tests with `pytest`.

Do not commit model checkpoints, datasets, generated images, local
configuration files, credentials, or absolute workstation paths.

## Pull requests

- Keep changes focused and explain the motivation.
- Add or update model-free tests for behavior changes.
- Document user-facing changes in the relevant README or guide.
- Run syntax checks, tests, and the configured linter before submitting.
- Do not claim support for a model, device, or workflow that was not tested.

