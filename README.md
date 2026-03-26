# Multicamera Calibration Library

Multicamera calibration based on pycolmap.


## Installation
For installing, we recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Then, you can install using uv:

```bash
uv pip install -e .
```

## Example usage
Change paths in main() and then run:

```
python run_calibration.py
```

## Todo

- [ ] Add image quality check
- [ ] Add automatic detection
- [ ] Add ceres BA with targets
- [ ] Add scaling with scalebars

