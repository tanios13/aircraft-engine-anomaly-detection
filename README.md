# Aircraft Engine Anomaly Detection

Anomaly detection in aircraft engines using state-of-the-art (SOTA) zero-shot models.

---

## Project Structure

```bash
├── LICENSE
├── README.md                        # You're reading this!
├── data
│   └── ...
├── main.py                          # Main script or entry point
├── notebooks
│   ├── test.ipynb
│   └── test_prediction.ipynb
├── pyproject.toml                   # Project metadata & dependencies
├── requirements.txt
├── src
│   ├── aircraft_anomaly_detection
│   │   ├── __init__.py
│   │   └── models
│   │       ├── __init__.py
│   │       ├── clip_predictor.py
│   │       ├── config
│   │       │   └── GroundingDINO_SwinT_OGC.py
│   │       ├── dino.py
│   │       ├── owlvit.py
│   │       ├── sam.py
│   │       └── yolo.py
│   └── dataloader
│       ├── __init__.py
│       └── loader.py
└── uv.lock                          # uv-specific lock file
```

- **`src/aircraft_anomaly_detection/`**: Primary Python package.
- **`notebooks/`**: Jupyter notebooks for exploration and prototyping.
- **`main.py`**: Example script or possible CLI entry point.
- **`pyproject.toml`**: Defines project requirements, build system, and metadata.

## Prerequisites

- **Python 3.10** (Make sure you have Python 3.10.x installed)
- **uv** (a fast Python package manager) or **pip** (standard Python installer)

### Installing uv

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (Powershell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

---

## Installation

### Option A: Using `uv` (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/aircraft-engine-anomaly-detection.git
   cd aircraft-engine-anomaly-detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   uv venv --python 3.10
   source .venv/bin/activate
   ```
   *(On Windows, use `./.venv/Scripts/activate`)*

3. **Install the project**:
   ```bash
   uv sync
   ```
   This installs all required dependencies from `pyproject.toml`.

4. **(Optional) Install dev dependencies**:
   ```bash
   uv sync --all-extras
   ```

5. **(Optional) Editable Install** (like `pip install -e .`):
   ```bash
   uv pip install -e .
   ```
   Allows immediate reflection of code changes.

6. **Install specific  pip dependencies**:
   ```bash
   uv pip install 'transformers[torch]'
   ```

### Option B: Using standard pip

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/aircraft-engine-anomaly-detection.git
   cd aircraft-engine-anomaly-detection
   ```
2. **Create and activate a Python 3.10 virtual environment**:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install -e .
   ```
### Check your installation
```bash
python -c "import aircraft_anomaly_detection; print('Package loaded, OK!')"
```
---

## Usage

After installing, run:
```bash
python main.py
```

Or import inside a Python session:
```python
from aircraft_anomaly_detection.models import clip_predictor
# ...use your modules...
```

---

## Development

1. **Install dev dependencies** (if you haven’t already):
   ```bash
   uv sync --all-extras
   ```
2. **Lint and format**:
   ```bash
   ruff check . --fix
   ```
3. **Type checking**:
   ```bash
   mypy .
   ```
4. ** VSCode**: Download ruff and mypy extensions for linting and type checking.
---

## Troubleshooting

- **Multiple top-level packages**: If you see an error about multiple packages discovered (e.g., `data`, `notebooks`, etc.), the `src` layout ensures that only `src/aircraft_anomaly_detection` is treated as a package.
- **Missing library stubs**: If MyPy complains, add `ignore_missing_imports = true` under `[tool.mypy]` in `pyproject.toml`. Aternatively, add a `# type: ignore` comment to the line.
  
- **Git-based installs**: If `uv` fails for certain Git-based sources, manually install with:
  ```bash
  pip install git+https://github.com/some_project/some_repo.git
  ```
  or try
  ```bash
  uv pip install some_package
  ```

---

## License
MIT License. See `LICENSE` for details.


