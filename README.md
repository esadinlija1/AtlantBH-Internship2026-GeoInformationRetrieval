# GeoIR

Information retrieval over geospatial data: OpenStreetMap (OSM) points of interest (POIs) for Bosnia and Herzegovina.
The project prepares the data for embedding-based search. Currently implemented: POI extraction from the Geofabrik PBF file and exploratory analysis of POI attributes.

## Prerequisites

Install [`uv`](https://docs.astral.sh/uv/):

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal, then verify:

```bash
uv --version
```

If you still get "command not found":

- Linux/macOS: add `$HOME/.local/bin` to `PATH`, e.g. `source $HOME/.local/bin/env`.
- Windows: check that `%USERPROFILE%\.local\bin` is in `PATH`.

uv downloads Python 3.11 by itself if it is missing. The system Python is irrelevant for this project.

## Environment setup

From the project root:

```bash
uv python pin 3.11
uv sync
```

`uv sync` installs the dependencies listed in `pyproject.toml` at the versions locked in `uv.lock`, including the `dev` group (`jupyterlab`, `ipykernel`).

Python 3.11 is used because `pyrosm` may fail to install on newer versions (3.12+). `uv.lock` is meant to be committed for reproducibility.

### How the environment was created (for reference only; not needed after cloning)

```bash
uv init --package --python 3.11
uv add pyrosm pandas numpy pyarrow
uv add sentence-transformers
uv add --dev jupyterlab ipykernel
```

`--package` matches the existing layout (`src/geoir/`, `[build-system]` and `[project.scripts]` in `pyproject.toml`).

### Jupyter kernel

Register the project environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name geo-ir --display-name "geo-ir"
```

### Installation check

`geopandas` and `shapely` are not listed directly in `pyproject.toml`. They are installed as dependencies of `pyrosm`.

```bash
uv run python -c "import pyrosm, geopandas, shapely; print(pyrosm.__version__, geopandas.__version__, shapely.__version__)"
```

### Launch Jupyter

```bash
uv run jupyter lab
```

If a notebook in VS Code uses the wrong kernel (e.g. Julia), select the `geo-ir` kernel or the interpreter in `.venv`.

## Data

Source: Geofabrik, <https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf>

The extraction script (`src/geoir/data/extract.py`) handles the download itself:

1. It creates `data/raw` and `data/interim` if they do not exist.
2. If `data/raw/bih.osm.pbf` does not exist, it downloads the PBF and its `.md5` file (to `data/raw/bih.osm.pbf.md5`). Each file is first downloaded to a `.part` file and renamed only when the download finishes.
3. If `data/raw/bih.osm.pbf.md5` exists, it computes the md5 of the PBF and compares it with the expected value. If they do not match, it stops with an error (the message is in Bosnian).

The PBF is downloaded only when it is missing. If the PBF exists but the `.md5` file does not, the md5 check is skipped.

### Manual download (alternative)

Linux/macOS:

```bash
mkdir -p data/raw data/interim
wget -O data/raw/bih.osm.pbf https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf
wget -O data/raw/bih.osm.pbf.md5 https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf.md5
md5sum data/raw/bih.osm.pbf
cat data/raw/bih.osm.pbf.md5
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force data\raw, data\interim
Invoke-WebRequest https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf -OutFile data\raw\bih.osm.pbf
Invoke-WebRequest https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf.md5 -OutFile data\raw\bih.osm.pbf.md5
(Get-FileHash data\raw\bih.osm.pbf -Algorithm MD5).Hash.ToLower()
Get-Content data\raw\bih.osm.pbf.md5
```

The two hashes must be identical. In Windows PowerShell 5, `&&` does not work. Run the commands one per line or separate them with `;`.

### Notes

- `latest` is a moving snapshot. The POI count and the md5 change when Geofabrik refreshes the file. If the md5 check fails after a refresh, delete the files in `data/raw` and rerun the script.
- An incomplete download of the PBF can still be read and gives a different POI count without any error. This is why the script downloads to a `.part` file and verifies the md5.
- `data/` and `.venv/` are in `.gitignore` and are not committed.

## Project structure

```
GeoIR/
├── .gitignore
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── data/                      # not committed
│   ├── raw/                   # bih.osm.pbf, bih.osm.pbf.md5
│   └── interim/
│       └── pois.parquet
├── notebooks/
│   └── eda.ipynb
└── src/
    └── geoir/
        ├── __init__.py
        └── data/
            └── extract.py
```

- `.gitignore`: ignores Python build artifacts, `.venv/` and `/data/`.
- `.python-version`: pins Python 3.11 for uv.
- `pyproject.toml`: project metadata, runtime dependencies (`numpy`, `pandas`, `pyarrow`, `pyrosm`, `sentence-transformers`), the `dev` group (`ipykernel`, `jupyterlab`) and the `uv_build` build backend.
- `uv.lock`: exact locked versions of all dependencies.
- `data/raw/`: the downloaded PBF and its md5 file.
- `data/interim/`: intermediate outputs. Currently only `pois.parquet`.
- `src/geoir/__init__.py`: defines `main()`, which only prints `Hello from geoir!`. It is exposed as the `geoir` command via `[project.scripts]`. This is the uv template placeholder. TODO: purpose not defined yet.
- `src/geoir/data/extract.py`: downloads and verifies the PBF, then extracts the POIs.
- `notebooks/eda.ipynb`: exploratory analysis of POI attributes.

### `src/geoir/data/extract.py`

- **Input:** `data/raw/bih.osm.pbf`, downloaded automatically if missing.
- **Output:** `data/interim/pois.parquet`, a GeoDataFrame from `pyrosm.OSM(...).get_pois()` with default settings. It also prints the POI count, the counts per `osm_type` and the bounding box.
- **Run** from the project root, because the paths are relative to the current directory:

  Windows (PowerShell):

  ```powershell
  uv run python src\geoir\data\extract.py
  ```

  Linux/macOS:

  ```bash
  uv run python src/geoir/data/extract.py
  ```

### `notebooks/eda.ipynb`

- **Input:** `../data/interim/pois.parquet`. The path is relative to `notebooks/`, so the notebook must run with `notebooks/` as its working directory (the default in Jupyter and VS Code).
- **Output:** printed tables only. No files are written.
- **What it does:**
  1. Merges each POI's `tags` (JSON) and non-empty columns into one attribute dict. Technical columns (`id`, `version`, `timestamp`, `geometry`, `lon`, `lat`, etc.) are skipped. It then builds a table of every attribute key with its count, coverage (%) and number of unique values.
  2. Shows the 25 most common `amenity` values.
  3. Shows how many key–value pairs are kept, and how many rows have no attributes left, when only the top-k most frequent keys are kept.
  4. Greedily picks descriptive keys that together cover the most POIs. Identifier and contact keys (`phone`, `website`, `email`, etc.) are excluded.
- **Run:** open it in Jupyter Lab (`uv run jupyter lab`) or VS Code with the `geo-ir` kernel, or run it headless:

  ```bash
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
  ```

## Pipeline / execution order

1. `uv sync`
2. `uv run python src/geoir/data/extract.py`: download the PBF, verify the md5 and write `data/interim/pois.parquet`.
3. `notebooks/eda.ipynb`: attribute analysis.

## Planned steps (not implemented)

None of the following exists in the code yet:

- Attribute normalization / selection of the final key set.
- Text generation from POI attributes.
- Embeddings. `sentence-transformers` is a dependency, but nothing uses it yet.
- Storage and search in PostgreSQL with pgvector/PostGIS.

## Git

The repository has no commits yet.

To be committed: `.gitignore`, `.python-version`, `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `notebooks/`.

Not committed (in `.gitignore`): `.venv/`, `data/`, `__pycache__/`, build artifacts.

Notebook outputs are currently saved in the `.ipynb` file. `nbstripout` is not set up. TODO: decide whether to add it (`uv add --dev nbstripout`, then `uv run nbstripout --install`) so outputs are stripped on commit. `.ipynb_checkpoints/` is not in `.gitignore` yet. TODO: add it.

## Known issues

- **`uv` not found after install:** reopen the terminal, or add `$HOME/.local/bin` (Linux/macOS) or `%USERPROFILE%\.local\bin` (Windows) to `PATH`.
- **Wrong notebook kernel** (e.g. Julia in VS Code): select the `geo-ir` kernel or the `.venv` interpreter.
- **`&&` in PowerShell 5:** not supported. Run the commands on separate lines or use `;`.
- **POI count differs between runs:** either Geofabrik refreshed `latest`, or the PBF is incomplete. Delete `data/raw/*` and rerun the extraction script, which re-downloads the file and verifies the md5.
