# PepSpace Visualization

PepSpace is an interactive Streamlit application for exploring bioactive peptide embedding spaces. It visualizes peptide datasets with UMAP and densMAP, supports default Veltri data paths, and allows users to upload custom embedding and metadata files.

## Features

* Interactive 2D UMAP and densMAP projections
* Consistent class colors in every dimension: AMP blue, non-AMP red, unlabeled grey
* Hover inspection for peptide IDs, sequences, and labels
* Upload custom embeddings (`.npy`) and metadata (`.csv`)
* Metadata source switching between processed Veltri metadata and raw Veltri CSV
* Level ScatterGL globe with rotation, scroll zoom, optional spin, and a reproducible reset view
* Native ScatterGL point and text-label modes, hover inspection, and 3D box selection
* Optional labeled axes, pointer hover labels, and Front / Side / Top camera presets
* Switchable light and dark themes shared by the page, tables, and both plots
* Points inside the globe by default, with optional sphere surface projection
* Linked Plotly 2D box/lasso and ScatterGL 3D box selection, with export of selected peptides
* Automatic globe filtering to the selected source rows, retained across 3D–8D modes
* Adjustable UMAP and densMAP parameters from the sidebar
* Cached data loading and persistent projection results for faster interaction and restarts
* Exact full-data trustworthiness scores calculated on demand and cached

These features are available in `app.py`.

## Python Version

Python 3.12 is recommended. This project was developed with Python 3.12.10.

## Installation

Clone the repository:

```bash
git clone https://github.com/alexghyet/Pepspace-Visualization
cd Pepspace-Visualization
```

On Windows, install Python 3.12 first if it is not already installed. Download it from [python.org](https://www.python.org/downloads/) and enable the Python launcher during installation. Open a new terminal afterward and confirm that `py --version` reports Python 3.12. If `python` opens the Microsoft Store or reports that Python was not found, the interpreter is not installed or available on `PATH`; installing a virtual environment cannot replace installing Python.

On Windows ARM devices, use the x64 build of Python 3.12 for this project. Some pinned scientific packages may not have ARM64 wheels, which can make pip try to compile them locally.

Create a virtual environment.

Windows (PowerShell or Command Prompt):

```bash
py -3.12 -m venv .venv
```

Mac/Linux:

```bash
python3 -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

Start the main Streamlit app:

```bash
python -m streamlit run app.py
```

After launching, Streamlit will print a local URL and usually open the application in your browser automatically.

If your installed dependencies are in `.venv-x64`, run this in PowerShell:

```powershell
.\.venv-x64\Scripts\python.exe -m streamlit run app.py
```

## Globe Controls and Structure

The camera menu offers **Front (XY)**, **Side (YZ)**, and **Top (XZ)** views. Moving or spinning the camera changes the menu to **Custom view**; Reset returns to Front. Presets change only the camera, leaving projections, selections, and exports intact. **Show axes** enables neutral axis lines and X/Y/Z annotations that follow camera rotation. **Show hover labels** enables a compact label beside the pointer; the full peptide details remain below the plot.

For a 3D UMAP/densMAP result, the axis descriptions name that result's three components. Outputs with 4–8 dimensions already use deterministic PCA inside `utils/sphere_projection.py` for display, so their descriptions read **Display PC 1–3**. The 2D and 3D UMAP/densMAP outputs are displayed without PCA. These are centered/scaled display coordinates rather than direct biological measurements.

The default **Inside globe** view centers the 3D UMAP coordinates and applies one uniform scale so that the points fit inside a complete wireframe globe. Translation and uniform scaling preserve all relative distances, angles, and nearest neighbors in the 3D UMAP result. They do not undo information lost by UMAP when reducing the original embeddings.

The globe uses the actual [ScatterGL](https://github.com/PAIR-code/scatter-gl) WebGL renderer. It starts stationary with vertical poles and a horizontal equator. The grid follows the camera's Y-up convention, with exactly closed latitude and longitude circles. Choose **Rotate / Pan** to explore the scene, scroll to zoom, and select **Reset view** to return to the same level camera. **Spin globe / Pause spin** controls optional rotation. **Points / Labels** switches between point markers and native ScatterGL text labels. Hover a peptide to inspect its ID, sequence, and AMP status. These browser controls reuse the existing projection. Unsupported characters in native text labels display as `?`; full IDs remain available in hover details, tables, and exports.

Box selection is enabled initially in the 2D plot. Select a region to show **only those peptides** in the globe automatically. Switching between 3D–8D, changing point styling, or applying projection settings keeps the same selected source rows. Clear the selection to restore all peptides. Loading a different embedding array or metadata table clears the selection. Different projections place the same peptides at different coordinates; filtering preserves each full projection's coordinates and bounds instead of refitting or recentering the selected subset.

Choose **Select region** and drag a rectangle over the 3D globe to refine the selection. Native selections update the table, exports, and 2D highlights. AMP stays blue, non-AMP stays red, and unlabeled peptides stay grey. Original row indices remain the identity across both views, even if IDs repeat or browser filtering changes the displayed indices. Only fresh selection gestures update the selected rows; a temporary empty Plotly widget state cannot erase a valid selection.

**Sphere surface** centers the coordinates before normalizing each point to unit length. This avoids the origin-dependent patch produced by normalizing uncentered UMAP coordinates. Radial normalization changes distances and can change neighborhoods; use **Inside globe** with **Sphere Dimensions = 3** when preserving the 3D layout matters. A point exactly at the centroid remains at the origin because it has no radial direction. Clusters are not artificially spread to cover the whole sphere.

For UMAP outputs with 4–8 dimensions, the display uses deterministic PCA to obtain three coordinates. This extra projection loses information. The n-dimensional UMAP result is retained, and optional quality scores distinguish it from the displayed coordinates.

## Light and Dark Appearance

Open the app's top-right **⋮ → Settings → Choose app theme** selector and choose **Light**, **Dark**, or **Use system setting**. The native Streamlit selector applies the palettes in `.streamlit/config.toml` across the page, sidebar, tables, Plotly chart, and ScatterGL view. Theme changes keep the current selection and camera; peptide colors remain blue for AMP, red for non-AMP, and grey for unlabeled. Each browser controls its own preference.

## Performance

The app fits two independent projections: 2D for selection and the selected number of dimensions for the globe. Both use the same active sidebar settings and seed. `optuna_results/3d_params.json` supplies the initial shared baseline, with validated defaults if it is absent or invalid. Changing the globe dimension does not change these settings or move the 2D projection. Apply sidebar edits explicitly to change projection settings. Defaults tuned for Veltri can be adjusted and evaluated for other datasets.

* Edit the parameters and click **Apply projection settings** once to fit them together.
* Projection results are saved in `.cache/projections/` and reused for matching embedding contents, parameters, and scientific-library versions, including after restarting the app.
* Display changes and selections reuse the saved projections. The app retains the random seed, all samples, and default optimization epochs.
* **Calculate exact quality scores** is off initially. Enable it to compute full-data trustworthiness, which involves quadratic pairwise work. Scores are cached and use the existing Euclidean input-space metric. Skipping this evaluation changes no embeddings. For small datasets, the evaluation neighborhood is capped to meet scikit-learn's requirement that `k < n_samples / 2`.
* The first unseen configuration still takes time, including UMAP/Numba startup and fitting. Removing the seed permits more parallel work but changes reproducibility; the app keeps the seed.

Precompute the default projections before opening the app:

```bash
python scripts/precompute_projections.py
```

To also prepare exact quality scores, or prepare another sphere dimension:

```bash
python scripts/precompute_projections.py --quality
python scripts/precompute_projections.py --dimensions 5
```

The script uses the same cache, validation, and shared baseline settings as the app, including when `--dimensions` changes. To precompute explicit alternative settings, pass `--params path/to/params.json` and use the matching settings in the app. **Clear Cache** clears saved projections and scores along with the in-memory data cache; the next render fits again. Cached results use local disk space as you explore new configurations.

## Data

The main app expects precomputed Veltri embeddings and metadata at these paths:

```text
embeddings/veltri/veltri_embeddings.npy
embeddings/veltri/veltri_metadata.csv
```

It can also read the raw Veltri CSV from:

```text
data/veltri/all_veltri.csv
```

If the default embedding files are not present, restore them before using the default dataset, generate them with the scripts in `scripts/`, or upload custom files through the sidebar.

Custom embeddings must be NumPy arrays:

```text
.npy
shape: (samples, embedding_dimension)
```

Custom metadata must be a CSV with exactly the same number of rows, in the same order as the embedding array. Mismatches are rejected rather than truncated. Embeddings must contain finite real numeric values with at least three rows and two columns; object, text, complex, NaN, and infinite values are rejected before fitting. Column detection ignores case and surrounding whitespace, while ambiguous duplicate column names are rejected.

Sequence and ID columns are optional. Missing IDs fall back to source row numbers, and repeated IDs retain distinct row identities. Choose the **AMP status column** in the sidebar if needed. Accepted explicit status values are booleans, `1`/`0`, or `AMP`/`Non-AMP` strings, ignoring case, spaces, hyphens, and underscores. Present status columns with missing or invalid values produce an actionable error. Without a status column, peptides are explicitly unlabeled and grey rather than being classified as non-AMP.

Common metadata columns:

```text
id, sequence, label, is_amp
```

or:

```text
peptide_id, sequence, source, AMP
```

## Example Workflow

1. Load the default Veltri embeddings, or upload custom embeddings and metadata.
2. Choose Veltri metadata or the raw Veltri CSV for labels and hover text.
3. Adjust UMAP or densMAP parameters in the sidebar and click Apply projection settings.
4. Explore the 2D projection and inspect peptides with hover text.
5. Rotate and zoom the ScatterGL globe, switch between **Points / Labels**, and inspect peptides with hover text.
6. Select a region with Plotly's 2D box/lasso tool. The globe automatically displays only those source rows; the selection remains active when switching between 3D–8D.
7. Refine the selection with ScatterGL's **Select region**, export it, or clear it to restore the full globe.

The 3D component includes vendored browser dependencies: ScatterGL 0.0.13 and Three.js 0.125.2 in `utils/globe_frontend/vendor/`, with their upstream licenses in `SCATTER_GL_LICENSE` and `THREE_LICENSE`. They are served locally; running the app requires no CDN connection or npm installation.

Selection JSON exports retain their existing fields and add schema version 2 provenance: dataset fingerprint, source paths, embedding shape/dtype, scientific-library versions, selected source row indices, and actual projection coordinates. The existing parameters include the metric and random seed. Globe exports contain both displayed coordinates and selected coordinates from the original n-dimensional UMAP result. Preserve the original embedding/metadata files and row order alongside an export to reproduce a run. The optimization script seeds both Optuna's sampler and UMAP; the embedding generator validates labels and sequences before loading the model.

## Project Structure

```text
Pepspace-Visualization/
  app.py                         Main Streamlit visualization app
  requirements.txt               Python dependencies
  README.md                      Project documentation
  .streamlit/config.toml          Switchable native light/dark palettes
  LICENSE                        Project license

  data/veltri/                   Raw Veltri dataset
  embeddings/                    Precomputed embeddings and metadata
  scripts/                       Data preparation, embedding, and optimization scripts
  utils/dataset.py               Input validation, labels, and stable source row identities
  utils/projection_settings.py   Shared reproducible app and CLI settings
  utils/embedding_runtime.py     Exact persistent projection and quality caches
  utils/sphere_projection.py     Geometry-preserving display transformations
  utils/globe_view.py            Python-to-ScatterGL data and selection bridge
  utils/globe_frontend/          Native renderer, level grid/camera geometry, local assets
  optuna_results/                UMAP optimization parameter files
  tests/                         Geometry, caching, and selection checks
  .cache/                        Generated local caches (ignored)
```

## Scripts

Useful scripts in this repository:

```text
scripts/make_veltri_embeddings.py    Generate Veltri embedding files
scripts/optimize_umap.py             Run UMAP parameter optimization
scripts/precompute_projections.py    Prepare the app's persistent projection cache
scripts/veltri_dataset.py            Prepare Veltri dataset inputs
scripts/make_fake_csv.py             Create sample fake metadata
scripts/make_real_csv.py             Create sample real metadata
```

## Tests

Run the dataset, settings, geometry, projection-cache, and linked-selection regression checks. Linked-view tests use small temporary datasets and stub projection fitting; they exercise Streamlit reruns across 3D–8D without refitting the full dataset:

```bash
python -m unittest discover -s tests -v
```

An optional frontend geometry check uses Node's built-in assertion library:

```bash
node tests/test_globe_geometry.cjs
```

Node is not needed to run the app.

## Methods

UMAP is used for nonlinear dimensionality reduction of peptide embedding vectors.

densMAP is a density-preserving extension of UMAP that can better maintain relative cluster density.

Protein language model embeddings convert peptide sequences into numerical vectors suitable for projection and visualization.

The default globe display centers and uniformly scales 3D UMAP outputs. An optional surface display uses radial normalization. Higher-dimensional UMAP outputs use PCA for their 3D display.

## Usage With Custom Datasets

Upload your own peptide dataset from the sidebar:

1. Upload embeddings as a `.npy` file.
2. Upload metadata as a `.csv` file.
3. Select sequence, ID, or label columns manually if auto-detection fails.
4. Run UMAP or densMAP visualization.

The metadata row count and order must match the embedding rows. If counts differ, the app stops with a clear error before fitting.

## Author

Alejandro Lopez <br>
Faulty Advisor: Jesus Armando Beltran Verdugo<br>
Department of Computer Science<br>
California State University - Los Angeles
