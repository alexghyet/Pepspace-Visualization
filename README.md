# PepSpace Visualization

PepSpace is an interactive Streamlit application for exploring bioactive peptide embedding spaces. It visualizes peptide datasets with UMAP and densMAP, supports default Veltri data paths, and allows users to upload custom embedding and metadata files.

## Features

* Interactive 2D UMAP and densMAP projections
* Consistent class colors in every dimension: AMP blue, non-AMP red, unlabeled grey
* Hover inspection for peptide IDs, sequences, and labels
* Upload existing embeddings as NPY, NPZ, CSV, TSV, or Parquet and optionally match metadata by row order or identifier
* Upload peptide sequences and generate compatible PeptideBERT embeddings on CPU
* Place new peptides in a saved Veltri UMAP map with independent reference/new layer visibility
* Reproducible holdout demos on Veltri or uploaded data, with reference-only fitting
* Activity/dataset coloring, optional numeric bubble sizes in 2D, and an uploaded PDB inspector
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

The main app loads PeptideBERT v2.0 solubility embeddings for Veltri by default (3,556 peptides, 256 dimensions). The files are:

```text
embeddings/veltri_peptidebert_sol_v2/veltri_embeddings.npy
embeddings/veltri_peptidebert_sol_v2/veltri_metadata.csv
```

It can also read the raw Veltri CSV from:

```text
data/veltri/all_veltri.csv
```

If the default embedding files are not present, restore them before using the default dataset, generate them with the scripts in `scripts/`, or upload custom files through the sidebar.

Custom embeddings can be a 2D `.npy` array, a matrix selected from a `.npz` archive, or a CSV, TSV, or Parquet table with one numeric column per dimension. Each row represents one point. Fitting an independent projection requires at least three rows and two dimensions containing finite real numbers. A Veltri overlay can accept a single query row.

Metadata is optional. For tables, columns outside the selected dimensions are available as metadata. You can also attach a CSV, TSV, or Parquet file. Matching by row order requires one metadata row per embedding in the same order. Matching by identifier accepts shuffled rows and different column names. Both files must contain the same unique, nonblank keys. Matching trims surrounding whitespace, respects case, and preserves leading zeros in text IDs.

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

Select one of three modes in **Choose workflow**:

| Mode | What it does |
| --- | --- |
| Explore a dataset | Fits UMAP or densMAP to Veltri or an uploaded dataset independently. Other encoders and synthetic vectors are supported here. |
| Project onto Veltri | Fits UMAP to Veltri and transforms new data into that reference map. |
| Holdout demo | Splits Veltri or an uploaded dataset, fits UMAP only on reference rows, and transforms the held-out rows. |

In either upload workflow, choose **Peptide sequences (.csv)** or **Existing embeddings**.
Sequences are embedded with the same official PeptideBERT v2.0 solubility weights,
residue encoding, and masked-mean pooling as the default reference. Select the
sequence column under **Metadata columns**, then click **Generate embeddings**.
The local checkpoint and matching configuration must be present in
`.cache/peptidebert/v2.0/sol/`. Inference runs on CPU and is cached. AMP labels are
optional; the encoder does not produce AMP predictions. Download the generated
embedding/metadata/manifest ZIP to reuse the results.

Existing vectors from any model can be explored independently. Veltri overlays
additionally require the matching embedding `manifest.json`.
The app compares the checkpoint hash, model, task, encoder configuration, pooling,
encoding, truncation, and feature count. If the manifest supplies an embeddings
hash, the app checks the original NPY bytes or the selected NPY member inside
an NPZ archive. For tables, it checks the selected matrix saved as a contiguous
float32 NPY in file column order.

The default holdout demo holds out 30 peptides with seed 42. Exact duplicate
sequences remain together, so the actual holdout can be slightly larger. If no
sequences are supplied, the split is by row. The reference needs at least three
rows. The split fits UMAP and never retrains PeptideBERT; labels do not enter UMAP.

**Show reference data** and **Show new data** independently control both plots,
including hiding both. Reference points start faint and new points opaque; each
layer has its own opacity slider. These controls never refit the map, recenter
coordinates, or clear selected source rows. In overlay workflows the globe shows
all visible layers by default. Enable **Filter globe to selection** to restrict it
to selected rows. A selected row can stay selected while its layer is hidden.
Selections and exports use dataset identity and source-row indices, even when
peptide IDs repeat.

**Color by** supports AMP status, label, activity, or dataset in both views.
Optional `activity`, `activity_type`, and `pdb_id`/`structure_id` columns appear in
hover details and exports. **2D bubble size** uses a numeric metadata column;
missing, negative, or nonfinite values get the minimum size. Example CSV:

```csv
id,sequence,activity,measurement,pdb_id
example-1,KWKLFKKIGAVLKVL,antibacterial,2.5,
```

The optional **Inspect a molecular structure (PDB)** panel shows alpha-carbon
positions from the first model in a supplied PDB file, colored by chain. It does
not predict structures or infer a link to an embedding point. Molecular positions
are distinct from UMAP/globe coordinates.

### Saved reference maps

Reference fitting and query transformations have separate persistent caches in
`.cache/projections/`, keyed by inputs, parameters, and library versions. Changing
queries reuses the reference model. The original reference coordinates stay fixed;
for the globe, PCA, centering, and scale are fitted on reference coordinates only.
New points may lie outside the reference globe. Plot bounds include all loaded
points so hiding a layer does not change the frame.

**Prepare fitted map download** packages the 2D and selected n-dimensional UMAP
reducers, reference coordinates, reference identities, display transformation,
and provenance into a `.joblib` file. The app reuses its own local cached models
automatically; it does not accept uploaded pickle/joblib files. A trusted exported
bundle can be reused in Python with the same library versions:

```python
import joblib
import numpy as np

bundle = joblib.load("pepspace_reference.joblib")  # Only load trusted files.
new_vectors = np.load("embeddings.npy", allow_pickle=False)
new_2d = bundle["models"][2]["reducer"].transform(new_vectors)
```

This requires the same embedding method as the reference. densMAP does not
support transforming unseen data, so reference overlays and holdout demos use
standard UMAP. **Download all projected rows (.csv)** exports reference and query
coordinates with their identities, independently of current visibility. Selection
JSON exports also include the split, embedding provenance, and display transform.

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
  utils/project_paths.py         Shared PeptideBERT and Veltri default paths
  utils/peptidebert.py           Shared checkpoint loading and sequence inference
  utils/workflow.py              Dataset identity, provenance, and holdout splits
  utils/workflow_ui.py           Uploads and workflow controls
  utils/reference_projection.py  Saved UMAP reducers and reference display transforms
  utils/structure_view.py        Optional PDB alpha-carbon viewer
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

### PeptideBERT embeddings

The default Veltri vectors use the official **PeptideBERT v2.0 solubility** checkpoint
and padding-masked mean pooling. To regenerate them, use the command below.
A repository clone does not include trained weights. Official weights are available
in the [authors' GitHub releases](https://github.com/ChakradharG/PeptideBERT/releases/tag/v2.0).
The v2.0 `checkpoints.tar` archive includes `sol`, `hemo`, and `nf` checkpoints
with matching configurations. The extractor supports both the original and v2
configuration/checkpoint layouts. On this machine the downloaded v2 files are
under `.cache/peptidebert/v2.0/` (ignored by Git).

```powershell
.\.venv-x64\Scripts\python.exe scripts/make_peptidebert_embeddings.py --checkpoint "C:\path\to\checkpoint\model.pt"
```

For the downloaded solubility checkpoint:

```powershell
.\.venv-x64\Scripts\python.exe scripts/make_peptidebert_embeddings.py --checkpoint .cache/peptidebert/v2.0/sol/model.pt --output-dir embeddings/veltri_peptidebert_sol_v2 --device cpu --batch-size 16
```

This writes `veltri_embeddings.npy`, `veltri_metadata.csv`, and `manifest.json`
under `embeddings/veltri_peptidebert_sol_v2/`, the same paths loaded by the app,
projection precomputation, and optimization scripts by default. Existing outputs
are never overwritten; choose a new `--output-dir` to regenerate. Use
`--batch-size` to control memory use.

The extractor matches this repository's custom residue vocabulary, without
inserting CLS/SEP tokens. It uses a padding-masked mean of final encoder residue
states, not classifier probabilities. This is an explicit embedding convention,
not a reproduction of the paper's CLS visualization. It rejects unsupported
residues and overlength sequences, strictly loads all encoder weights without
downloading or substituting base ProtBERT weights, and records checkpoint and
dataset hashes. The checkpoint task (e.g. solubility) describes its training;
Veltri AMP labels remain metadata, not model predictions.

Useful scripts in this repository:

```text
scripts/make_peptidebert_embeddings.py Generate Veltri embeddings with PeptideBERT
scripts/optimize_umap.py             Run UMAP parameter optimization
scripts/precompute_projections.py    Prepare the app's persistent projection cache
scripts/veltri_dataset.py            Prepare Veltri dataset inputs
scripts/make_fake_csv.py             Create synthetic embeddings and metadata
scripts/make_real_csv.py             Export the legacy solubility dataset to CSV
```

## Tests

Run the dataset, settings, geometry, projection-cache, and linked-selection regression checks. Linked-view tests use small temporary datasets and stub projection fitting; they exercise Streamlit reruns across 3D–8D without refitting the full dataset:

```bash
python -m unittest discover -s tests -v
```

An optional frontend geometry check uses Node's built-in assertion library:

```bash
node tests/test_globe_geometry.cjs
node tests/test_globe_layers.cjs
```

Node is not needed to run the app.

## Methods

UMAP is used for nonlinear dimensionality reduction of peptide embedding vectors.

densMAP is a density-preserving extension of UMAP that can better maintain relative cluster density.

Protein language model embeddings convert peptide sequences into numerical vectors suitable for projection and visualization.

The default globe display centers and uniformly scales 3D UMAP outputs. An optional surface display uses radial normalization. Higher-dimensional UMAP outputs use PCA for their 3D display.

## Usage With Custom Datasets

1. Choose **Explore a dataset → Upload my data → Existing embeddings** and upload your file.
2. For a table, preview the rows and select the **Embedding columns**. Names such as `emb_1` and `dim_2` are suggested automatically. The app uses file column order. For a wide matrix, **Select columns to leave out instead** lets you choose the metadata columns. For a headerless CSV or TSV, clear the first-row checkbox. For an NPZ, choose the matrix and optionally a 1D identifier array.
3. If you have a separate metadata file, attach it and choose how to match rows. With identifiers in both files, choose **Identifier** and select the key on each side. A plain NPY matrix uses row order. The preview follows embedding order. If a column name appears in both files, the attached column gets an `attached.` prefix.
4. Under **Metadata columns**, choose any column for ID, label, sequence, or activity. Leave the ID unset to use zero-based row numbers. Review the point preview and click **Project dataset**.

To try a single-file upload, use [sample_embeddings.csv](data/examples/sample_embeddings.csv), also available from **Download example embeddings** in the sidebar. It contains 72 synthetic points with eight dimensions. Use `emb_1` through `emb_8` as the dimensions, `specimen_no` as the ID, and `cohort` as the label. Both plots use the three cohort colors. The numeric `batch_number` is available for bubble size.

## Author

Alejandro Lopez <br>
Faulty Advisor: Jesus Armando Beltran Verdugo<br>
Department of Computer Science<br>
California State University - Los Angeles
