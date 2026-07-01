# Scan-to-CAD Deviation Analyzer

Quantifies the geometric gap between a designed CAD part and its physical scan (the "as-designed vs. as-built" delta that every manufacturing QA pipeline has to deal with eventually).

## What it does

Upload a reference CAD mesh (STL/OBJ) and an as-built point cloud scan (PLY/XYZ). The tool aligns the scan to the CAD reference with ICP (a from-scratch SVD-based implementation, no Open3D), computes per-point distance to the CAD surface, and renders the result as a green→red deviation heatmap on the mesh, plus max/mean/RMS stats and a distribution histogram. No scan on hand? Flip on "Simulate scan from CAD" and it'll generate one with configurable noise.

A second tab runs the same pipeline across a simulated production batch instead of one part: N parts off the same CAD reference, each with baseline measurement noise, and a tool-wear defect that grows in after a chosen part number. Deviations get bucketed into spatial zones (k-means) and tracked with a Hotelling's T² control chart (the standard multivariate SPC approach for catching a process drifting out of control before any single part fails its tolerance outright).

## Quickstart

```bash
pip install -r requirements.txt
python generate_sample.py   # creates assets/sample_bracket.stl
python app.py                # http://localhost:7860
```

Click **Load Demo Part → Run Analysis** to see it work end to end.

## Layout

- `app.py` (Gradio UI)
- `core/simulation.py` (synthetic scan generation)
- `core/registration.py` (ICP alignment)
- `core/deviation.py` (point-to-mesh distance, heatmap colorization)
- `core/batch_analysis.py` (production-run simulation, Hotelling's T² SPC)
- `generate_sample.py` (builds the demo bracket STL)

Built on `trimesh` for mesh I/O/sampling, `scipy` for the KD-tree queries, k-means, and F-distribution control limits, `gradio` for the UI, `plotly` for the charts.

## Example output

Bracket with 0.8mm of simulated manufacturing noise, ±0.5mm tolerance:

```
52.3% of scan points within ±0.5 mm tolerance
Max: 2.83 mm | Mean: 0.61 mm | RMS: 0.77 mm | Median: 0.51 mm
ICP fitness: 0.977
```
