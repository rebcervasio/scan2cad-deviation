# Scan-to-CAD Deviation Analyzer

A manufacturing QA tool that quantifies the geometric deviation between a designed CAD part and its physical scan — enabling automated tolerance verification at scale.

![Demo: deviation heatmap on a mechanical bracket](assets/demo_screenshot.png)

## What it does

Upload a **reference CAD mesh** (STL/OBJ) and an **as-built point cloud scan** (PLY/XYZ). The tool:

1. **Aligns** the scan to the CAD reference via ICP (Iterative Closest Point) — custom SVD-based implementation, no external registration library required
2. **Computes** per-point distance from scan to CAD surface using spatial proximity queries
3. **Colorizes** the CAD mesh as a deviation heatmap: green (within tolerance) → red (out of tolerance)
4. **Reports** key QA statistics: max/mean/RMS deviation, % within tolerance

No scan available? Enable **Simulate scan** to generate realistic manufacturing noise on top of the CAD geometry.

## Why this matters

This bridges the core gap in digital manufacturing: the "design–build" delta. Every manufactured part deviates from its CAD geometry due to machining tolerances, material deformation, and assembly stresses. Quantifying this deviation at scale — from point cloud data — is fundamental to automated QA pipelines.

## Quickstart

```bash
pip install -r requirements.txt
python generate_sample.py   # creates assets/sample_bracket.stl
python app.py               # launches at http://localhost:7860
```

Then click **Load Demo Part → Run Analysis** to see it in action.

## Stack

| Component | Library |
|-----------|---------|
| Mesh I/O + surface sampling | `trimesh` |
| ICP registration | Custom SVD implementation (`numpy`, `scipy`) |
| Point-to-surface distance | `trimesh.proximity` + `scipy.spatial.cKDTree` |
| 3D colored mesh export | `trimesh` → GLB |
| Web UI | `gradio` |
| Deviation histogram | `plotly` |

## Project structure

```
scan2cad-deviation/
├── app.py                  # Gradio web app
├── core/
│   ├── simulation.py       # Synthetic scan generator
│   ├── registration.py     # ICP alignment (SVD-based)
│   └── deviation.py        # Point-to-mesh distance + colorization
├── assets/
│   └── sample_bracket.stl  # Demo mechanical part
├── generate_sample.py      # Generates the demo STL
└── requirements.txt
```

## Input formats

| Type | Formats |
|------|---------|
| CAD mesh | STL, OBJ |
| Point cloud scan | PLY, XYZ (space-separated x y z) |

## Example output

For a bracket with 0.8mm manufacturing noise and ±0.5mm tolerance:

```
✅ 52.3% of scan points within ±0.5 mm tolerance
Max: 2.83 mm | Mean: 0.61 mm | RMS: 0.77 mm | Median: 0.51 mm
ICP fitness: 0.977
```
