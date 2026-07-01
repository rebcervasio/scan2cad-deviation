"""
Scan-to-CAD Deviation Analyzer
Gradio web app: upload a CAD mesh + point cloud scan → deviation heatmap + QA report.
"""
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from core.deviation import colorize_mesh_by_deviation, compute_deviation, compute_statistics
from core.registration import align_point_cloud_to_mesh
from core.simulation import simulate_scan

SAMPLE_STL = os.path.join(os.path.dirname(__file__), "assets", "sample_bracket.stl")

DESCRIPTION = """
# 🔬 Scan-to-CAD Deviation Analyzer

**Upload a CAD reference mesh** (STL/OBJ) and an **as-built point cloud scan** (PLY/XYZ).
The tool aligns the scan to the CAD model via ICP and produces a **color-coded deviation heatmap**.

> **No scan?** Enable *Simulate scan from CAD* to generate a synthetic scan with realistic manufacturing noise.

*Colors: 🟢 green = within tolerance · 🟡 yellow = borderline · 🔴 red = out of tolerance*
"""


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh


def load_pcd_as_array(path: str) -> np.ndarray:
    """Load a point cloud file and return (N,3) numpy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        mesh_or_pcd = trimesh.load(path)
        if hasattr(mesh_or_pcd, "vertices"):
            return np.array(mesh_or_pcd.vertices)
        raise ValueError("Could not extract points from PLY file.")
    # XYZ / TXT: whitespace-separated x y z per line
    return np.loadtxt(path, usecols=(0, 1, 2))


def make_histogram(distances: np.ndarray, tolerance: float) -> go.Figure:
    colors = ["green" if d <= tolerance else "red" for d in distances]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=60,
        name="scan points",
        marker_color="steelblue",
        opacity=0.75,
    ))
    fig.add_vline(
        x=tolerance, line_dash="dash", line_color="orange", line_width=2,
        annotation_text=f"tolerance: {tolerance} mm", annotation_position="top right",
    )
    fig.update_layout(
        title="Deviation Distribution",
        xaxis_title="Distance to CAD surface (mm)",
        yaxis_title="Point count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def run_analysis(
    cad_file,
    scan_file,
    use_simulation: bool,
    noise_std: float,
    n_points: int,
    tolerance: float,
    run_icp: bool,
):
    if cad_file is None:
        raise gr.Error("Please upload a CAD mesh (STL or OBJ).")

    mesh = load_mesh(cad_file.name)

    # Scale heuristic: if bounding box diagonal < 1, assume meters → convert to mm
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diag < 1.0:
        mesh.apply_scale(1000)

    if use_simulation or scan_file is None:
        points = simulate_scan(mesh, n_points=int(n_points), noise_std=noise_std)
        icp_info = "Simulated scan (no upload)"
    else:
        points = load_pcd_as_array(scan_file.name)
        icp_info = f"Loaded scan: {len(points)} points"

    if run_icp:
        points, transform, fitness = align_point_cloud_to_mesh(points, mesh)
        icp_info += f" | ICP fitness: {fitness:.3f}"
    else:
        icp_info += " | ICP skipped (assumed pre-aligned)"

    distances = compute_deviation(points, mesh)
    stats = compute_statistics(distances, tolerance)
    stats["icp_status"] = icp_info

    glb_path = colorize_mesh_by_deviation(points, mesh, distances, tolerance)
    fig = make_histogram(distances, tolerance)

    pct = stats["pct_within_tolerance"]
    emoji = "✅" if pct >= 90 else ("⚠️" if pct >= 70 else "❌")
    summary = (
        f"### {emoji} {pct}% of scan points within ±{tolerance} mm tolerance\n\n"
        f"| Max | Mean | RMS | Median |\n"
        f"|-----|------|-----|--------|\n"
        f"| {stats['max_deviation_mm']} mm | {stats['mean_deviation_mm']} mm "
        f"| {stats['rms_deviation_mm']} mm | {stats['median_deviation_mm']} mm |"
    )

    return glb_path, fig, stats, summary


def load_demo(_):
    """Load the bundled sample bracket with moderate noise for demo."""
    if not os.path.exists(SAMPLE_STL):
        raise gr.Error("Sample STL not found. Run `python generate_sample.py` first.")
    # Return values for: cad_file, use_sim, noise_std, n_points, tolerance, run_icp
    return SAMPLE_STL, True, 1.2, 6000, 0.5, True


with gr.Blocks(title="Scan-to-CAD Deviation Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Inputs")
            cad_file = gr.File(
                label="CAD Reference Mesh (STL / OBJ)",
                file_types=[".stl", ".obj"],
            )
            scan_file = gr.File(
                label="Scan Point Cloud (PLY / XYZ) — optional",
                file_types=[".ply", ".xyz", ".txt"],
            )

            gr.Markdown("### 🎲 Scan Simulation")
            use_sim = gr.Checkbox(
                label="Simulate scan from CAD (ignore uploaded scan)", value=True
            )
            noise_std = gr.Slider(
                0.05, 5.0, value=1.2, step=0.05,
                label="Simulated noise std (mm) — mimics manufacturing variation",
            )
            n_points = gr.Slider(
                500, 20000, value=6000, step=500,
                label="Number of scan points",
            )

            gr.Markdown("### ⚙️ Analysis Settings")
            tolerance = gr.Slider(
                0.05, 5.0, value=0.5, step=0.05,
                label="Tolerance threshold (mm) — parts outside this are flagged red",
            )
            run_icp = gr.Checkbox(label="Run ICP alignment (recommended for real scans)", value=True)

            with gr.Row():
                demo_btn = gr.Button("▶ Load Demo Part", variant="secondary")
                run_btn = gr.Button("🔍 Run Analysis", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 🎨 Deviation Heatmap")
            model_view = gr.Model3D(label="Colored CAD Mesh (green=ok, red=deviation)")
            summary_md = gr.Markdown()

    with gr.Row():
        histogram = gr.Plot(label="Deviation Histogram")
        stats_json = gr.JSON(label="Full QA Statistics")

    # Demo button: load sample STL and set parameters
    demo_btn.click(
        fn=load_demo,
        inputs=[demo_btn],
        outputs=[cad_file, use_sim, noise_std, n_points, tolerance, run_icp],
    )

    run_btn.click(
        fn=run_analysis,
        inputs=[cad_file, scan_file, use_sim, noise_std, n_points, tolerance, run_icp],
        outputs=[model_view, histogram, stats_json, summary_md],
    )

if __name__ == "__main__":
    demo.launch(share=False)
