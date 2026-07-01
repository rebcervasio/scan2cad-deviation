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
from core.batch_analysis import colorize_mesh_by_mode, pca_drift_analysis, project_tolerance_crossing, simulate_production_run
from core.deviation import colorize_mesh_by_deviation, compute_deviation, compute_statistics
from core.registration import align_point_cloud_to_mesh
from core.simulation import simulate_scan

SAMPLE_STL = os.path.join(os.path.dirname(__file__), "assets", "sample_bracket.stl")

BATCH_DESCRIPTION = """
### 📈 Production Run — Hotelling's T² Drift Detection

Simulates a production run of N parts sharing one CAD reference, each with baseline
measurement noise plus a **localized tool-wear defect that grows after a chosen part
number**. Deviations are aggregated into spatial zones (k-means) and monitored with a
**Hotelling's T² control chart** — the standard multivariate SPC technique (Mahalanobis
distance from the baseline/Phase-I reference, F-distribution control limit) — to detect
*when* the process drifts out of statistical control, ahead of any single part failing
its physical tolerance.
"""

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


def make_t2_chart(t2: np.ndarray, ucl: float, n_baseline: int, first_violation) -> go.Figure:
    parts = np.arange(len(t2))
    fig = go.Figure()
    fig.add_vrect(x0=-0.5, x1=n_baseline - 0.5, fillcolor="lightgreen", opacity=0.15, line_width=0,
                  annotation_text="baseline (Phase I)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=parts, y=t2, mode="lines+markers", name="T² (Hotelling)", marker_color="steelblue"))
    fig.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL (α=0.27%)", annotation_position="top right")
    if first_violation is not None:
        fig.add_vline(x=first_violation, line_dash="dot", line_color="orange",
                       annotation_text=f"first violation: part {first_violation}", annotation_position="bottom right")
    fig.update_layout(
        title="Hotelling's T² Control Chart (multivariate SPC)",
        xaxis_title="Part number in production run",
        yaxis_title="T² statistic",
        yaxis_type="log",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def run_batch_analysis(
    cad_file,
    n_parts: float,
    base_noise_std: float,
    drift_start_part: float,
    drift_rate: float,
    n_baseline: float,
    tolerance: float,
):
    if cad_file is None:
        raise gr.Error("Please upload a CAD mesh (STL or OBJ) above first.")

    n_parts = int(n_parts)
    drift_start_part = int(drift_start_part)
    n_baseline = int(n_baseline)

    mesh = load_mesh(cad_file.name)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if diag < 1.0:
        mesh.apply_scale(1000)

    run = simulate_production_run(
        mesh,
        n_parts=n_parts,
        n_points=1200,
        base_noise_std=base_noise_std,
        drift_start_part=drift_start_part,
        drift_rate=drift_rate,
        seed=0,
    )

    if n_baseline <= run["n_zones"]:
        raise gr.Error(
            f"Baseline parts ({n_baseline}) must exceed the number of monitoring zones "
            f"({run['n_zones']}) for a stable covariance estimate — increase baseline parts."
        )
    if n_baseline >= n_parts - 5:
        raise gr.Error("Leave at least 5 parts after the baseline window for monitoring.")
    if n_baseline > drift_start_part:
        gr.Warning("Baseline window overlaps the simulated drift onset — control limits may be contaminated.")

    result = pca_drift_analysis(run["deviations"], run["zone_ids"], run["n_zones"], n_baseline=n_baseline)
    fig = make_t2_chart(result["t2"], result["ucl"], n_baseline, result["first_violation_part"])
    mode_glb = colorize_mesh_by_mode(mesh, run["canonical_points"], result["point_loading"])

    max_dev_per_part = run["deviations"].max(axis=1)
    fit_from = result["first_violation_part"] if result["first_violation_part"] is not None else drift_start_part
    crossing = project_tolerance_crossing(np.arange(n_parts), max_dev_per_part, tolerance, fit_from)

    if result["first_violation_part"] is None:
        summary = (
            f"### ✅ Process stayed in statistical control for all {n_parts} simulated parts\n\n"
            f"No part exceeded the Hotelling T² control limit (UCL={result['ucl']:.1f}, α=0.27%)."
        )
    else:
        fv = result["first_violation_part"]
        delay = fv - drift_start_part
        summary = (
            f"### ⚠️ Out-of-control signal detected at part #{fv}\n\n"
            f"Hotelling's T² exceeded the control limit ({result['t2'][fv]:.1f} > UCL {result['ucl']:.1f}) "
            f"— {delay} parts after the simulated tool-wear onset (part #{drift_start_part}).\n\n"
        )
        if crossing is not None:
            summary += f"At the current drift trend, the physical ±{tolerance} mm tolerance is projected to be exceeded around **part #{crossing}**."
        else:
            summary += f"The drift trend doesn't yet project a ±{tolerance} mm tolerance crossing within this run."

    diagnostics = {
        "n_zones": run["n_zones"],
        "n_baseline": n_baseline,
        "ucl": round(result["ucl"], 2),
        "baseline_covariance_explained_variance_ratio": [round(v, 4) for v in result["explained_variance_ratio"]],
        "first_violation_part": result["first_violation_part"],
        "projected_tolerance_crossing_part": crossing,
        "max_t2": round(float(result["t2"].max()), 2),
    }

    return fig, mode_glb, summary, diagnostics


with gr.Blocks(title="Scan-to-CAD Deviation Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    gr.Markdown("### 📂 CAD Reference (shared by both tabs below)")
    cad_file = gr.File(
        label="CAD Reference Mesh (STL / OBJ)",
        file_types=[".stl", ".obj"],
    )

    with gr.Tabs():
        with gr.Tab("Single Part Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
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

        with gr.Tab("Production Run — PCA Drift Detection"):
            gr.Markdown(BATCH_DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    batch_n_parts = gr.Slider(20, 100, value=60, step=5, label="Number of parts in production run")
                    batch_n_baseline = gr.Slider(15, 40, value=20, step=1,
                                                  label="Baseline (Phase I) parts used to fit control limits")
                    batch_noise_std = gr.Slider(0.01, 0.3, value=0.05, step=0.01,
                                                 label="Baseline process noise std (mm)")
                    batch_drift_start = gr.Slider(5, 80, value=20, step=1, label="Part # when tool wear begins")
                    batch_drift_rate = gr.Slider(0.005, 0.08, value=0.025, step=0.005,
                                                  label="Wear growth rate (mm/part)")
                    batch_tolerance = gr.Slider(0.1, 2.0, value=0.5, step=0.05,
                                                 label="Physical tolerance (mm) for crossing projection")
                    run_batch_btn = gr.Button("📈 Run Production Simulation", variant="primary")

                with gr.Column(scale=2):
                    control_chart = gr.Plot(label="Hotelling's T² Control Chart")
                    batch_summary_md = gr.Markdown()

            with gr.Row():
                mode_view = gr.Model3D(label="Zones driving the signal (red = growing, blue = shrinking, for the flagged part)")
                batch_json = gr.JSON(label="SPC Diagnostics")

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

    run_batch_btn.click(
        fn=run_batch_analysis,
        inputs=[cad_file, batch_n_parts, batch_noise_std, batch_drift_start, batch_drift_rate, batch_n_baseline, batch_tolerance],
        outputs=[control_chart, mode_view, batch_summary_md, batch_json],
    )

if __name__ == "__main__":
    demo.launch(share=False)
