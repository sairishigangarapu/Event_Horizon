import os
import html
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# Import project modules
from crowd_reliability.config import settings
from crowd_reliability.database import SessionLocal, init_db
from crowd_reliability.models.schema import (
    ZoneRecord, ConnectionRecord, ObservationRecord,
    TreeOutputRecord, ForecastRecord, ReferenceStateRecord,
    ModelManifestRecord
)
from crowd_reliability.simulation.scenarios import SCENARIO_PRESETS, get_default_venue
from crowd_reliability.services.comparator import ForecastComparator
from crowd_reliability.services.evaluator import EvaluationService

st.set_page_config(
    page_title="Crowd Disaster Information Reliability",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .badge-stream-a {
        background-color: #ff7f0e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .badge-stream-b {
        background-color: #1f77b4;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_comparator_and_evaluator():
    return ForecastComparator(), EvaluationService()

comparator, evaluator = get_comparator_and_evaluator()

def load_data(scenario_id: str):
    db = SessionLocal()
    try:
        zones = db.query(ZoneRecord).all()
        connections = db.query(ConnectionRecord).all()
        forecasts = db.query(ForecastRecord).filter(ForecastRecord.scenario_id == scenario_id).all()
        reference_states = db.query(ReferenceStateRecord).filter(ReferenceStateRecord.scenario_id == scenario_id).all()
        observations = db.query(ObservationRecord).filter(ObservationRecord.scenario_id == scenario_id).all()
        manifest = db.query(ModelManifestRecord).filter(ModelManifestRecord.is_active == True).first()
        return zones, connections, forecasts, reference_states, observations, manifest
    finally:
        db.close()

# ----------------- Sidebar -----------------
st.sidebar.title("🚨 Crowd Reliability")
st.sidebar.caption("Dual-Stream Information & Forecast Evaluator")

# Scenario Selector
scenarios = list(SCENARIO_PRESETS.keys())
selected_scenario = st.sidebar.selectbox(
    "Evaluation Scenario",
    scenarios,
    index=0,
    format_func=lambda x: f"{SCENARIO_PRESETS[x].title} ({x})"
)
sc_info = SCENARIO_PRESETS[selected_scenario]
st.sidebar.info(f"**Description:** {sc_info.description}\n\n**Duration:** {sc_info.duration_seconds}s | **Step:** {sc_info.step_seconds}s")

# Model readiness: never present untrained neural forecasts as validated.
from crowd_reliability.ml.pipeline import ModelArtifactError, load_manifest
from crowd_reliability.models.manifest import assess_readiness
try:
    _readiness = assess_readiness(load_manifest(settings.model_dir))
    if not _readiness["production_ready"]:
        st.warning(f"**Experimental Stream B.** {_readiness['notice']}")
except ModelArtifactError as _exc:
    st.error(f"Model artifacts unavailable: {_exc}")

# Load DB data (create the tables first so a fresh database shows the seed button, not an error)
init_db()
zones, connections, forecasts, reference_states, observations, manifest = load_data(selected_scenario)

if not forecasts or not reference_states:
    st.warning("⚠️ No simulation data found in database. Please seed the demo first.")
    if st.button("🌱 Seed Demo Data Now"):
        with st.spinner("Seeding demo database from the committed models (no training)..."):
            from crowd_reliability.cli import cmd_seed_serving
            cmd_seed_serving(seed=42)
            st.success("Demo seeded successfully! Please refresh.")
            st.rerun()
    st.stop()

# Zone Selector
zone_ids = [z.zone_id for z in zones]
zone_names = {z.zone_id: f"{z.name} ({z.zone_type})" for z in zones}
selected_zone = st.sidebar.selectbox("Select Zone", zone_ids, format_func=lambda z: zone_names.get(z, z))

# Variable Selector (Channel-specific, non-collapsed)
VARIABLES = [
    ("density", "Crowd Density", "p/m²"),
    ("speed", "Walking Speed", "m/s"),
    ("inflow", "Inflow Rate", "p/s"),
    ("outflow", "Outflow Rate", "p/s"),
    ("queue_length", "Queue Length", "persons"),
    ("exit_open", "Exit Availability", "probability"),
]
selected_var_tuple = st.sidebar.selectbox("Channel Variable", VARIABLES, format_func=lambda x: f"{x[1]} ({x[2]})")
var_key, var_title, var_unit = selected_var_tuple

# Ground Truth Toggle
reveal_truth = st.sidebar.checkbox("Reveal Reference Outcomes (Ground Truth)", value=True)

# Main Title & Badges
st.title("Crowd Disaster Information Reliability")
st.markdown("""
Assessment of report freshness, sensor conflicts, and dual-stream forecast divergence during simulated crowd emergencies.
Density, speed, inflow, outflow, queue length and exit availability remain **strictly separate channels**.
""")


# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

zone_obs = [o for o in observations if o.zone_id == selected_zone and o.variable == var_key]
latest_obs_time = max([o.observed_at for o in zone_obs]) if zone_obs else None
latest_eval_time = max([o.received_at for o in observations]) if observations else datetime.now(timezone.utc)

assessment = comparator.assess_information(observations, latest_eval_time, selected_zone, var_key)

with col1:
    st.metric("Zone Type", next((z.zone_type.capitalize() for z in zones if z.zone_id == selected_zone), "Unknown"))
with col2:
    age = assessment.observation_age_seconds
    st.metric("Observation Age", f"{age:.1f}s", delta="-Stale" if assessment.per_variable_flags.get("stale") else "Fresh", delta_color="inverse")
with col3:
    conflicts = len(assessment.conflict_ids or [])
    st.metric("Report Conflicts", f"{conflicts} detected", delta="Conflict Alert" if conflicts > 0 else "Normal", delta_color="inverse")
with col4:
    st.metric("Evidence Coverage", f"{assessment.coverage * 100:.1f}%")

# Main Content Tabs
tab_forecast, tab_evidence, tab_venue, tab_eval, tab_manifest = st.tabs([
    "📈 Dual-Stream Forecast",
    "🛡️ Evidence & Conflicts",
    "🗺️ Venue Layout & Flow",
    "📊 Model Evaluation",
    "📋 Pipeline Manifest"
])

# ----------------- Tab 1: Dual-Stream Forecast -----------------
with tab_forecast:
    st.subheader(f"Dual Forecast Stream: {var_title} in {zone_names.get(selected_zone, selected_zone)}")

    # Filter forecasts for this zone and variable
    arch_f = [f for f in forecasts if f.stream_id == "stream_b_architecture" and f.zone_id == selected_zone and f.variable == var_key]
    cent_f = [f for f in forecasts if f.stream_id == "stream_a_central" and f.zone_id == selected_zone and f.variable == var_key]
    ref_f = [r for r in reference_states if r.zone_id == selected_zone and r.variable == var_key]

    fig = go.Figure()

    # Stream A: Central
    if cent_f:
        df_cent = pd.DataFrame([{"target_at": f.target_at, "value": f.value} for f in cent_f]).sort_values("target_at")
        fig.add_trace(go.Scatter(
            x=df_cent["target_at"], y=df_cent["value"],
            mode="lines+markers",
            name="Stream A: Central Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(symbol="triangle-up", size=6)
        ))

    # Stream B: Architecture
    if arch_f:
        df_arch = pd.DataFrame([{"target_at": f.target_at, "value": f.value} for f in arch_f]).sort_values("target_at")
        fig.add_trace(go.Scatter(
            x=df_arch["target_at"], y=df_arch["value"],
            mode="lines+markers",
            name="Stream B: Architecture Pipeline",
            line=dict(color="#1f77b4", width=3),
            marker=dict(symbol="circle", size=6)
        ))

    # Reference Ground Truth
    if reveal_truth and ref_f:
        df_ref = pd.DataFrame([{"state_at": r.state_at, "value": r.value} for r in ref_f]).sort_values("state_at")
        fig.add_trace(go.Scatter(
            x=df_ref["state_at"], y=df_ref["value"],
            mode="lines",
            name="Reference State (Ground Truth)",
            line=dict(color="#2ca02c", width=2, dash="dot")
        ))

    fig.update_layout(
        title=f"{var_title} ({var_unit}) vs Target Time",
        xaxis_title="Target Timestamp",
        yaxis_title=f"{var_title} ({var_unit})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Difference Analysis Table
    st.subheader("Forecast Divergence & Lead-Time Analysis")
    diffs = comparator.align_and_compare(arch_f, cent_f)
    if diffs:
        diff_rows = []
        for d in diffs[-15:]:  # show recent 15
            diff_rows.append({
                "Target Time": d.target_at.strftime("%H:%M:%S"),
                "Stream A (Central)": f"{d.cent_forecast.value:.3f}" if d.cent_forecast else "N/A",
                "Stream B (Architecture)": f"{d.arch_forecast.value:.3f}" if d.arch_forecast else "N/A",
                "Signed Diff (B - A)": f"{d.signed_diff:+.3f}" if d.signed_diff is not None else "N/A",
                "Abs Disagreement": f"{d.abs_diff:.3f}" if d.abs_diff is not None else "N/A",
                "Review Flag": "⚠️ High Disagreement" if d.abs_diff and d.abs_diff > 0.5 else "✅ Aligned"
            })
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

# ----------------- Tab 2: Evidence & Conflicts -----------------
with tab_evidence:
    st.subheader("Report Freshness & Sensor Disagreement Audit")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("#### Recent Observations & Latency")
        zone_all_obs = [o for o in observations if o.zone_id == selected_zone][-20:]
        if zone_all_obs:
            obs_table = []
            for o in zone_all_obs:
                obs_table.append({
                    "Report ID": o.report_id,
                    "Variable": o.variable,
                    "Source": o.source_id,
                    "Reported Value": f"{o.value} {o.unit}",
                    "Observed Time": o.observed_at.strftime("%H:%M:%S"),
                    "Received Time": o.received_at.strftime("%H:%M:%S"),
                    "Latency": f"{(o.received_at - o.observed_at).total_seconds():.1f}s"
                })
            st.dataframe(pd.DataFrame(obs_table), use_container_width=True)

    with col_e2:
        st.markdown("#### Information Reliability Indicators")
        if assessment.conflict_ids:
            st.error(f"🚨 **Active Sensor Conflict Detected!** {len(assessment.conflict_ids)} contradictory reports identified in zone '{selected_zone}'.")
            for cid in assessment.conflict_ids:
                st.write(f"- Discrepant Report ID: `{cid}`")
        else:
            st.success("✅ No contradictory sensor reports detected in the current observation window.")

        if assessment.per_variable_flags.get("stale"):
            st.warning(f"⚠️ **Stale Data Warning:** Observation age ({assessment.observation_age_seconds:.1f}s) exceeds freshness threshold (45.0s).")
        else:
            st.info(f"ℹ️ **Observation Freshness:** Data is fresh ({assessment.observation_age_seconds:.1f}s old).")

# ----------------- Tab 3: Venue Layout & Flow -----------------
with tab_venue:
    st.subheader("Pedestrian Spatial Network Graph")
    st.markdown("Physical pedestrian connections between entrances, corridors, concourses, and exits.")

    # Build Plotly network graph
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for z in zones:
        geom = z.geometry or {}
        gx = geom.get("x", np.random.randint(50, 400))
        gy = geom.get("y", np.random.randint(50, 400))
        node_x.append(gx)
        node_y.append(gy)
        esc_name = html.escape(str(z.name))
        esc_id = html.escape(str(z.zone_id))
        esc_type = html.escape(str(z.zone_type))
        node_text.append(f"<b>{esc_name}</b><br>ID: {esc_id}<br>Type: {esc_type}<br>Capacity: {z.capacity_definition} p")

        # Color based on zone type
        colors = {"entrance": "#2ca02c", "corridor": "#1f77b4", "assembly": "#9467bd", "stairway": "#8c564b", "exit": "#d62728"}
        node_colors.append(colors.get(z.zone_type, "#7f7f7f"))

    edge_x = []
    edge_y = []
    zone_pos = {z.zone_id: ((z.geometry or {}).get("x", 0), (z.geometry or {}).get("y", 0)) for z in zones}

    for c in connections:
        if c.from_zone in zone_pos and c.to_zone in zone_pos:
            x0, y0 = zone_pos[c.from_zone]
            x1, y1 = zone_pos[c.to_zone]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig_net = go.Figure()

    # Add edges
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=3, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[z.zone_id for z in zones],
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=28,
            line_width=2,
            line=dict(color='white')
        )
    ))

    fig_net.update_layout(
        title="Venue Layout Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig_net, use_container_width=True)

# ----------------- Tab 4: Model Evaluation -----------------
with tab_eval:
    st.subheader("Reference-Based Evaluation Benchmark")
    st.markdown("Head-to-head comparison between **Stream A (Central Source)** and **Stream B (Architecture Pipeline)** against Ground Truth labels.")

    metrics = evaluator.score_predictions(forecasts, reference_states)

    col_m1, col_m2 = st.columns(2)

    for stream_id, col, label in [
        ("stream_a_central", col_m1, "Stream A: Central Baseline"),
        ("stream_b_architecture", col_m2, "Stream B: Architecture Forecaster")
    ]:
        with col:
            st.markdown(f"#### {label}")
            stream_metrics = metrics.get(stream_id, {})
            if stream_metrics:
                rows = []
                for h, v_dict in sorted(stream_metrics.items()):
                    for v_name, m in sorted(v_dict.items()):
                        r = {
                            "Horizon": f"{h}s",
                            "Variable": v_name,
                            "MAE": f"{m.get('mae', 0):.4f}",
                            "RMSE": f"{m.get('rmse', 0):.4f}",
                        }
                        if "accuracy" in m:
                            r["Accuracy"] = f"{m['accuracy'] * 100:.1f}%"
                        rows.append(r)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("No metrics available for this stream.")

# ----------------- Tab 5: Pipeline Manifest -----------------
with tab_manifest:
    st.subheader("Active Model Manifest & Pipeline Versioning")
    if manifest:
        man_dict = {
            "Pipeline Version": manifest.pipeline_version,
            "Decision Tree Bank Version": manifest.tree_version,
            "Graph Hypernetwork Version": manifest.hypernet_version,
            "Spatial GNN Version": manifest.gnn_version,
            "Multivariate Forecaster Version": manifest.temporal_version,
            "Scaler Version": manifest.scaler_version,
            "Schema Version": manifest.schema_version,
            "Scenario Split": manifest.scenario_split,
            "Training Cutoff": manifest.training_cutoff.strftime("%Y-%m-%d %H:%M:%S UTC") if manifest.training_cutoff else "N/A"
        }
        st.json(man_dict)
    else:
        st.json(PipelineManifest().model_dump())
