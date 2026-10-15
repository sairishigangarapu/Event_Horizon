"""Twin Studio evaluation page (Phase 7): select run -> observations -> forecasts -> explicit reveal
of labels already released by the reference service -> metrics.

Reads only evaluator output files (services.twin_studio.report_view). It holds no reference
credentials and makes no call to the reference API or object storage.
"""
import os

import pandas as pd
import streamlit as st

from services.twin_studio.report_view import (UNAVAILABLE_NOTES, list_evaluations, load_evaluation, metric_rows,
                                              model_warnings, observation_rows, revealed_label_rows,
                                              forecast_rows)

st.set_page_config(page_title="Twin Studio Evaluation", layout="wide")
st.title("Twin Studio evaluation (eh_twin_studio)")

root = os.environ.get("EH_TWIN_EVALUATIONS_DIR", "data/twin_evaluations")
evaluations = list_evaluations(root)
if not evaluations:
    st.info(f"No evaluator reports found under `{root}`. Run `python -m services.twin_studio.evaluation` first.")
    st.stop()

labels = [f"{e['run_id']} · {e['scenario_id']} · {e['evaluation']}" for e in evaluations]
choice = st.sidebar.selectbox("Published run / evaluation", range(len(evaluations)), format_func=lambda i: labels[i])
data = load_evaluation(evaluations[choice]["path"])
report, forecasts = data["report"], data["forecasts"]

for w in model_warnings(report):
    st.warning(w)
with st.expander("Unavailable or unsupported fields", expanded=True):
    for note in UNAVAILABLE_NOTES:
        st.markdown(f"- {note}")

st.subheader("1. Observations (as received; timestamps shown)")
obs = pd.DataFrame(observation_rows(report))
if obs.empty:
    st.write("No observations were visible to the evaluator.")
else:
    zone = st.selectbox("Zone", ["(all)"] + sorted(obs["zone_id"].unique()))
    st.dataframe(obs if zone == "(all)" else obs[obs["zone_id"] == zone], use_container_width=True)

st.subheader("2. Forecasts (persisted before their labels became eligible)")
streams = sorted(report["streams"])
stream = st.radio("Stream", streams, horizontal=True)
horizon = st.selectbox("Horizon (s)", report["horizons_seconds"])
fc = pd.DataFrame(forecast_rows(forecasts, stream=stream, horizon=horizon))
st.dataframe(fc, use_container_width=True)

st.subheader("3. Reference labels")
st.caption("Labels below were released by the reference service's server-side watermark "
           "(label_available_at = target_at + label delay). This page cannot request labels itself.")
reveal_key = f"reveal::{evaluations[choice]['path']}"
if st.button("Reveal eligible labels"):
    st.session_state[reveal_key] = True
if st.session_state.get(reveal_key):
    st.dataframe(pd.DataFrame(revealed_label_rows(report, stream=stream, horizon=horizon)), use_container_width=True)

    st.subheader("4. Metrics per variable and horizon")
    st.caption("Computed only from released, applicable labels; missing labels are counted, not imputed.")
    st.dataframe(pd.DataFrame(metric_rows(report)), use_container_width=True)
else:
    st.info("Labels hidden. Press **Reveal eligible labels** to show them.")
