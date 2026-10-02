import pandas as pd
import streamlit as st

from analytics import charts
from core import config, db
from ml import anomaly, risk_model
from ui import session as ui
from ui import widgets as w

actor = ui.require_admin()
st.title("AI/ML insights", anchor=False)

t_risk, t_anom, t_model = st.tabs(["Risk list", "Anomaly review queue", "Model panel"])

with t_risk:
    metrics = risk_model.latest_metrics()
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Recompute predictions", type="primary", icon=":material/refresh:"):
            with st.spinner("Scoring every student x subject..."):
                n = ui.guarded(risk_model.refresh_predictions, actor, rerun=False)
            if n is not None:
                ui.flash(f"Generated {n} predictions.")
            st.rerun()
        st.caption(f"Model: {metrics['version']}" if metrics else
                   "No trained model yet - all predictions use the rule-based estimate. Train one in the Model panel.")
    risk = risk_model.stored_predictions(actor)
    if risk.empty:
        w.empty_state("No predictions stored yet. Press Recompute predictions.")
    else:
        lv = st.pills("Risk level", ["High", "Medium", "Low"], selection_mode="multi", default=["High", "Medium"])
        subj = st.selectbox("Subject", ["All"] + sorted(risk["code"].unique()), key="risk_subj", width=220)
        view = risk[risk["level"].isin(lv or ["High", "Medium", "Low"])]
        if subj != "All":
            view = view[view["code"] == subj]
        view = view.assign(factors=view["top_factors"].map(w.factors_text))
        st.caption(f"Generated {risk['generated_at'].iloc[0]} · Low < {config.get('ml.risk_low')} ≤ Medium ≤ "
                   f"{config.get('ml.risk_high')} < High. 'rule' = fewer than "
                   f"{config.get('ml.min_sessions_for_model')} sessions, model not used.")
        st.dataframe(view[["roll_no", "full_name", "code", "probability", "level", "method", "factors"]],
                     hide_index=True, column_config={
                         "probability": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1,
                                                                        format="%.2f"),
                         "level": "Level", "method": "Method", "factors": st.column_config.TextColumn(
                             "Top contributing factors", width="large"),
                         "roll_no": "Roll no", "full_name": "Name", "code": "Subject"})

with t_anom:
    last = anomaly.last_run()
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Run anomaly detection", type="primary", icon=":material/search_insights:"):
            with st.spinner("Running Isolation Forest..."):
                res = anomaly.run_detection(actor)
            ui.flash(res["message"], "success" if res["ran"] else "warning")
            st.rerun()
        st.caption(f"Contamination {config.get('ml.anomaly_contamination')} · minimum "
                   f"{config.get('ml.anomaly_min_records')} kiosk records · "
                   + (f"last run {last['at']}: {last['message']}" if last else "not run yet"))
    st.caption("A flag never changes attendance. Confirm it to keep a record of a suspected proxy/spoof, "
               "then correct the attendance manually if needed.")
    status = st.segmented_control("Show", ["pending", "confirmed", "dismissed"], default="pending",
                                  format_func=str.title)
    flags = anomaly.list_flags(actor, status)
    if flags.empty:
        w.empty_state(f"No {status} flags.")
    for fl in flags.itertuples():
        with st.container(border=True):
            st.markdown(f"**{fl.full_name}** ({fl.roll_no}) · {fl.code} · {fl.date} · checked in {fl.check_in_time} "
                        f"· marked {w.status_label(fl.attendance_status)}")
            st.caption(f"Anomaly score {fl.score:.3f} · match distance {fl.match_distance:.3f} · liveness "
                       f"{fl.liveness_score:.2f} · failed attempts {fl.failed_attempts}")
            for t in fl.top_features:
                st.markdown(f"- {t['label']}: **{t['value']}** (typical {t['typical']})")
            if status == "pending":
                note = st.text_input("Admin notes", key=f"an_{fl.id}")
                with st.container(horizontal=True):
                    if st.button("Confirm", key=f"ac_{fl.id}", icon=":material/flag:"):
                        ui.guarded(anomaly.review_flag, actor, fl.id, "confirmed", note, success="Flag confirmed.")
                    if st.button("Dismiss", key=f"ad_{fl.id}", icon=":material/done:"):
                        ui.guarded(anomaly.review_flag, actor, fl.id, "dismissed", note, success="Flag dismissed.")
            else:
                st.caption(f"Reviewed by {fl.reviewer} at {fl.reviewed_at}" + (f" - {fl.notes}" if fl.notes else ""))

with t_model:
    metrics = risk_model.latest_metrics()
    st.markdown("**Risk model** - Random Forest vs Logistic Regression baseline, trained on simulated semesters "
                "(cold start) plus any completed real semesters, evaluated on held-out students.")
    if st.button("Retrain models", type="primary", icon=":material/model_training:"):
        with st.spinner("Generating synthetic semesters and training (about a minute)..."):
            m = ui.guarded(risk_model.train, actor, rerun=False)
            if m:
                risk_model.refresh_predictions(actor)
                res = anomaly.run_detection(actor)
                ui.flash(f"Trained {m['version']} (RF AUC {m['random_forest']['roc_auc']:.3f}). "
                         f"Predictions refreshed. {res['message']}")
        st.rerun()
    if not metrics:
        w.empty_state("No trained risk model yet.")
    else:
        st.caption(f"Version **{metrics['version']}** · trained {metrics['trained_at']} · "
                   f"{metrics['train_samples']} train / {metrics['test_samples']} test samples · "
                   f"{metrics['synthetic_samples']} synthetic, {metrics['real_samples']} real · {metrics['split']}")
        rows = []
        for key, name in [("random_forest", "Random Forest"), ("logistic_regression", "Logistic Regression (baseline)")]:
            m = metrics[key]
            rows.append({"Model": name, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                         "ROC-AUC": m["roc_auc"], "Accuracy": m["accuracy"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.roc_chart(metrics))
        with c2:
            st.plotly_chart(charts.importance_bar(metrics["feature_importance"]))
        cm = metrics["random_forest"]["confusion_matrix"]
        st.caption(f"Random Forest confusion matrix (threshold 0.5): TN {cm[0][0]}, FP {cm[0][1]}, "
                   f"FN {cm[1][0]}, TP {cm[1][1]}.")
    last = anomaly.last_run()
    st.markdown("**Anomaly model** - Isolation Forest "
                + (f"`{last['version']}`, {last['records']} records analysed." if last and last.get("ran") else
                   "(not trained yet)."))
