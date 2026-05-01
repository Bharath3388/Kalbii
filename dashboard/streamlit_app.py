"""Streamlit dashboard. Reads from the same MongoDB the API writes to."""

from __future__ import annotations
import os
import datetime as dt

import pandas as pd
import plotly.express as px
import streamlit as st

from app.db.repositories import RecordsRepo

st.set_page_config(page_title="Kalbii — Risk Insights", layout="wide")
st.title("Encrypted Multi-Modal Intelligence — Risk Dashboard")

repo = RecordsRepo()
records = repo.recent(limit=500)

if not records:
    st.info("No records yet. POST to `/ingest` to populate the dashboard.")
    st.stop()

df = pd.DataFrame(records)
df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values("created_at")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records",  len(df))
c2.metric("Avg risk score", f"{df['risk_score'].mean():.1f}")
c3.metric("HIGH risk %",    f"{(df['risk_label'] == 'HIGH').mean() * 100:.1f}%")
last_24h = df[df["created_at"] >= (dt.datetime.utcnow() - dt.timedelta(days=1))]
c4.metric("Records (24h)",  len(last_24h))

# time series
st.subheader("Risk score over time")
st.plotly_chart(px.line(df, x="created_at", y="risk_score", color="risk_label",
                        markers=True), use_container_width=True)

# distribution
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Risk-label distribution")
    st.plotly_chart(px.pie(df, names="risk_label"), use_container_width=True)
with col_b:
    st.subheader("Risk score histogram")
    st.plotly_chart(px.histogram(df, x="risk_score", nbins=20), use_container_width=True)

# recent records
st.subheader("Recent records")
display_cols = ["created_at", "risk_label", "risk_score", "decrypted_text"]
st.dataframe(df[display_cols].tail(50)[::-1], use_container_width=True)

with st.expander("Inspect raw record"):
    job_ids = df["job_id"].tolist()
    pick = st.selectbox("job_id", job_ids[::-1])
    if pick:
        st.json(repo.by_job(pick))
