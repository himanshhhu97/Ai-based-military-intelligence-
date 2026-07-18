import streamlit as st
import plotly.express as px

from utils.data_loader import load_data, apply_global_filters, global_filter_sidebar
from utils.auth import require_login, logout_button
from utils.theme import apply_theme

st.set_page_config(
    page_title="AI Military Intelligence Dashboard",
    page_icon="🛡",
    layout="wide",
)

# ----------------------------------------------------------------------
# Access control — every page in pages/ also calls require_login(), so
# a user cannot deep-link around the gate.
# ----------------------------------------------------------------------
require_login()
apply_theme()
logout_button()

st.title("🛡 AI-Based Military Intelligence Dashboard")

st.markdown("""
Analytical dashboard built on the **Global Terrorism Database (GTD)**,
combining historical trend analysis, geospatial intelligence, and
machine-learning forecasting for open-source research and coursework use.
""")

df = load_data()
year_range, countries = global_filter_sidebar(df)
df_f = apply_global_filters(df)

# ----------------------------------------------------------------------
# Summary KPIs
# ----------------------------------------------------------------------
st.subheader("Dashboard Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Incidents", f"{len(df_f):,}")
c2.metric("Fatalities", f"{int(df_f['nkill'].sum()):,}")
c3.metric("Injured", f"{int(df_f['nwound'].sum()):,}")
c4.metric("Countries", df_f["country_txt"].nunique())
c5.metric("Active Groups", df_f["gname"].nunique())

st.divider()

# ----------------------------------------------------------------------
# Trend
# ----------------------------------------------------------------------
st.subheader("Attacks Over Years")

yearly = df_f.groupby("iyear").size().reset_index(name="Attacks")
fig = px.area(yearly, x="iyear", y="Attacks", markers=True)
fig.update_traces(line_color="#ff4b4b")
st.plotly_chart(fig, width='stretch')

st.divider()

# ----------------------------------------------------------------------
# Quick regional snapshot (new vs. the original single-chart Home page)
# ----------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Incidents by Region")
    region = df_f["region_txt"].value_counts().reset_index()
    region.columns = ["Region", "Incidents"]
    fig_r = px.bar(region, x="Incidents", y="Region", orientation="h", color="Incidents")
    st.plotly_chart(fig_r, width='stretch')

with right:
    st.subheader("Top 5 Most Active Groups")
    groups = df_f[df_f["gname"] != "Unknown"]["gname"].value_counts().head(5).reset_index()
    groups.columns = ["Group", "Attacks"]
    fig_g = px.pie(groups, names="Group", values="Attacks", hole=0.4)
    st.plotly_chart(fig_g, width='stretch')

st.success("👈 Use the sidebar to explore the Global Threat Map, run predictions, forecast trends, and more.")

st.info("""
**Available Modules**
- 🌍 Global Threat Map · 🌎 Country Analysis · 🧭 Hotspot Clustering
- 🤖 Attack Prediction · 🚨 Threat Level Prediction · 🕸 Network Analysis
- 📈 Forecasting · 🧠 AI Intelligence Report · 📊 Data Explorer · ⚙ Settings
""")
