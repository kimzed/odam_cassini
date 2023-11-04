import streamlit as st
from charts_tab import charts
from team_tab import team
from about_tab import about

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
)

chart_tab, about_tab, team_tab = st.tabs(["Charts", "About", "Team"])

with chart_tab:
    charts()

with about_tab:
    about()

with team_tab:
    team()
