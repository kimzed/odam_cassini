import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt
from generate_analytics import analytics_fig

def make_rect(x, y, col):
    return alt.Chart(pd.DataFrame({'x': x, 'y': y})).mark_rect(
    color=col,
    opacity=0.3
    ).encode(
        x=alt.X2Datum(x[0]),
        x2=alt.X2Datum(x[1]),
        y=alt.X2Datum(y[0]),
        y2=alt.X2Datum(y[1]),
    )

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

st.image("frontend/images/oDam_logo.png")

## data ##
capacity = 88
df = pd.read_csv('frontend/dams.csv', header=0, low_memory=False)
dummy_data = pd.DataFrame()
dummy_data["Date"] = np.linspace(0, 10, 12)
dummy_data["Predicted Inflow"] = [np.sin(i) for i in dummy_data["Date"]]
dummy_data["Outflow"] = [np.cos(i) for i in dummy_data["Date"]]
dummy_data["Storage"] = [np.tan(i) for i in dummy_data["Date"]]

monthly_df = pd.read_csv('frontend/data/monthly_average_water_stored.csv', header=0, low_memory=False)
monthly_df = monthly_df.rename(columns={"month": "Month", "water_stored": "Storage"})
monthly_df["Demand"] = [0, 0, 2, 4, 6, 8, 8, 4, 0, 0, 0, 0]
forecast_df = pd.read_csv('frontend/data/forecasting.csv', header=0, low_memory=False)
###


option = st.selectbox('Select Dam',(df['name']))
update = st.button("Find")
if update:
    selected_df = df[(df['name'] == option)]
    st.map(selected_df, zoom=2, color='#008000')

    st.divider()
    st.header("Services")
    col1, col2 = st.columns(2)
    with col1:
        #Early Warning System
        early_chart = alt.Chart(forecast_df)
        early_line = early_chart.mark_line(color="#1D84CD").encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('normalized_downstream_flux:Q', scale=alt.Scale(domain=[-1.5, 1.5]), title='<--   Drought Risk   |   Flood Risk   -->')
        )
        rect1 = make_rect(x=[0, 12], y=[1, 1.5], col="#ffeda0")
        rect2 = make_rect(x=[0, 12], y=[-1.5, -1], col="#ffeda0")
        rect3 = make_rect(x=[0, 12], y=[1.5, 2], col="#feb24c")
        rect4 = make_rect(x=[0, 12], y=[-1.5, -2], col="#feb24c")      

        early_combined = (early_line + rect1 + rect2 + rect3 + rect4).properties(
            title={'text': 'Early Warning System', 'anchor': 'middle'}
        )
        # add risk of flooding/dry conditions
        # update data 
        st.altair_chart(early_combined)

    with col2:
        # Annual Water Management Plan
        annual_chart = alt.Chart(monthly_df)
        annual_bars = annual_chart.mark_bar(color='grey').encode(
            x='Month:Q',
            y=alt.Y('Demand:Q', title='Water'),
        )
        annual_line = annual_chart.mark_line().encode(
            x='Month:Q',
            y=alt.Y('Storage:Q', title='Water'),
        )
        combined_chart = alt.layer(annual_bars, annual_line).properties(
        title={'text': 'Annual Water Management Plan', 'anchor': 'middle'}
        )
        st.altair_chart(combined_chart)

    st.divider()

    st.header("Dam Analytics")
    analytics = analytics_fig()
    st.plotly_chart(analytics)

st.image("frontend/images/logos.PNG")