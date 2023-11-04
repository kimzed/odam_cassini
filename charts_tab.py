import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
from matplotlib import pyplot as plt


def charts():
    ## dummy data ##
    locations = ["Sudan", "Ethiopia", "Iraq", "Lesotho", "South Africa"]
    date_range = [date(2022, 1, 1), date(2023, 1, 1)]
    capacity = 88

    df = pd.read_csv('dams.csv', header=0, low_memory=False)
    df = df.head(10)
    df = df[['Dam Name', 'Other Names', 'Former Names', 'Longitude', 'Latitude']]
    df.rename(columns={'Longitude': 'longitude', "Latitude": "latitude"}, inplace=True)


    option = st.selectbox('Dam Location',(df['Dam Name']))
    st.write(df)
    st.map(df[(df['Dam Name'] == option)])
    st.date_input("Date Range", date_range)

    st.markdown("# oDam")
    st.image("cover.jpg")
    st.markdown("###### :ocean: Know your water :ocean:")


    col1, col2 = st.columns(2)


    dummy_data = pd.DataFrame()
    dummy_data["Date"] = np.linspace(0, 10, 100)
    dummy_data["Inflow"] = [np.sin(i) for i in dummy_data["Date"]]
    dummy_data["Outflow"] = [np.cos(i) for i in dummy_data["Date"]]
    dummy_data["Storage"] = [np.tan(i) for i in dummy_data["Date"]]


    with col1:
        st.line_chart(dummy_data, x="Date", y="Inflow")
        st.line_chart(dummy_data, x="Date", y="Storage")

    with col2:
        st.line_chart(dummy_data, x="Date", y=["Inflow", "Outflow"])

        with st.container():
            fig, ax = plt.subplots()
            ax.pie([(100-capacity), capacity], colors=["#A6ACAF", "#1D84CD"], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
            ax.set_title(f"Dam is at {capacity}% capacity")
            st.pyplot(fig)
            st.divider()