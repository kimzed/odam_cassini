import streamlit as st

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)
st.image("frontend/images/oDam_logo.png")

import pandas as pd
import random


df = pd.read_csv('frontend/dams.csv', header=0, low_memory=False)

river = st.selectbox('Select River',(['Blue Nile']))

dams = df[(df['river'] == river)]
zoom_level = 2.5
st.map(dams, zoom=zoom_level, color='#008000')
#st.error('This is an error', icon="🚨")

with st.container():
    st.header(f'Dam Capacities for the {river} River')

    dam_names = dams['name'].tolist()
    dam_names.sort(key=lambda x: dams[dams['name'] == x]['rank'].values[0])

    dam_levels = [(name, random.randint(0,100)) for name in dam_names]
   
    dam_levels_dict = dict(dam_levels)
    st.bar_chart(dam_levels_dict, use_container_width=True)

st.image("frontend/images/logos.PNG")