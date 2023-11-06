import streamlit as st

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)
st.image("frontend/images/oDam_logo.png")
st.video("https://www.youtube.com/watch?v=TkGeAKCw7cw")

st.header("📒 Project Overview")
st.markdown(
"""
**oDam: Open Diplomatic Aquatic Modeling**

Our idea at oDam is to address the critical challenges posed by hydropolitical tensions and water management issues that affect local communities. These tensions can lead to limited access to clean drinking water, compromised food security, and disruptions to local hydropower generation. Decisions made upstream impact the downstream communities but information exchange between parties is limited due to political interests or lack of infrastructure. 
"""    
)
st.image("frontend/images/oDam_approach.png") 

st.image("frontend/images/logos.PNG")