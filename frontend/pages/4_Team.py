import streamlit as st

st.set_page_config(
    page_title="oDam",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)
st.image("frontend/images/oDam_logo.png")


st.header("🤼 Meet the team")
bios = """
**Awad:** A Sudanese professional with a background in Civil Engineering, he recently acquired  a Master's degree in Earth and Environment at Wageningen.

**Fedor:** Master's student in Computer Science at KU Leuven, enthusiastic about hackathons and entrepreneurship.

**Janina:** Meteorologist and climatologist in the field of food security with a focus on the global south and experience in business development.

**Cédric:** A French geo-data scientist with a degree from Wageningen University, also studied History and Geography in France. He's set to join the UNDP's Sustainable Development Goals AI Lab in November, contributing his skills to sustainability projects. 

**Lars:**  A recent Artificial Intelligence graduate passionate about data visualization and modeling, interested in bringing these tools to the Space and Humanitarian sector.

**Edgar:** A computer vision PhD and CTO at his startup.

**Tom:** A civil engineer with experience in infrastructure.

**Merlijn:** An electromechanical engineering student with an interest in IT and business.
"""

st.markdown(bios)

st.image("frontend/images/logos.PNG")