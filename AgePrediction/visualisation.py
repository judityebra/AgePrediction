import streamlit as st

st.set_page_config(layout="wide")
st.title("Age Prediction project")
st.markdown('### Age estimation for people between 15 and 80 years old.')

losses = ["ce", "coral", "ordinal"]
selected_option = st.selectbox("Select a loss for the Resnet34 model to be applied", losses, index = None)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', width=200)