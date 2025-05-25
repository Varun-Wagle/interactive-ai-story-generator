import streamlit as st
from transformers import pipeline

#Title
st.title("Interactive AI Story & Art Generator")
st.write("Generate captivating stories and visuals with Generative AI!")

#User Input
theme = st.text_input("Enter a theme or genre (e.g., fantasy, cyberpunk, sci-fi):")
generate_button = st.button("Generate Story & Art")

#Load story generator pipeline (GPT 2)
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

if generate_button and theme:
    with st.spinner("Generating Story..."):
        story = generator(f"Once upon a time in a world of {theme}, ", max_length=200, num_return_sequences=1)[0]['generated_text']
    st.markdown("### 📝 Generated Story")
    st.write(story)

    st.markdown("### 🖼 Generated Art")
    st.write("AI Art generation coming soon!")
else:
    st.info("Enter a theme and click 'Generate Story & Art' to begin")