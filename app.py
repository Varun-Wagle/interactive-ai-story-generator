import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from fpdf import FPDF
from io import BytesIO
from PIL import Image

# ------------------- Utility Functions ------------------- #

@st.cache_resource(show_spinner=True)
def load_story_generator():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0, truncation=True)

@st.cache_resource(show_spinner=True)
def load_image_generator():
    return StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_story(theme, max_tokens, temperature, top_p, continuation=None):
    generator = load_story_generator()
    prompt = continuation if continuation else f"In a {theme} world, there was a secret hidden beneath the surface. Once upon a time,"
    output = generator(prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=1.5, do_sample=True, num_return_sequences=1, eos_token_id=50256)
    story_text = output[0]['generated_text'] if output else "Error: No story generated."
    return story_text

def generate_image(prompt):
    pipe = load_image_generator()
    with st.spinner("Generating Image..."):
        image = pipe(prompt).images[0]
    return image

def story_to_pdf(story_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in story_text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return BytesIO(pdf_bytes)

def story_to_txt(story_text):
    return BytesIO(story_text.encode('utf-8'))

def image_to_bytes(image: Image.Image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ------------------- Streamlit App ------------------- #

st.set_page_config(page_title="AI Story & Art Generator", layout="wide")
st.title("🚀 Interactive AI Story & Art Generator")

# Sidebar for Settings
st.sidebar.header("Settings")
theme = st.sidebar.text_input("Enter Theme/Genre", value="romance")
max_new_tokens = st.sidebar.slider("Story Length (max_new_tokens)", 50, 1000, 100, step=50)
temperature = st.sidebar.slider("Creativity (temperature)", 0.5, 1.5, 0.7, step=0.1)
top_p = st.sidebar.slider("Focus (top_p)", 0.5, 1.0, 0.9, step=0.05)

# Initialize Session State
if "story" not in st.session_state:
    st.session_state.story = ""
if "image" not in st.session_state:
    st.session_state.image = None

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("✨ Generate New Story"):
        st.session_state.story = generate_story(theme, max_new_tokens, temperature, top_p)
        st.session_state.image = None

with col2:
    if st.button("➡️ Continue Story"):
        st.session_state.story = generate_story(theme, max_new_tokens, temperature, top_p, st.session_state.story)
        st.session_state.image = None

if st.session_state.story:
    st.subheader("Generated Story")
    st.write(st.session_state.story)

    # Download options for story
    st.download_button("📄 Download Story (TXT)", data=story_to_txt(st.session_state.story), file_name="story.txt", mime="text/plain")
    st.download_button("📕 Download Story (PDF)", data=story_to_pdf(st.session_state.story), file_name="story.pdf", mime="application/pdf")

    # Generate Image
    if st.button("🎨 Generate Image for Story"):
        prompt = f"{theme} {st.session_state.story[:100]}"
        st.session_state.image = generate_image(prompt)

if st.session_state.image:
    st.subheader("AI-Generated Illustration")
    st.image(st.session_state.image, caption="Generated Art", use_column_width=True)

    # Download option for image
    st.download_button("🖼️ Download Image", data=image_to_bytes(st.session_state.image), file_name="generated_image.png", mime="image/png")
    