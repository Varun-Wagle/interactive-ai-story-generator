# ===========================
# 📚 AI Story & Art Generator
# ===========================

import streamlit as st
from transformers import pipeline
from diffusers import DiffusionPipeline
import torch
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import re
import random

# ===========================
# 🔧 Utility Functions
# ===========================

@st.cache_resource(show_spinner=True)
def load_story_generator():
    device = 0 if torch.cuda.is_available() else -1  # ✅ Fix: dynamic device selection
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=device)

@st.cache_resource(show_spinner=True)
def load_image_generator(model_name):
    model_map = {
        "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
        "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
        "Realistic Vision": "SG161222/Realistic_Vision_V4.0_noVAE",
        "DreamShaper": "Lykon/dreamshaper-7"
    }
    model_id = model_map.get(model_name, "stabilityai/stable-diffusion-2-1")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_story(theme, length, temperature, top_p, characters, setting, mood):
    generator = load_story_generator()
    prompt = f"A {theme} story set in {setting} with characters {characters}, mood: {mood}. Once upon a time,"
    output = generator(prompt, max_new_tokens=length, temperature=temperature, top_p=top_p,
                       repetition_penalty=1.5, do_sample=True, num_return_sequences=1, eos_token_id=50256)
    return output[0]['generated_text']

def extract_image_prompt(story_text):
    characters = re.findall(r'\b[A-Z][a-z]+\b', story_text)
    settings = re.findall(r'\b(?:forest|castle|city|mountain|beach|village|desert|kingdom|world)\b', story_text, re.IGNORECASE)
    emotions = re.findall(r'\b(?:happy|sad|tense|mysterious|dark|bright|colorful|gloomy)\b', story_text, re.IGNORECASE)
    prompt = f"{', '.join(characters[:3])} in {', '.join(settings[:2])} with a {', '.join(emotions[:2])} mood, digital art"
    return prompt or story_text[:100]

def generate_image(prompt, model_name, resolution, guidance_scale, seed=None):
    pipe = load_image_generator(model_name)
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed) if seed else None
    with st.spinner(f"Generating Image with {model_name}..."):
        image = pipe(prompt, height=resolution, width=resolution, guidance_scale=guidance_scale, generator=generator).images[0]
    return image

def story_to_pdf(story):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in story.splitlines():
        pdf.multi_cell(0, 10, line)
    return BytesIO(pdf.output(dest='S').encode('latin1'))

def story_to_txt(story):
    return BytesIO(story.encode('utf-8'))

def image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ===========================
# 🎛️ Streamlit App Layout
# ===========================

st.set_page_config(page_title="StoryForge AI - Story & Art Generator", layout="wide")
st.markdown("""
    <style>
    .stButton>button { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home", "📖 Story Generator", "🎨 Image Generator"])

# ===========================
# 🏠 Home Tab
# ===========================
with tabs[0]:
    st.title("🎉 Welcome to StoryForge AI "
             "- Story & Art Generator")
    st.write("""
    Dive into a world of creativity powered by cutting-edge AI.
    - Generate immersive stories in your favorite genres.
    - Transform stories or your own prompts into stunning artworks.
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Go to Story Generator"):
            st.session_state.page = "story"
    with col2:
        if st.button("🎨 Go to Image Generator"):
            st.session_state.page = "image"
    st.sidebar.empty()  # ✅ Disable sidebar in Home tab

# ===========================
# 📖 Story Generator Tab
# ===========================
with tabs[1]:
    st.header("✨ Story Generator")
    with st.sidebar:
        st.subheader("📚 Story Configuration")
        theme = st.text_input("Theme/Genre", value="fantasy")
        characters = st.text_input("Characters (comma-separated)", value="Alice, Bob")
        setting = st.text_input("Setting", value="enchanted forest")
        mood = st.text_input("Mood", value="mysterious")
        length = st.slider("Story Length", 50, 1000, 300, step=50)
        temperature = st.slider("Creativity", 0.5, 1.5, 1.0, step=0.1)
        top_p = st.slider("Focus", 0.5, 1.0, 0.9, step=0.05)
        generate_story_button = st.button("Generate Story")

    if generate_story_button:
        st.session_state.story = generate_story(theme, length, temperature, top_p, characters, setting, mood)
        st.session_state.image = None

    if st.session_state.get("story"):
        st.subheader("Generated Story")
        st.write(st.session_state.story)
        st.download_button("Download TXT", data=story_to_txt(st.session_state.story), file_name="story.txt", mime="text/plain")
        st.download_button("Download PDF", data=story_to_pdf(st.session_state.story), file_name="story.pdf", mime="application/pdf")
        if st.button("🎨 Generate Image from Story"):
            prompt = extract_image_prompt(st.session_state.story)
            st.session_state.image = generate_image(prompt, "Stable Diffusion 2.1", 512, 7.5)
        if st.session_state.get("image"):
            st.subheader("Generated Illustration")
            st.image(st.session_state.image)
            st.download_button("Download Image", data=image_to_bytes(st.session_state.image), file_name="story_image.png", mime="image/png")

# ===========================
# 🎨 Image Generator Tab
# ===========================
with tabs[2]:
    st.header("🎨 Image Generator")
    with st.sidebar:
        st.subheader("🎨 Image Configuration")
        prompt = st.text_input("Custom Image Prompt")
        model_name = st.selectbox("Select Model", ["Stable Diffusion 2.1", "SDXL", "Realistic Vision", "DreamShaper"])
        resolution = st.selectbox("Resolution", [512, 768, 1024], index=0)
        guidance_scale = st.slider("Guidance Scale", 3.0, 15.0, 7.5, step=0.5)
        seed = st.number_input("Random Seed (optional)", min_value=0, max_value=999999, step=1)
        generate_image_button = st.button("Generate Image")

    if generate_image_button:
        if prompt:
            st.session_state.image = generate_image(prompt, model_name, resolution, guidance_scale, seed if seed else None)
        else:
            st.warning("Enter a prompt!")

    if st.session_state.get("image"):
        st.subheader("Generated Image")
        st.image(st.session_state.image)
        st.download_button("Download Image", data=image_to_bytes(st.session_state.image), file_name="custom_image.png", mime="image/png")
