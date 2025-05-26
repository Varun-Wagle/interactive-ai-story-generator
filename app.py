import streamlit as st
from transformers.pipelines import pipeline
from diffusers import StableDiffusionPipeline
import torch
from typing import Optional
from io import BytesIO
from fpdf import FPDF

# ------------------------ Setup ------------------------

st.set_page_config(page_title="AI Story & Art Generator", layout="centered")

st.title("🚀 Interactive AI Story & Art Generator")
st.write("Generate captivating stories and visuals with Generative AI!")

# ------------------------ Load Models ------------------------

@st.cache_resource(show_spinner="Loading story generator...")
def load_story_generator():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=0, truncation=True)

@st.cache_resource(show_spinner="Loading image generator...")
def load_image_generator():
    return StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

story_generator = load_story_generator()
image_generator = load_image_generator()

# ------------------------ Story Generation ------------------------

def generate_story(theme: str, max_new_tokens: int, temperature: float, top_p: float, continue_story: Optional[str]=None) -> str:
    prompt = continue_story if continue_story else f"In a {theme} world, there was a secret hidden beneath the surface. Once upon a time,"
    output = story_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.5,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=50256
    )
    return output[0]['generated_text'] if output else "Error: No story generated."

# ------------------------ Image Generation ------------------------

def generate_image(prompt: str):
    with st.spinner("Generating image..."):
        try:
            with torch.no_grad():
                image = image_generator(prompt).images[0]
            return image
        except Exception as e:
            st.error(f"Image generation failed: {e}")
            return None

# ------------------------ Download Functions ------------------------

def image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def story_to_txt(story_text):
    return story_text.encode('utf-8')

def story_to_pdf(story_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    lines = story_text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, txt=line, align='L')
    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    return pdf_bytes.getvalue()

# ------------------------ Session State ------------------------

if "generated_story" not in st.session_state:
    st.session_state.generated_story = ""

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# ------------------------ Sidebar for Settings ------------------------

st.sidebar.header("🔧 Settings")
theme = st.sidebar.text_input("Theme/Genre:", value="romance")
max_new_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 100, step=50)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.5, 0.5, step=0.1)
top_p = st.sidebar.slider("Top-P Sampling", 0.5, 1.0, 0.55, step=0.05)

# ------------------------ Buttons ------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate New Story"):
        st.session_state.generated_story = generate_story(theme, max_new_tokens, temperature, top_p)
        st.session_state.generated_image = None  # Reset image
        st.subheader("Generated Story")
        st.write(st.session_state.generated_story)

with col2:
    if st.button("Continue Story"):
        continuation = st.session_state.generated_story or ""
        st.session_state.generated_story = generate_story(theme, max_new_tokens, temperature, top_p, continue_story=continuation)
        prompt_for_image = f"A beautiful illustration of a {theme} story"
        st.session_state.generated_image = generate_image(prompt_for_image)
        st.subheader("Extended Story")
        st.write(st.session_state.generated_story)

# ------------------------ Display Image and Downloads ------------------------

if st.session_state.generated_image:
    st.image(st.session_state.generated_image, caption="AI-Generated Illustration", use_column_width=True)
    img_bytes = image_to_bytes(st.session_state.generated_image)
    st.download_button("Download Image", data=img_bytes, file_name="generated_image.png", mime="image/png")

if st.session_state.generated_story:
    st.download_button("Download Story (TXT)", data=story_to_txt(st.session_state.generated_story), file_name="generated_story.txt", mime="text/plain")
    st.download_button("Download Story (PDF)", data=story_to_pdf(st.session_state.generated_story), file_name="generated_story.pdf", mime="application/pdf")