import streamlit as st

st.title("Interactive AI Art & Story Generator")
st.write("Let's build something amazing together!")

#Placeholder for story and image generation
theme = st.text_input("Enter a theme or genre:")
if theme:
    st.write(f"Generating story and art for: {theme}")

def generate_story(theme):
    return f"Once upon a time in a world of {theme}, an incredible adventure unfolded..."

print(generate_story("cyberpunk"))