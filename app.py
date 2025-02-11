import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_recipe_model():
    tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/t5-recipe-generation")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

recipe_pipe = load_recipe_model()

def generate_recipe(description):
    result = recipe_pipe(
        description,
        max_length=300,
        num_beams=5,  
        repetition_penalty=2.0, 
        num_return_sequences=1
    )
    return result[0]['generated_text']


def clean_recipe_output(output):
    sentences = output.split(". ")
    cleaned_steps = list(dict.fromkeys([sentence.strip() for sentence in sentences if sentence.strip()]))
    steps = [f"Step {i + 1}: {step}" for i, step in enumerate(cleaned_steps)]
    return steps

st.title("AI Food Detective 🍲")
st.write("Describe the dish you remember, and the AI will generate a recipe closest to your memory!")


description = st.text_area(
    "Describe your dish (e.g., ingredients, flavors, textures, and any details you remember):",
    placeholder="E.g., crispy fried samosas filled with spiced paneer and peas"
)


if st.button("Find My Recipe"):
    if description.strip():
        cleaned_description = description.strip().capitalize()
        with st.spinner("Detecting your recipe..."):
            recipe = generate_recipe(cleaned_description)
            recipe_steps = clean_recipe_output(recipe)
        
        st.success("Here’s your recipe, step by step:")
        
      
        for step in recipe_steps:
            st.write(step)
    else:
        st.error("Please provide a description of the dish!")

st.markdown("---")
st.caption("Powered by AI and Transformers")
