# --- Imports ---
import os
import re
import streamlit as st
import time
import openai

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# --- Phoenix Setup ---
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={st.secrets['PHOENIX_API_KEY']}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# --- Secrets / API Key ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Tracing Setup ---
from phoenix.otel import register

tracer_provider = register(
    project_name="recipe-builder",
    auto_instrument=True,
    set_global_tracer_provider=True,
    batch=True
)

LangChainInstrumentor().instrument()

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful, budget-conscious home cook assistant."),
    HumanMessagePromptTemplate.from_template(
        "Given the ingredients: {ingredients}, location: {location}, and budget: ${budget}, "
        "suggest a dinner recipe using the ingredients. "
        "If additional groceries are needed, list them with estimated cost based on city averages. "
        "Also, suggest where to buy these ingredients locally, and imagine what the completed dish might look like."
        "Also, return a vivid one-sentence visual description of the completed dish named Visual Description:"
    )
])

# ✅ Runnable chain
recipe_chain = prompt | llm

# --- UI Setup ---
st.set_page_config(page_title="Fuad's Recipe Builder", page_icon="🧑🏽‍🍳")
st.title("🧑🏽‍🍳 Fuad's Recipe Builder")
st.markdown("Enter your ingredients, location, and budget to get a smart dinner suggestion!")

ingredients = st.text_input("Ingredients you have (comma-separated):", placeholder="e.g. eggs, spinach, rice")
location = st.selectbox("Your City + Country:", [
    "New York, United States", "Toronto, Canada", "London, United Kingdom",
    "San Francisco, United States", "Montreal, Canada", "Paris, France",
    "Berlin, Germany", "Tokyo, Japan", "Sydney, Australia", "Denver, United States", "Rio de Janeiro, Brazil"
])
budget = st.slider("Budget (USD)", min_value=5, max_value=100, step=1, value=20)


# --- Generate Recipe ---
@st.cache_resource
def call_llm(ingredients, location, budget):
    result = recipe_chain.invoke({
        "ingredients": ingredients,
        "location": location,
        "budget": budget
    })
    return result

# Generate image
def generate_recipe_image(prompt: str) -> str:
    response = openai_client.images.generate(
        model="dall-e-3",  # or "dall-e-2"
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url


def generate_recipe():
    if st.button("Suggest a Recipe"):
        with st.spinner("Cooking up something delicious..."):
            start_time = time.time()

            result = call_llm(ingredients, location, budget)

            latency = round(time.time() - start_time, 2)

            # Output section
            st.markdown(f"🧠 **Model used:** `{llm.model_name}`")
            st.markdown(f"⏱️ **Response time:** `{latency} seconds`")
            st.success("Here's your dinner idea:")

            try:
                st.markdown(result.content)
            except AttributeError:
                st.markdown(result)

            st.markdown(f"📎 **Raw JSON Output:**")
            st.write(result)

            # Show Image

            visual_description = f"A realistic photo of a dish made with {ingredients}"

            with st.spinner("Generating an image of your dish..."):
                image_url = generate_recipe_image(visual_description)
                st.image(image_url, caption="🍽️ Your Dish (AI-generated)")


generate_recipe()
