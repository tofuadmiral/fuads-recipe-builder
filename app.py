# --- Imports ---
import os
import streamlit as st
import time

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # ✅ updated import
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Phoenix Setup
os.environ["PHONEIX_CLIENT_HEADERS"] = f"api_key={st.secrets['PHOENIX_API_KEY']}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# --- Secrets / API Key ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Configure the Phoenix tracer
tracer_provider = register(
    project_name="recipe-builder",
    auto_instrument=True,
    set_global_tracer_provider=True
)
LangChainInstrumentor().instrument()


prompt = PromptTemplate(
    input_variables=["ingredients", "location", "budget"],
    template="""
You are a helpful, budget-conscious home cook assistant.
Given the ingredients: {ingredients}, the user's location: {location}, and a budget of ${budget},
suggest a creative dinner recipe using the listed items.
If additional groceries are needed, list them with an estimated cost based on the city's average prices.
"""
)

# ✅ Runnable syntax replaces deprecated LLMChain
recipe_chain = prompt | llm

# --- UI Setup ---
st.set_page_config(page_title="Fuad's Recipe Builder", page_icon="🧑🏽‍🍳")
st.title("🧑🏽‍🍳 Fuad's Recipe Builder")
st.markdown("Enter your ingredients, location, and budget to get a smart dinner suggestion!")

ingredients = st.text_input("Ingredients you have (comma-separated):", placeholder="e.g. eggs, spinach, rice")
location = st.selectbox("Your City + Country:", [
    "New York, United States", "Toronto, Canada", "London, United Kingdom",
    "San Francisco, United States", "Montreal, Canada", "Paris, France",
    "Berlin, Germany", "Tokyo, Japan", "Sydney, Australia", "Denver, United States"
])
budget = st.slider("Budget (USD)", min_value=5, max_value=100, step=1, value=20)

# --- Generate Recipe ---
if st.button("Suggest a Recipe"):
    with st.spinner("Cooking up something delicious..."):
        # time our response
        start_time = time.time()

        result = recipe_chain.invoke({
            "ingredients": ingredients,
            "location": location,
            "budget": budget
        })

        latency = round(time.time() - start_time, 2)
        st.markdown(f"🧠 **Model used:** `{llm.model_name}`")
        st.markdown(f"⏱️ **Response time:** `{latency} seconds`")
        st.success("Here’s your dinner idea:")
        try:
            st.markdown(result.content)
        except AttributeError:
            st.markdown(result)  # fallback if result is plain text
        st.markdown(f"🧾 **JSON Result:**")
        st.write(result)
