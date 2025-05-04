# --- Imports ---
import os
import streamlit as st
import time

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

# --- Tracing Setup ---
tracer_provider = register(
    project_name="recipe-builder",
    auto_instrument=True,
    set_global_tracer_provider=True
)

LangChainInstrumentor().instrument()

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful, budget-conscious home cook assistant."),
    HumanMessagePromptTemplate.from_template(
        "Given the ingredients: {ingredients}, location: {location}, and budget: ${budget}, "
        "suggest a dinner recipe using the ingredients. "
        "If additional groceries are needed, list them with estimated cost based on city averages. "
        "Also, suggest where to buy these ingredients locally, and imagine what the completed dish might look like."
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
if st.button("Suggest a Recipe"):
    with st.spinner("Cooking up something delicious..."):
        start_time = time.time()

        result = recipe_chain.invoke({
            "ingredients": ingredients,
            "location": location,
            "budget": budget
        })

        latency = round(time.time() - start_time, 2)

        # Output section
        st.markdown(f"🧠 **Model used:** `{llm.model_name}`")
        st.markdown(f"⏱️ **Response time:** `{latency} seconds`")
        st.success("Here’s your dinner idea:")

        try:
            st.markdown(result.content)
        except AttributeError:
            st.markdown(result)

        st.markdown(f"📎 **Raw JSON Output:**")
        st.write(result)
