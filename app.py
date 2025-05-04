# --- Imports ---
import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

# 🔁 Phoenix Tracing
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# --- Tracing Setup ---
register(
    project_name="recipe-builder",
    auto_instrument=False,
    set_global_tracer_provider=False,
)
LangChainInstrumentor().instrument()

# --- Secrets / API Key ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["ingredients", "location", "budget"],
    template="""
You are a helpful, budget-conscious home cook assistant.
Given the ingredients: {ingredients}, the user's location: {location}, and a budget of ${budget},
suggest a creative dinner recipe using the listed items.
If additional groceries are needed, list them with an estimated cost based on the city's average prices.
"""
)

recipe_chain = LLMChain(llm=llm, prompt=prompt)

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
        st.success("Here’s your dinner idea:")
        st.markdown(f"🧠 **Model used:** `{llm.model_name}`")
        st.markdown(f"⏱️ **Response time:** `{latency} seconds`")
        st.markdown(f"**User Facing Result")
        if isinstance(result, dict) and "text" in result:
            st.markdown(result["text"])
        else:
            st.markdown(result)  # fallback if it's plain markdown
        st.markdown(f"⏱️ **JSON Result:**")
        st.write(result)
