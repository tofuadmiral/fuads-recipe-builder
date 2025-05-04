import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# -- ENVIRONMENT SETUP ------------------------------------------------

# Set your OpenAI key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Register Phoenix tracing
register(project_name="recipe-builder", auto_instrument=True)
LangChainInstrumentor().instrument()

# -- Select Dropdowns Setup --------------------------------------------
cities = [
    "New York, New York, United States",
    "Toronto, Ontario, Canada",
    "London, United Kingdom",
    "San Francisco, California, United States",
    "Montreal, Quebec, Canada",
    "Paris, France",
    "Berlin, Germany",
    "Tokyo, Japan",
    "Sydney, Australia", 
    "Denver, Colarado, United States"
]

# -- LANGCHAIN SETUP ---------------------------------------------------

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Prompt to guide recipe generation
prompt = PromptTemplate(
    input_variables=["ingredients", "location", "budget"],
    template="""
You are a helpful, budget-conscious home cook assistant.
Given the ingredients: {ingredients}, the users location {location}, and a budget of {budget} dollars,
suggest one creative dinner recipe I can make tonight.
Use as many listed ingredients as possible.
If additional groceries are needed, list them with estimated cost. 
The cost should be based on average grocery costs for the user's city. Scrape the internet if you need to understand how much groceries cost in that area. 
"""
)

# Chain for recipe generation
recipe_chain = LLMChain(llm=llm, prompt=prompt)

# -- STREAMLIT UI -------------------------------------------------------

st.set_page_config(page_title="Fuad's Recipe Builder", page_icon="🧑🏽‍🍳")
st.title("🧑🏽‍🍳 Fuad's Recipe Builder")

st.markdown("Enter your existing ingredients, your target budget, and location. Get a recipe to cook!")
ingredients = st.text_input("Ingredients you have (comma-separated):", placeholder="e.g. eggs, spinach, rice")
location = st.selectbox("Your City + Country:", cities)
budget = st.slider("What’s your budget (in USD)?", min_value=5, max_value=100, step=1, value=20)

if st.button("Suggest a Recipe"):
    with st.spinner("Cooking up something delicious..."):
        result = recipe_chain.run({"ingredients": ingredients, "location": location, "budget": budget})
        st.success("Here’s your dinner idea:")
        st.write(result)
