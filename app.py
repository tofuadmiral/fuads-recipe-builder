import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# -- ENVIRONMENT SETUP ------------------------------------------------

# Set your OpenAI key safely
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "sk-...your-key-here...")

# Register Phoenix tracing
register(project_name="recipe-builder", auto_instrument=True)
LangChainInstrumentor().instrument()

# -- LANGCHAIN SETUP ---------------------------------------------------

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Prompt to guide recipe generation
prompt = PromptTemplate(
    input_variables=["ingredients", "budget"],
    template="""
You are a helpful, budget-conscious home cook assistant.
Given the ingredients: {ingredients} and a budget of {budget} dollars,
suggest one creative dinner recipe I can make tonight.
Use as many listed ingredients as possible.
If additional groceries are needed, list them with estimated cost.
"""
)

# Chain for recipe generation
recipe_chain = LLMChain(llm=llm, prompt=prompt)

# -- STREAMLIT UI -------------------------------------------------------

st.set_page_config(page_title="Smart Recipe Builder", page_icon="🍳")
st.title("🍳 Smart Recipe Builder")

st.markdown("Enter what ingredients you have and how much you're willing to spend. Get a recipe!")

ingredients = st.text_input("Ingredients you have (comma-separated):", placeholder="e.g. eggs, spinach, rice")
budget = st.slider("What’s your budget (in USD)?", min_value=5, max_value=100, step=1, value=20)

if st.button("Suggest a Recipe"):
    with st.spinner("Cooking up something delicious..."):
        result = recipe_chain.run({"ingredients": ingredients, "budget": budget})
        st.success("Here’s your dinner idea:")
        st.write(result)
