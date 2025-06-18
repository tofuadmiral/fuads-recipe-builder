import os
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Load environment variables
load_dotenv()

# --- Arize Setup ---
space_id = os.environ.get("ARIZE_AUTO_SPACE_ID")
if not space_id:
    raise ValueError("ARIZE_AUTO_SPACE_ID environment variable is not set")
    
tracer_provider = register(
    space_id=space_id,
    project_name="auto-trace-langchain-demo",
    set_global_tracer_provider=True,
    batch=True
)

# Auto-instrument LangChain
LangChainInstrumentor().instrument()

# --- LLM Setup ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# --- Prompt Setup ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template(
        "What's the weather in {city}?"
    )
])

# --- Chain Setup ---
weather_chain = prompt | llm

def call_weather_chain(city):
    return weather_chain.invoke({"city": city})

if __name__ == "__main__":
    city = "London"
    print(f"Querying weather for: {city}")
    result = call_weather_chain(city)
    print("Result:", getattr(result, "content", result))
