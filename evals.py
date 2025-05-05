import os
import streamlit as st
import phoenix as px

from pandas import DataFrame as df

from phoenix.evals import run_evals
from phoenix.evals.models import OpenAIModel
from phoenix.evals.evaluators import LLMEvaluator, HallucinationEvaluator, RelevanceEvaluator
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery

# --- Load secrets ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={st.secrets['PHOENIX_API_KEY']}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
os.environ["PHOENIX_PROJECT_NAME"] = "recipe-builder"


# --- Initialize client and model ---
judge_model = OpenAIModel(model="gpt-4o")
client = px.Client()

# --- Get Evaluation Targets ---
query = SpanQuery().where("span_kind == 'LLM'").select(
    input="input.value",
    output="output.value"
)

spans_df = client.query_spans(query)

# --- Define criteria for evaluation ---
hallucination_eval = HallucinationEvaluator(model=judge_model)
relevance_eval = RelevanceEvaluator(model=judge_model)

# --- Run evals  ---
hallucination_eval_df, relevance_eval_df = run_evals(
    dataframe=spans_df, evaluators=[hallucination_eval, relevance_eval], provide_explanation=True)

# --- Log evals to Phoenix ---

client.log_evaluations(
    SpanEvaluations(
        dataframe=hallucination_eval_df,
        eval_name="Hallucination"
    ),
    SpanEvaluations(
        dataframe=relevance_eval_df,
        eval_name="Relevance"
    ),
)

print("Evaluation submitted to Phoenix.")