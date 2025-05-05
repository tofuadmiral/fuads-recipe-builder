import os
import streamlit as st
import phoenix as px

from pandas import DataFrame as df

from phoenix.evals import run_evals
from phoenix.evals.models import OpenAIModel
from phoenix.evals.evaluators import LLMEvaluator, HallucinationEvaluator, RelevanceEvaluator
from phoenix.trace import SpanEvaluations

# --- Load secrets ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PHOENIX_API_KEY"] = st.secrets["PHOENIX_API_KEY"]
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"


# --- Evaluation model ---
judge_model = OpenAIModel(model="gpt-4o")

# --- Define criteria ---
helpfulness_eval = LLMEvaluator(name="Helpfulness", model=judge_model)
hallucination_eval = HallucinationEvaluator(name="Hallucination", model=judge_model)
relevance_eval = RelevanceEvaluator(name="Relevance", model=judge_model)

# --- Run evals  ---
helpfulness_eval_df, hallucination_eval_df, relevance_eval_df = run_evals(
    dataframe=df, evaluators=[helpfulness_eval, hallucination_eval, relevance_eval], provide_explanation=True)

# --- Log evals to Phoenix ---0

client = px.Client()

client.log_evaluations(
    SpanEvaluations(
        dataframe=helpfulness_eval_df,
        eval_name="Helpfulness"
    ),
    SpanEvaluations(
        dataframe=hallucination_eval_df,
        eval_name="Hallucination"
    ),
    SpanEvaluations(
        dataframe=relevance_eval_df,
        eval_name="Relevance"
    ),
)

print("✅ Evaluation submitted to Phoenix.")