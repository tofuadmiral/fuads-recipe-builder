import os
import streamlit as st
import phoenix as px

from pandas import DataFrame as df

from phoenix.evals import run_evals
from phoenix.evals.models import OpenAIModel
from phoenix.evals.evaluators import LLMEvaluator
from phoenix.evals.evaluators import ToxicityEvaluator
from phoenix.evals.templates import ClassificationTemplate
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery

# --- Load secrets ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={st.secrets['PHOENIX_API_KEY']}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
os.environ["PHOENIX_PROJECT_NAME"] = "recipe-builder"


# --- Initialize client and model ---
judge_model = OpenAIModel(model="gpt-4.1")
client = px.Client()

# --- Get Evaluation Targets ---
query = SpanQuery().where("span_kind == 'LLM'").select(
    input="input.value",
    output="output.value"
)

spans_df = client.query_spans(query)

# --- Define custom evaluators without the need of references ----
factuality_template = ClassificationTemplate(
    rails=["1", "2", "3", "4", "5"],
    template=(
        "Rate the factual accuracy of the answer below using your general world knowledge. "
        "Give a score from 1 to 5, where:\n"
        "1 = Completely hallucinated (made-up)\n"
        "3 = Some factual content, some hallucinations\n"
        "5 = Fully accurate\n\n"
        "Question: {input}\n"
        "Answer: {output}\n\n"
        "Your rating (1–5):"
    )
)

relevance_template = ClassificationTemplate(
    rails=["1", "2", "3", "4", "5"],
    template=(
        "Rate how relevant the answer is to the question on a scale from 1 to 5, where:\n"
        "1 = Not relevant at all\n"
        "3 = Somewhat related but missing the point\n"
        "5 = Highly relevant and directly answers the question\n\n"
        "Question: {input}\n"
        "Answer: {output}\n\n"
        "Your rating (1–5):"
    )
)


# --- Instantiate evaluators ---
toxicity_eval = ToxicityEvaluator(model=judge_model)

factuality_eval = LLMEvaluator(
    model=judge_model,
    template=factuality_template
)

relevance_eval = LLMEvaluator(
    model=judge_model,
    template=relevance_template
)


# --- Run evals  ---
toxicity_eval_df, factuality_eval_df, relevance_eval_df = run_evals(
    dataframe=spans_df, evaluators=[toxicity_eval, factuality_eval, relevance_eval], provide_explanation=True)

# --- Log evals to Phoenix ---

client.log_evaluations(
    SpanEvaluations(
        dataframe=toxicity_eval_df,

        eval_name="Toxicity"
    ),
    SpanEvaluations(
        dataframe=factuality_eval_df,
        eval_name="Factuality 1-5"
    ),
        SpanEvaluations(
        dataframe=relevance_eval_df,
        eval_name="Relevance 1-5"
    )
)

print("Evaluation submitted to Phoenix.")