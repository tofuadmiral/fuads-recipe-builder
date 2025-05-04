import streamlit as st
import openai
import time
import os

# Configure OpenAI client (v1.x syntax)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

start = time.time()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Give me a 1-sentence dinner recipe using eggs and spinach."}
    ],
    max_tokens=100,
    temperature=0.7,
    timeout=15  # renamed from request_timeout
)

end = time.time()

# Show results
st.markdown("### ✅ GPT-4o-mini Test Complete")
st.write(f"**Model:** `{response.model}`")
st.write(f"**⏱️ Response time:** `{round(end - start, 2)} seconds`")
st.code(response.choices[0].message.content)
