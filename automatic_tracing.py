import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from arize.otel import register
from openai import OpenAI
from anthropic import Anthropic
from typing import Union
from anthropic.types import ContentBlock, TextBlock
from dotenv import load_dotenv
import json
from openinference.semconv.trace import SpanAttributes

from openai.types.chat import ChatCompletionToolParam

# Load environment variables from .env file
load_dotenv()

# --- Set your Arize API key here ---
print("Environment variables:")
print(f"ARIZE_API_KEY: {os.environ.get('ARIZE_API_KEY')}")
print(f"ARIZE_AUTO_SPACE_ID: {os.environ.get('ARIZE_AUTO_SPACE_ID')}")
print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY')}")
print(f"ANTHROPIC_API_KEY: {os.environ.get('ANTHROPIC_API_KEY')}")

ARIZE_API_KEY = os.environ.get('ARIZE_API_KEY')
if not ARIZE_API_KEY:
    raise ValueError("ARIZE_API_KEY environment variable is not set")

ARIZE_AUTO_SPACE_ID = os.environ.get('ARIZE_AUTO_SPACE_ID')
if not ARIZE_AUTO_SPACE_ID:
    raise ValueError("ARIZE_AUTO_SPACE_ID environment variable is not set")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")


# --- Set up auto-instrumentation ---

# Setup OTel via our convenience function
tracer_provider = register(
    space_id = ARIZE_AUTO_SPACE_ID, # in app space settings page
    api_key = ARIZE_API_KEY, # in app space settings page
    project_name="Fuad's Test Project Auto Tracing",
)

# Import openinference instrumentor to map Anthropic traces to a standard format
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

# Setup automatic tracers

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


# --- Model Providers Setup ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

def call_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    # Get the current tracer
    tracer = trace.get_tracer(__name__)
    
    # Create a span for the entire OpenAI call
    with tracer.start_as_current_span("openai_call") as main_span:
        main_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        main_span.set_attribute("model", model)
        
        # Define the tool
        tools: list[ChatCompletionToolParam] = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Finds the weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'London'",
                        }
                    },
                    "required": ["city"],
                },
            },
        }]

        # First call with tool definition
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )

        message = response.choices[0].message

        # If there's a tool call, handle it
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            # Create a span for the tool call
            with tracer.start_as_current_span("tool_call.get_weather") as tool_span:
                # Link this span to the main span
                tool_span.add_link(main_span.get_span_context())
                tool_span.set_attribute("tool.name", tool_name)
                tool_span.set_attribute("tool.args", str(tool_args))
                tool_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "TOOL")
                
                # Simulate the tool's response 
                if tool_name == "get_weather":
                    # This OpenAI call will be auto-instrumented and show up as a separate span
                    city = json.loads(tool_args).get('city', 'London')
                    weather_prompt = f"What is the weather in {city}?"
                    weather_response = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": weather_prompt}]
                    )
                    tool_response = weather_response.choices[0].message.content or ""
                else:
                    tool_response = ""

                tool_span.set_attribute("tool.response", tool_response)

            # Add the tool response to the conversation
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_response
                }
            ]

            # Make the final call with the tool response
            final_response = openai_client.chat.completions.create(
                model=model,
                messages=messages
            )
            result = final_response.choices[0].message.content
        else:
            result = message.content

        assert result is not None, "OpenAI response content was None"
        return result

def call_anthropic(prompt: str, model: str = "claude-3-opus-20240229") -> str:
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    # Extract text from the first content block
    content_block = response.content[0]
    if isinstance(content_block, TextBlock):
        result = content_block.text
    else:
        result = str(content_block)
    assert result is not None, "Anthropic response content was None"
    return result

# Example usage:
if __name__ == "__main__":
    # test_prompt = "What's a quick dinner recipe using eggs and spinach?"
    test_tool_prompt = "What's the weather in London?"

    # Try OpenAI
    print("OpenAI Response:")
    openai_response = call_openai(test_tool_prompt)
    print(openai_response)

    # Try Anthropic
    print("\nAnthropic Response:")
    anthropic_response = call_anthropic(test_tool_prompt)
    print(anthropic_response)
