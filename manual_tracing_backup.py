import os
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind
from arize.otel import register
from openai import OpenAI
from anthropic import Anthropic
from typing import Union
from anthropic.types import ContentBlock, TextBlock
from openinference.semconv.trace import (
    SpanAttributes,
    MessageAttributes,
)
from openai.types.chat import ChatCompletionToolParam
import json
import uuid
from openinference.instrumentation import using_attributes

# Load environment variables from .env file
load_dotenv()

# --- Set your Arize API key here ---
print("Environment variables:")
print(f"ARIZE_API_KEY: {os.environ.get('ARIZE_API_KEY')}")
print(f"ARIZE_SPACE_ID: {os.environ.get('ARIZE_SPACE_ID')}")
print(f"OPENAI_API_KEY: {os.environ.get('OPENAI_API_KEY')}")
print(f"ANTHROPIC_API_KEY: {os.environ.get('ANTHROPIC_API_KEY')}")

ARIZE_API_KEY = os.environ.get('ARIZE_API_KEY')
if not ARIZE_API_KEY:
    raise ValueError("ARIZE_API_KEY environment variable is not set")

ARIZE_SPACE_ID = os.environ.get('ARIZE_SPACE_ID')
if not ARIZE_SPACE_ID:
    raise ValueError("ARIZE_SPACE_ID environment variable is not set")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

tracer_provider = register(
    space_id = ARIZE_SPACE_ID,
    api_key = ARIZE_API_KEY,
    project_name = "Fuad's Test Project", # name this to whatever you would like
)

tracer = trace.get_tracer(__name__)

# --- Model Providers Setup ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# We know what the structure of our spans attributes needs to be, so we can define them here

# Open AI span attributes
def get_openai_span_attributes(model: str, prompt: str):
    return {
        # LLM attributes
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.LLM_PROVIDER: "openai",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
        SpanAttributes.LLM_SYSTEM: "openai",
        
        # Input/Output attributes (key for visibility)
        SpanAttributes.INPUT_VALUE: prompt,
        SpanAttributes.INPUT_MIME_TYPE: "text/plain",
        
        # Message attributes
        MessageAttributes.MESSAGE_ROLE: "user",
        MessageAttributes.MESSAGE_CONTENT: prompt,
        
        # OpenAI specific attributes
        "openai.api_base": "https://api.openai.com/v1",
        "openai.api_type": "open_ai",
        "openai.api_version": "2024-01-01",
        "openai.organization": "",
        "openai.user": "",
        
        # Request attributes
        "http.request.method": "POST",
        "http.request.header.content_type": "application/json",
        "http.request.header.authorization": "Bearer ***",
        "http.request.header.user_agent": "openai-python/1.76.2",
        
        # Response attributes (will be updated after response)
        "http.response.status_code": None,
        "http.response.header.content_type": None,
        
        # Status attributes
        "span.status": "pending",
        "span.status_code": None,
        "span.status_message": None,
        
        # LLM specific attributes
        "llm.request.model": model,
        "llm.request.temperature": 0.7,
        "llm.request.max_tokens": None,
        "llm.request.top_p": 1.0,
        "llm.request.frequency_penalty": 0.0,
        "llm.request.presence_penalty": 0.0,
        "llm.request.stop": None,
        "llm.request.n": 1,
        "llm.request.stream": False,
        "llm.request.logit_bias": None,
        "llm.request.user": None,
        "llm.request.logprobs": None,
        "llm.request.top_logprobs": None,
        "llm.request.response_format": None,
        "llm.request.seed": None,
        "llm.request.tools": None,
        "llm.request.tool_choice": None,
        "llm.request.functions": None,
        "llm.request.function_call": None,
    }

# Anthropic span attributes
def get_anthropic_span_attributes(model: str, prompt: str):
    return {
        # LLM attributes
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.LLM_PROVIDER: "anthropic",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
        SpanAttributes.LLM_SYSTEM: "anthropic",
        
        # Input/Output attributes (key for visibility)
        SpanAttributes.INPUT_VALUE: prompt,
        SpanAttributes.INPUT_MIME_TYPE: "text/plain",
        
        # Message attributes
        MessageAttributes.MESSAGE_ROLE: "user",
        MessageAttributes.MESSAGE_CONTENT: prompt,
        
        # Anthropic specific attributes
        "anthropic.api_base": "https://api.anthropic.com",
        "anthropic.api_version": "2023-06-01",
        "anthropic.user": "",
        
        # Request attributes
        "http.request.method": "POST",
        "http.request.header.content_type": "application/json",
        "http.request.header.authorization": "Bearer ***",
        "http.request.header.user_agent": "anthropic-python/0.54.0",
        "http.request.header.x-api-version": "2023-06-01",
        "http.request.header.anthropic-version": "2023-06-01",
        
        # Response attributes to be updated after response
        "http.response.status_code": None,
        "http.response.header.content_type": None,
        
        # Status attributes
        "span.status": "pending",
        "span.status_code": None,
        "span.status_message": None,
        
        # LLM specific attributes
        "llm.request.model": model,
        "llm.request.max_tokens": 1000,
        "llm.request.temperature": None,
        "llm.request.top_p": None,
        "llm.request.top_k": None,
        "llm.request.system": None,
        "llm.request.metadata": None,
        "llm.request.stop_sequences": None,
        "llm.request.stream": False,
        "llm.request.user": None,
    }

# Tool span attributes
def get_tool_span_attributes(tool_name: str, tool_args: str, tool_call_id: str):
    return {
        "tool.name": tool_name,
        "tool.args": str(tool_args),
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
        "tool.call_id": tool_call_id,
        "tool.function.name": tool_name,
        "tool.function.arguments": tool_args,
        # Add input/output for tool visibility
        SpanAttributes.INPUT_VALUE: tool_args,
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        # Status attributes
        "span.status": "pending",
        "span.status_code": None,
        "span.status_message": None,
    }

# Helper function to set span attributes in batch
def set_span_attributes_batch(span, attributes: dict):
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)

def call_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    with tracer.start_as_current_span("openai_call") as span:
        # Get initial span attributes
        span_attributes = get_openai_span_attributes(model, prompt)
        
        # Set all attributes in batch
        set_span_attributes_batch(span, span_attributes)

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

        # Update tools attribute and add LLM_INVOCATION_PARAMETERS
        invocation_params = {
            "model": model,
            "temperature": 0.7,
            "tools": tools
        }
        span.set_attribute("llm.request.tools", str(tools))
        span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(invocation_params))

        # First call with tool definition
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )

        message = response.choices[0].message

        # Update response attributes in batch
        response_attributes = {
            "http.response.status_code": 200,
            "http.response.header.content_type": "application/json",
            "llm.response.model": response.model,
            # Add OUTPUT_VALUE for visibility
            SpanAttributes.OUTPUT_VALUE: message.content or "",
            SpanAttributes.OUTPUT_MIME_TYPE: "text/plain",
            # Update status attributes
            "span.status": "success",
            "span.status_code": 200,
            "span.status_message": "Request completed successfully",
        }
        
        # Update usage attributes if available
        if response.usage:
            response_attributes.update({
                "llm.response.usage.prompt_tokens": response.usage.prompt_tokens,
                "llm.response.usage.completion_tokens": response.usage.completion_tokens,
                "llm.response.usage.total_tokens": response.usage.total_tokens,
            })
        
        set_span_attributes_batch(span, response_attributes)
        
        # Set OpenTelemetry status
        span.set_status(Status(StatusCode.OK, "Request completed successfully"))

        # Create a decision point span to show branching
        with tracer.start_as_current_span("llm_decision_point", kind=trace.SpanKind.INTERNAL) as decision_span:
            decision_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "chain")
            decision_span.set_attribute("decision.type", "tool_call_decision")
            decision_span.set_attribute("decision.description", "LLM decides whether to use tools or respond directly")
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments

                # Mark decision as tool call path
                decision_span.set_attribute("decision.result", "use_tool")
                decision_span.set_attribute("decision.tool_name", tool_name)

                # Trace the tool call in a child span
                if tool_name == "get_weather":
                    # Create a chain span for the tool execution sequence
                    with tracer.start_as_current_span("tool_chain.get_weather", kind=trace.SpanKind.INTERNAL) as chain_span:
                        # Set chain span attributes
                        chain_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "chain")
                        chain_span.set_attribute("chain.name", "weather_tool_chain")
                        chain_span.set_attribute("chain.description", "Chain for weather tool execution")
                        
                        # Get tool span attributes
                        tool_attributes = get_tool_span_attributes(tool_name, tool_args, tool_call_id)
                        
                        # Create the tool execution as part of the chain
                        with tracer.start_as_current_span("tool_execution.get_weather", kind=trace.SpanKind.INTERNAL) as tool_span:
                            # Set all tool attributes in batch
                            set_span_attributes_batch(tool_span, tool_attributes)
                            
                            # Simulate the tool's response by making a secondary OpenAI call
                            city = json.loads(tool_args).get('city', 'London')
                            weather_prompt = f"What is the weather in {city}?"
                            
                            # This secondary OpenAI call will be auto-instrumented and show up as a separate span
                            weather_response = openai_client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": weather_prompt}]
                            )
                            tool_response = weather_response.choices[0].message.content or ""
                            
                            tool_span.set_attribute("tool.response", tool_response)
                            # Add output value for tool visibility
                            tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, tool_response)
                            tool_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
                            # Update tool status
                            tool_span.set_attribute("span.status", "success")
                            tool_span.set_attribute("span.status_code", 200)
                            tool_span.set_attribute("span.status_message", "Tool executed successfully")
                            # Set OpenTelemetry status
                            tool_span.set_status(Status(StatusCode.OK, "Tool executed successfully"))
                            
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

                            # Make the final call with the tool response as a child of the tool span
                            with tracer.start_as_current_span("final_llm_call", kind=trace.SpanKind.INTERNAL) as final_span:
                                final_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                                final_span.set_attribute("llm.request.model", model)
                                final_span.set_attribute("llm.request.temperature", 0.7)
                                final_span.set_attribute("chain.step", "final_response")
                                
                                final_response = openai_client.chat.completions.create(
                                    model=model,
                                    messages=messages
                                )
                                result = final_response.choices[0].message.content
                                
                                # Set final span attributes
                                final_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result or "")
                                final_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
                                final_span.set_attribute("span.status", "success")
                                final_span.set_status(Status(StatusCode.OK, "Final response generated"))
                        
                        # Set chain completion attributes
                        chain_span.set_attribute("chain.completed", True)
                        chain_span.set_attribute("chain.steps", ["tool_execution", "final_llm_call"])
                        chain_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result or "")
                        chain_span.set_status(Status(StatusCode.OK, "Tool chain completed successfully"))
                else:
                    tool_response = ""
                    result = message.content
            else:
                # Mark decision as direct response path
                decision_span.set_attribute("decision.result", "direct_response")
                
                # No tool calls - direct response path
                # Create a separate span that's a sibling to the main LLM call, not a child
                with tracer.start_as_current_span("direct_chat_completion", kind=trace.SpanKind.INTERNAL) as direct_span:
                    direct_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
                    direct_span.set_attribute("chain.step", "direct_response")
                    direct_span.set_attribute("llm.request.model", model)
                    direct_span.set_attribute("llm.request.temperature", 0.7)
                    
                    result = message.content
                    
                    # Set direct response attributes
                    direct_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result or "")
                    direct_span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
                    direct_span.set_attribute("span.status", "success")
                    direct_span.set_status(Status(StatusCode.OK, "Direct response generated"))
            
            # Set decision completion
            decision_span.set_attribute("decision.completed", True)
            decision_span.set_attribute(SpanAttributes.OUTPUT_VALUE, result or "")
            decision_span.set_status(Status(StatusCode.OK, "Decision completed"))

        assert result is not None, "OpenAI response content was None"
        span.set_attribute(MessageAttributes.MESSAGE_ROLE, "assistant")
        span.set_attribute(MessageAttributes.MESSAGE_CONTENT, result)
        return result

def call_anthropic(prompt: str, model: str = "claude-3-opus-20240229") -> str:
    with tracer.start_as_current_span("anthropic_call") as span:
        # Get initial span attributes
        span_attributes = get_anthropic_span_attributes(model, prompt)
        
        # Set all attributes in batch
        set_span_attributes_batch(span, span_attributes)
        
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Update response attributes in batch
        response_attributes = {
            "http.response.status_code": 200,
            "http.response.header.content_type": "application/json",
            "llm.response.model": response.model,
            "llm.response.usage.input_tokens": response.usage.input_tokens,
            "llm.response.usage.output_tokens": response.usage.output_tokens,
            # Update status attributes
            "span.status": "success",
            "span.status_code": 200,
            "span.status_message": "Request completed successfully",
        }
        
        set_span_attributes_batch(span, response_attributes)
        
        # Set OpenTelemetry status
        span.set_status(Status(StatusCode.OK, "Request completed successfully"))
        
        content_block = response.content[0]
        if isinstance(content_block, TextBlock):
            result = content_block.text
        else:
            result = str(content_block)
        assert result is not None, "Anthropic response content was None"
        
        # Set response attributes including OUTPUT_VALUE
        span.set_attribute(MessageAttributes.MESSAGE_ROLE, "assistant")
        span.set_attribute(MessageAttributes.MESSAGE_CONTENT, result)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result)
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
        return result

# Example usage with sessions:
if __name__ == "__main__":
    # Create a session ID for this conversation
    session_id = str(uuid.uuid4())
    user_id = "test-user-123"
    
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    
    # Simulate a multi-turn conversation in the same session
    test_prompt = "What's a quick dinner recipe using eggs and spinach?"
    test_tool_prompt = "What's the weather in London?"
    follow_up_prompt = "Can you make that recipe vegetarian?"
    
    # Test 1: Direct response (no tool call) - Session 1
    print("\n=== Session 1: Direct Response ===")
    openai_response = call_openai_with_session(test_prompt, session_id, user_id)
    print(f"Response: {openai_response}")
    
    # Test 2: Tool call response - Same session
    print("\n=== Session 1: Tool Call Response ===")
    openai_tool_response = call_openai_with_session(test_tool_prompt, session_id, user_id)
    print(f"Response: {openai_tool_response}")
    
    # Test 3: Follow-up question - Same session
    print("\n=== Session 1: Follow-up Question ===")
    follow_up_response = call_openai_with_session(follow_up_prompt, session_id, user_id)
    print(f"Response: {follow_up_response}")
    
    # Test 4: New session with Anthropic
    new_session_id = str(uuid.uuid4())
    print(f"\nNew Session ID: {new_session_id}")
    print("\n=== Session 2: Anthropic Response ===")
    anthropic_response = call_anthropic_with_session(test_prompt, new_session_id, user_id)
    print(f"Response: {anthropic_response}")
