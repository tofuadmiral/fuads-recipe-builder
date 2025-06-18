if __name__ == "__main__":
    client = openai.OpenAI()
    messages = [
        ChatCompletionUserMessageParam(
            role="user",
            content="What's the weather like in San Francisco?",
        )
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "finds the weather for a given city",
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
            },
        ],
        messages=messages,
    )
    message = response.choices[0].message
    assert (tool_calls := message.tool_calls)
    tool_call_id = tool_calls[0].id
    messages.append(message)
    messages.append(
        ChatCompletionToolMessageParam(content="sunny", role="tool", tool_call_id=tool_call_id),
    )
    client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
    )
    print(response.usage)
    print(response)