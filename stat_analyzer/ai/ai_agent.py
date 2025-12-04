from openai import OpenAI

client = OpenAI(
    OPENAI_API_KEY="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",
    OPENROUTER_API_KEY = "https://openrouter.ai/api/v1"
)
def ai_hypothesis_test(prompt: str, model: str = 'openai/gpt-5.1'):
    response = client.chat.completions.create(
        model = model,
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    first_answer = response.choices[0].message["content"]
    messages = [
        {
            "role": "user", "content": prompt
        },
        {
            "role": "assistant",
            "content": first_answer
        },
        {
            "role": "user",
            "content": "Are you sure?"
        }
    ]

    #2nd API call
    response2 = client.chat.completions.create(
        model = model,
        messages = messages
    )
    second_answer = response2.choices[0].message["content"]
    return first_answer, second_answer

