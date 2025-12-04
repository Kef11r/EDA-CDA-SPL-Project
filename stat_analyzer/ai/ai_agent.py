from openai import OpenAI
from stat_analyzer.config import OPENROUTER_URL, OPENAI_API_KEY

client = OpenAI(
    api_key = OPENAI_API_KEY,
    base_url = OPENROUTER_URL
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

