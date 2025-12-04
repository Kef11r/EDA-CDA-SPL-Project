import os
from langchain_openai import ChatOpenAI
from stat_analyzer.config import OPENAI_API_KEY, OPENROUTER_URL

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENROUTER_URL,
    model="google/gemini-2.5-flash",
    max_tokens = 512,
    max_retries=0,
    timeout = 3
)
def ai_hypothesis_test(prompt: str):
    try:
        first_answer = llm.invoke(prompt).content
        follow_up_prompt = f"User hypothesis: {prompt}. Your previous answer: {first_answer}. User: Are you sure?"
        second_answer = llm.invoke(follow_up_prompt).content

        return first_answer, second_answer
    except:
        raise RuntimeError(f"Gemini app error")
