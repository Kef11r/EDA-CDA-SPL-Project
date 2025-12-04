from langchain_openai import ChatOpenAI
from stat_analyzer.config import OPENAI_API_KEY, OPENROUTER_URL

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENROUTER_URL,
    model="openai/gpt-5.1"
)
def ai_hypothesis_test(prompt: str):
    first_answer = llm.invoke(prompt).content
    follow_up_prompt = f"User hypothesis: {prompt}. Your previous answer: {first_answer}. User: Are you sure?"
    second_answer = llm.invoke(follow_up_prompt).content

    return first_answer, second_answer
