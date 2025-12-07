import os
from typing import Dict, List
from langchain_openai import ChatOpenAI
from stat_analyzer.config import GEMINI_API_KEY, OPENROUTER_URL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=OPENROUTER_URL,
    model="openai/gpt-oss-20b:free",
    max_tokens = 512,
    max_retries=0,
    timeout = 3
)
def ai_hypothesis_test(prompt: str):
    """Take description of the hypothesis and return 1 ender AI response, if anything gone wrong - raise RunTime Error"""
    prompt = prompt.strip()
    try:
        first_answer = llm.invoke(prompt).content
        return first_answer
    except:
        raise RuntimeError(f"Gemini app error")

json_parser = JsonOutputParser()
prompt_template = ChatPromptTemplate.from_template("""
Ти статистичний консультант.
Користувач формулює гіпотезу українською або англійською, а також має перелік доступних статистичних тестів.
Гіпотеза: {hypothesis}
Доступні тести: {available_tests}
Завдання:
0. Розпиши як колонки залежать між собою у датасеті і якусь коротку інформацію.
1. Обери один або кілька тестів які найкраще підходять до цієї гіпотези.
2. Обирай тільки з доступних тестів, не вигадуй нові назви.
3. Коротко поясни українською чому ці тести підходять.

Поверни відповідь тільки у вигляді валідного JSON з полями:
- "recommended_tests"  список рядків  назв тестів
- "explanation"  рядок з коротким поясненням
{format_instructions}""")
# Ланцюжок prompt -> llm -> JSON парсер
recommend_chain = prompt_template | llm | json_parser

def recommend_tests_from_hypothesis(
    hypothesis: str,
    available_tests: List[str],) -> Dict[str, object]:
    """
    Викликає LLM і повертає структуру:
    {
        "recommended_tests": [...],
        "explanation": "...",
        "columns_comment": "..."
    }
    """
    if not available_tests:
        return {
            "recommended_tests": [],
            "explanation": "Немає доступних тестів для цієї пари змінних.",
            "columns_comment": ""
        }
    try:
        result = recommend_chain.invoke(
            {
                "hypothesis": hypothesis,
                "available_tests": ", ".join(available_tests),
                "format_instructions": json_parser.get_format_instructions(),
            }
        )
    except Exception as e:
        return {
            "recommended_tests": [],
            "explanation": f"АІ не зміг сформувати рекомендацію. Помилка {e}",
            "columns_comment": ""
        }
    rec = result.get("recommended_tests") or []
    rec = [t for t in rec if t in available_tests]
    return {
        "recommended_tests": rec,
        "explanation": (result.get("explanation") or "").strip(),
        "columns_comment": (result.get("columns_comment") or "").strip()
    }