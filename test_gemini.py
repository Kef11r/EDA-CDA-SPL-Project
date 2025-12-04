from stat_analyzer.ai.ai_agent import ai_hypothesis_test

if __name__ == "__main__":
    print("Sending request to Gemini…")
    first, second = ai_hypothesis_test("I want to check if there is a relationship between Genre and Global_Sales.")
    print("\nFirst answer:")
    print(first)
    print("\nSecond answer:")
    print(second)
