import openai

# Hardcoded API Key (Replace with your actual key)
API_KEY = "sk-0f4c0436dead421ea790111dfbcdb847"
print(f"API Key: {API_KEY}")

if not API_KEY:
    raise ValueError("DeepSeek API key is missing. Please set it inside summarizer.py.")

def summarize_text(text):
    """
    Summarizes the given text using DeepSeek AI.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    try:
        # Initialize OpenAI (DeepSeek) Client with API Key
        openai.api_key = API_KEY  # Using openai directly

        response = openai.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Summarize this text in simple terms:"},
                {"role": "user", "content": text}
            ]
        )

        # Extract and return summary
        return response.choices[0].message["content"]

    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    sample_text = "Artificial intelligence is the simulation of human intelligence in machines..."
    summary = summarize_text(sample_text)
    print("Summary:", summary)
