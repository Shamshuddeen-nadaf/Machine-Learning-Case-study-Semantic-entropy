import google.generativeai as genai
from google.colab import userdata
import requests
import time
import re

try:
    gemini_api_key = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=gemini_api_key)
    print("API key configured successfully from Colab Secrets.")
except userdata.SecretNotFoundError:
    print("ERROR: GOOGLE_API_KEY not found in Colab Secrets. Please add it.")
    exit()
def generate_three_formatted_responses(prompt_template, user_prompt):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    try:
        api_response = model.generate_content(prompt_template)
        if not api_response.text:
            raise ValueError("No text content in the API response.")
        pattern = r"res(\d):\s*(.*?)(?=\s*res\d:|\s*$)"
        matches = re.findall(pattern, api_response.text, re.DOTALL)
        if not matches or len(matches) < 3:
            # Fallback if regex parsing fails or doesn't find all three responses
            print("Warning: Could not parse exactly three responses. Here is the raw output:")
            print(api_response.text)
            return [{"query": user_prompt, "response": "Parsing failed. Please check raw output."}]
        formatted_responses = []
        for match in matches:
            response_text = match[1].strip()
            formatted_responses.append({
                "query": user_prompt,
                "response": f"res{match[0]}: {response_text}"
            })
        return formatted_responses
    except Exception as e:
        return [{"query": user_prompt, "response": f"Failed to get responses: {e}"}]
# --- Main execution ---
user_prompt = input("Please enter your question: ")
prompt_with_formatting = (
    f"Provide three concise and different responses for the question: '{user_prompt}'. "
    "Start each response with exactly 'res1:', 'res2:', and 'res3:' on a new line. "
    "Here are the three responses:"
)
print("Generating responses...")
three_responses = generate_three_formatted_responses(prompt_with_formatting, user_prompt)
for res in three_responses:
    print(res)