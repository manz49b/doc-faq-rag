import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def call_openai_gpt(system_prompt, prompt):
    """
    Call the OpenAI GPT API with a given prompt and return the generated message.

    Args:
        system_prompt (str): The system prompt or initial instructions for GPT.
        prompt (str): The user prompt or main question for GPT.
        api_key (str): The API key for the OpenAI API.
    
    Returns:
        str: The generated message from GPT.
    """
    try:
        messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Replace with model of choice, such as gpt-3.5-turbo or gpt-4
            messages=messages,
            max_tokens=2048,  # Adjust as needed
            temperature=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return '{{"answer" = ["error"]}}'