from litellm import completion
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display
from pprint import pprint, pformat

def show_reponse_stats(response):

    print(f"Input tokens: {response.usage.prompt_tokens}")
    print(f"Output tokens: {response.usage.completion_tokens}")
    print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
    print(f"Total cost: {response._hidden_params["response_cost"]}")

hamlet_path = os.path.join(os.path.dirname(__file__), "..", "week2", "hamlet.txt")
with open(hamlet_path, "r", encoding="utf-8") as f:
    hamlet = f.read()

long_prompt = f"""

You are a helpful and reliable assistant whose primary responsibility is to analyse short pieces of text and provide clear, concise, and accurate responses. Your answers should always be written in plain English, using straightforward vocabulary that can be understood by someone without any specialist knowledge.
When responding to a question, carefully identify the main subject, determine what information is being requested, and avoid introducing unnecessary details or unrelated explanations. Maintain a friendly but professional tone throughout your response.
You should always prioritise correctness, clarity, and brevity. Do not include introductory phrases, disclaimers, or additional recommendations unless they are specifically requested. Do not use Markdown formatting, bullet points, or numbered lists.
Your task is to answer the following question in exactly one sentence.

use the complete text of the hamlet to answer the question:
{hamlet}
"""

# first question
question = "What happens when the ghost of Hamlet's father appears to him?"

messages = [
    {"role": "system", "content": long_prompt},
    {"role": "user", "content": question}
    ]

response = completion(model="openai/gpt-5.1", messages=messages)
reply = response.choices[0].message.content
print(reply)
show_reponse_stats(response)




# second question
question = question = "What happens when the ghost of Hamlet's father appears to him?"

messages = [
    {"role": "system", "content": long_prompt},
    {"role": "user", "content": question}
    ]


response = completion(model="openai/gpt-5.1", messages=messages)
reply = response.choices[0].message.content
print(reply)
show_reponse_stats(response)


# third question
question = "What happened to the King in the end?"

messages = [
    {"role": "system", "content": long_prompt},
    {"role": "user", "content": question}
    ]


response = completion(model="openai/gpt-5.1", messages=messages)
reply = response.choices[0].message.content
print(reply)
show_reponse_stats(response)


# Total cost: $0.063 initial vs $0.007 cached - 9x cheaper (with gpt-5.1)