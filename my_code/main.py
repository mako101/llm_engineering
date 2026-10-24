import logging
from universal_client import UniversalLLMClient as ulc
from openrouter_client import OpenRouterLLMClient as orc

logging.basicConfig(level=logging.INFO)


# default_model = "gpt-4o-mini"

# client = ulc(model=default_model, ollama_local_api_key="ollama")

user_prompt = "give me some global news headlines"

# system_prompt = """
# You are a news reporter on respecatble global network. 
# Provide news headlines in a concise professional manner.
# """
# print("--------------------------------")

# response = client.chat(user_prompt=user_prompt, system_prompt=system_prompt)
# print(default_model, ": ", response)

# print("--------------------------------")

# model="gemini-3.5-flash-lite", # Google model
model="deepseek-r1:1.5b"     # local ollama model, takes about 45s to respond

system_prompt = """
You are a hipster DJ on a local university radio station. 
Provide a snarky, sarcastic report of the news.
"""
# response = client.chat(
#     user_prompt=user_prompt, 
#     system_prompt=system_prompt, 
#     model=model,
#     )
# print(model, ": ",  response)

model="anthropic/claude-haiku-4.5"
print("--------------------------------")
client = orc()
response = client.chat(user_prompt=user_prompt, 
    system_prompt=system_prompt, 
    model=model)
print(model, ": ",  response)
