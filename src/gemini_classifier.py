'''
Passes data into the Gemini API to classify text. It is free to use as of 30 may 2024.

Ensure the API key is generated from the Gemini website as stated in the code documentation.
The API key is stored in a .env file with variable name "gemini_api_key", at the root folder.
'''

import pandas as pd
import os
import requests
import json
import base64
import hmac
import hashlib
import time
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize

load_dotenv()

def count_tokens(text):
    tokens = word_tokenize(text)
    token_count = len(tokens)
    return tokens, token_count

df = pd.read_excel('data/results/mitre-attack-framework.xlsx', sheet_name='Threats')
df["NameDesc"] = df["Name"] + " " + df["Desc"]

gemini_api_key = os.getenv("gemini_api_key")
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-pro", safety_settings=safety_settings)

instruction = '''The STRIDE framework stands for Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege.
Each letter in the STRIDE framework represents a category of a threat model. The categories are defined as follows:
Spoofing: Spoofing consists of pretending to be someone or something else. An attacker would use the ability to ‘be someone or something else’ to perform malicious actions that they should not be capable of, or as a stepping stone for further attack.
Tampering: Tampering consists of tampering with (or modifying) data when this should not be possible. This impacts the integrity of data.
Repudiation: Associated with users who deny performing an action without other parties having any way to prove otherwise—for example, a user performs an illegal operation in a system that lacks the ability to trace the prohibited operations. Non-Repudiation refers to the ability of a system to counter repudiation threats. For example, a user who purchases an item might have to sign for the item upon receipt. The vendor can then use the signed receipt as evidence that the user did receive the package.
Information Disclosure: Information Disclosure consists of gaining access to confidential information when this should not be possible.
Denial of Service: Denial of Service consists of disabling a system from working and thus making it unavailable to legitimate users.
Elevation of Privilege: Elevation of privilege consists of gaining (higher) access privileges and using those privileges to perform unauthorized actions.
I will give a sentence, reply only with '0', '1', '2', '3', '4', '5' where each number corresponds to the STRIDE category.
'''

predictions = []
requests_count = 0

st = time.time()
for i in range(0, len(df)):
    text_to_send = f"{instruction} The sentence: {df.loc[i, 'NameDesc']}"
    response = model.generate_content(text_to_send)
    df.loc[i, 'predicted'] = response.text.strip()

    print(f'Row {i+1} with prediction: {df.loc[i, "predicted"]}')
    
    requests_count += 1
    if requests_count >= 60:
        et = time.time()
        elapsed_time = et - st
        if elapsed_time < 60:
            time.sleep(70 - elapsed_time)
        start_time = time.time()
        requests_count = 0
    
df.to_excel('data/results/mitre-predictions.xlsx', index=False)
print(f'Total execution time: {(time.time() - st):.3f} seconds')
