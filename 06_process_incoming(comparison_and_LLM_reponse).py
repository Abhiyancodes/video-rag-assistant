import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def create_embedding(text_list, batch_size=100):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        # print(batch)  # list of texts
        try:
            r = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "bge-m3", "input": batch}
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not reach Ollama at localhost:11434 — is it running? "
                "Start it with `ollama serve` and make sure `bge-m3` is pulled."
            )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama embedding request failed ({r.status_code}): {r.text}")
                               
        embedding = r.json()['embeddings']
        all_embeddings.extend(embedding)
    return all_embeddings

def inference_gemini(prompt, model="gemini-3.1-flash-lite"):
    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

df = joblib.load('embeddings.joblib')

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities of question embedding with other emeddings (df['embeddings'])
similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
top_reults = 5
max_indx = similarities.argsort()[::-1][0:top_reults]

new_df = df.loc[max_indx].copy()
new_df['start'] = new_df['start'].apply(format_timestamp)
new_df['end'] = new_df['end'].apply(format_timestamp)

prompt = f'''I am learning web development using Sigma web development course. Here are video subtitle chunks containing video title, video number, start time (mm:ss), end time (mm:ss), the text at that time:

{new_df[['title', 'number', 'start', 'end', 'text']].to_json(orient='records')}
----------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, its just for you) where and how much content is taught in which video (also at what timestamp) and guide the user to go to that particular video. If user asks unrelated questions, tell him that you can only answer questions related to the course (just give me the answer without asking any other questions in the end).
'''

with open('prompt.txt','w') as f:
    f.write(prompt)

response = inference_gemini(prompt)
print("Thinking...")

if response:
    with open("response.txt", 'w', encoding="utf-8") as f:
        f.write(response)
    print(response)
else:
    print("Failed to get a response from Gemini.")
     
# for index,item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item    ['start'], item['end'])
