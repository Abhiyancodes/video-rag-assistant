import requests
import os
import json
import pandas as pd
import numpy as np
import joblib

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
        # print(embedding)  # list of lists 
        all_embeddings.extend(embedding)  # extends insted of apending list after list it flattens along the list
        # print(all_embeddings) # list of lists
    return all_embeddings

jsons = os.listdir("jsons")
my_chunks = []
chunk_id = 0

for json_file in jsons: #load the jsonfile 
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    
        
    print(f"Creating embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    # print(f"content : {content}")  # Dictionary {"chunks" : [{'number' : 1,}], "text":[' ']}
    # print(embeddings) # list of lists

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_chunks.append(chunk)  # my_chunks is a list[{dictionary}]

# print(content)
df = pd.DataFrame.from_records(my_chunks)
# Saving this dataframe
joblib.dump(df, 'embeddings.joblib')

                                