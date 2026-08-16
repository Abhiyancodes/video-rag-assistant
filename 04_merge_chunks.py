# this file basically merges the chunks created from mp3 in file 03 so LLM can get more information (we merge 5 chunks (n=5))
# this was added at the end to get good and complete response from the LLM

import os
import json
import math

n = 5

for filename in os.listdir("oldjsons"):
    if filename.endswith(".json"):
        file_path = os.path.join("oldjsons",filename)
        
        output_path = os.path.join("jsons", filename)

        if os.path.exists(output_path):
            print(f"{filename} already exist.")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            new_chunks = []
            num_chunks = len(data['chunks']) 
            num_groups = math.ceil(num_chunks/n)

            for i in range(num_groups):
                start_idx = i*n
                end_idx = min((i+1)*n, num_chunks)

                chunk_group = data['chunks'][start_idx:end_idx]

                new_chunks.append({
                    "number": chunk_group[0]['number'],
                    "title": chunk_group[0]['title'],
                    "start": chunk_group[0]['start'],
                    "end": chunk_group[-1]['end'],
                    "text": " ".join(c['text'] for c in chunk_group)
                })

        # Save file without double .json
        os.makedirs("jsons", exist_ok=True)
        with open(os.path.join("jsons", filename), "w", encoding="utf-8") as json_file:
            json.dump({"chunks": new_chunks, "text": data['text']}, json_file, indent=4)
