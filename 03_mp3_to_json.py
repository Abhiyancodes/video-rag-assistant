import whisper
import json
import os

model = whisper.load_model('large-v2')

audios = os.listdir('audios')

for audio in audios:
    if audio.lower().endswith(('.mp3','.wav','.m4a','.flac')):
        filename, ext = os.path.splitext(audio)

        if('_' in filename):
            number, title = filename.split('_', 1)

            output_path = os.path.join('oldjsons',f'{filename}.json')

            if os.path.exists(output_path):
                print(f"{filename} already processed.")
                continue
            
            print(audio)
            result = model.transcribe(audio = f'audios/{audio}',
                                    language='hi',
                                    task='translate',
                                    word_timestamps=False)
            
            chunks = []
            for segment in result['segments']:
                chunks.append({"number": number, "title": title ,"start": segment['start'], "end": segment['end'], "text": segment['text']})
            
            chunks_with_metadata = {"chunks" : chunks, "text" : result['text']}

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)
