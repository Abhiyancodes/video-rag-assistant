# Video RAG Assistant

A Retrieval-Augmented Generation (RAG) system that turns a folder of course video lectures into a queryable Q&A assistant. Ask a question in plain English, and it tells you *what's taught, in which video, and at what timestamp*, so you can jump straight to the right part of the right lecture instead of scrubbing through hours of footage.

Built as a learning project to understand the full RAG pipeline end-to-end: audio extraction, transcription, chunking, embeddings, similarity search, and LLM-based generation.

---

## How it works

```
videos/ (.mp4/.mkv)
      │  ffmpeg
      ▼
audios/ (.mp3)
      │  Whisper (large-v2, Hindi → English translation)
      ▼
oldjsons/ (raw per-segment transcripts, timestamped)
      │  merge into larger chunks 
      ▼
jsons/ (merged, LLM-friendly chunks)
      │  Ollama embeddings (bge-m3)
      ▼
embeddings.joblib (pandas DataFrame: text + metadata + embedding vector)
      │  cosine similarity search against a live question
      ▼
Top-5 relevant chunks → prompt → Gemini → answer with video + timestamp
```

Each stage is a standalone script, run once in sequence to build the knowledge base, with one final script (`06_process_incoming.py`) you run interactively to ask questions.

---

## Tech stack

| Tool | Role | Why |
|---|---|---|
| **ffmpeg** | Video → audio extraction | Standard, reliable, handles any video container format |
| **Whisper (large-v2)** | Speech-to-text + translation | Runs locally, no per-minute transcription cost, handles Hindi→English translation in one pass |
| **Ollama + bge-m3** | Text embeddings | Free, runs fully locally — no API cost for embedding potentially thousands of chunks |
| **pandas + joblib** | Storage | Simple, transparent storage for a dataset this size — no need for a vector DB yet |
| **scikit-learn** | Cosine similarity | Standard, well-tested implementation; keeps the retrieval math visible instead of hidden inside a framework |
| **Gemini API** | Answer generation | Only the final generation step needs a hosted LLM — everything upstream is free and local (you can also use Ollama's open source models here, but they were less reliable and accurate for this use case) |

---

## Setup

**1. Install dependencies**
```bash
pip install openai-whisper pandas numpy scikit-learn joblib requests python-dotenv google-genai
```
You'll also need `ffmpeg` installed and available on your PATH, and [Ollama](https://ollama.com) installed and running.

**2. Pull the embedding model**
```bash
ollama pull bge-m3
```

**3. Add your Gemini API key**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

**4. Add your videos**

Drop your video files into the `videos/` folder. Filenames are expected to contain a tutorial number, e.g. `Tutorial #12 (Some Title).mp4`.

**5. Run the pipeline, in order**
```bash
python 02_video_to_mp3.py        # videos -> audios
python 03_mp3_to_json.py         # audios -> oldjsons (raw transcripts)
python 04_merge_chunks.py        # oldjsons -> jsons (merged chunks)
python 05_preprocessed_json.py   # jsons -> embeddings.joblib
python 06_process_incoming.py    # ask questions interactively
```

Steps 2–5 are one-time (or incremental) setup, they build your searchable knowledge base. Step 6 is what you actually run to ask questions once the knowledge base exists.

---

## Example

```
Ask a Question: How do I center a div using flexbox?
```

The system retrieves the 5 most relevant transcript chunks (by cosine similarity between the question's embedding and every chunk's embedding), then passes them to Gemini along with the original question. A sample response:

> To center a div using Flexbox, you'll want to check out **Tutorial #14 (Flexbox Basics)** around **3:20–4:05**, where the instructor covers `justify-content: center` and `align-items: center` on the parent container. There's also a follow-up example in **Tutorial #14** around **6:40** showing centering both horizontally and vertically together.

---

## Design decisions

**Chunking strategy.** Whisper produces short, sentence-level segments — too granular to give an LLM useful context on their own. Chunks are merged based on a character budget (rather than a fixed count of segments) so that each merged chunk stays a consistent, meaningful size regardless of how densely or sparsely the speaker talked in that section.

**Timestamps precomputed, not generated.** Rather than asking the LLM to convert raw seconds into minutes:seconds in its response (which LLMs are inconsistent at), timestamps are formatted in Python before being handed to the model — the LLM only has to relay a value that's already correct, never compute it.

**Cosine similarity over a flat DataFrame.** With a course-sized dataset (thousands, not millions, of chunks), an in-memory DataFrame plus `sklearn.cosine_similarity` is simple, fully transparent, and fast enough — no need for a dedicated vector database at this scale.

**Local embeddings, hosted generation only.** Embeddings are the expensive, repeated part of a RAG pipeline (every chunk needs one). Running that locally via Ollama keeps the whole indexing process free. Only the final, one-shot generation step calls a hosted API.

**Idempotent pipeline stages.** Each script checks whether its output already exists before reprocessing a file, so re-running the pipeline after adding a few new videos doesn't re-transcribe or re-embed everything from scratch.

**Iterative additions based on output quality.** The pipeline didn't start with merged chunks or Gemini — both were added later, after seeing the initial results.
- Early versions fed the raw, sentence-level chunks from Whisper directly into the LLM. These were too short and fragmented to give good context, so the `04_merge_chunks.py` step was added afterward to group chunks into larger, more meaningful pieces before embedding.
- The generation step originally ran on a local Ollama model. Responses were noticeably less reliable and less accurate than a hosted model, so generation was switched to the Gemini API, while embeddings stayed on Ollama locally since that step didn't show the same quality gap.

---

## Limitations & possible improvements

- **No similarity threshold** — the top 5 chunks are always retrieved and sent to the LLM, even for questions unrelated to the course content. A minimum similarity cutoff would let the system recognize out-of-scope questions before generation, rather than relying on prompt instructions alone.
- **Linear similarity search** — cosine similarity is computed against every chunk on every query. Fine at this scale; a proper vector index (e.g. FAISS, Chroma) would be a natural next step if the dataset grew much larger.
- **No reranking** — retrieval is single-pass cosine similarity with no reranking step to further refine the top results.
- **Basic filename parsing** — video number/title extraction from filenames relies on a specific naming convention and isn't robust to arbitrary filenames.
- **Whisper transcription is slow without a GPU** — `large-v2` is the most accurate Whisper model, but it's also the slowest, and transcribing even a single hour of video can take a long time on a CPU-only machine or a laptop without a dedicated GPU. A few ways to speed this up:
  - Use a smaller Whisper model (`medium`, `small`, or `base`) — faster, at some cost to transcription accuracy.
  - Run on a machine with a dedicated NVIDIA GPU (CUDA) — Whisper is dramatically faster on GPU than CPU.
  - Use [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a reimplementation of Whisper using CTranslate2, which can be several times faster than the original implementation at the same accuracy.
  - Transcribe in the background/overnight as a one-time batch job — since 03_mp3_to_json.py already skips files that are already processed, this only needs to be slow once per video, not on every run.

---

## Project structure

```
.
├── videos/              # input video files
├── audios/               # extracted mp3s
├── oldjsons/             # raw per-segment transcripts
├── jsons/                # merged chunks
├── embeddings.joblib      # final DataFrame: chunks + embeddings
├── 02_video_to_mp3.py
├── 03_mp3_to_json.py
├── 04_merge_chunks.py
├── 05_preprocessed_json.py
├── 06_process_incoming.py
└── .env
```
