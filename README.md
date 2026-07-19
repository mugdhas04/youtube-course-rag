\# 📚 Course RAG Assistant



A Retrieval-Augmented Generation (RAG) system that turns any YouTube course playlist into an interactive Q\&A assistant — ask questions in natural language and get answers grounded in the actual course content, with clickable timestamps linking straight to the relevant moment in the video.



Built and tested on \[CodeWithHarry's DSA course](https://youtube.com/playlist?list=PLu0W\_9lII9ahIappRPN0MCAgtOu3lQjQi) (Hindi, 92 videos).



\## ✨ Features



\- \*\*Ask questions about any YouTube course\*\* — paste a playlist URL, the system fetches transcripts, and you can start chatting once processing completes

\- \*\*Source-grounded answers\*\* — every answer is generated only from retrieved course content, not the LLM's general knowledge

\- \*\*Clickable timestamps\*\* — every answer links directly to the exact moment in the source video

\- \*\*Multi-playlist support\*\* — process and switch between multiple courses without reprocessing

\- \*\*Off-topic detection\*\* — flags when a question isn't covered in the loaded course material, instead of hallucinating an answer

\- \*\*100% local \& free\*\* — runs entirely on your machine using Ollama, no API keys or cloud costs



\## 🏗️ How it works



This is a standard RAG (Retrieval-Augmented Generation) pipeline:



1\. \*\*Ingestion\*\* — YouTube transcripts are fetched (`youtube-transcript-api`) and split into \~400-character chunks, each tagged with its source video and timestamp

2\. \*\*Embedding\*\* — Each chunk is converted into a vector embedding using Ollama's `nomic-embed-text` model, capturing its semantic meaning

3\. \*\*Storage\*\* — Embeddings are stored in a local ChromaDB vector database, one collection per playlist

4\. \*\*Retrieval\*\* — When a question is asked, it's embedded the same way, and ChromaDB returns the most semantically similar chunks

5\. \*\*Generation\*\* — Retrieved chunks are passed as context to `llama3.1:8b` (via Ollama), which generates an answer grounded strictly in that context



\## 🛠️ Tech Stack



| Component | Tool |

|---|---|

| LLM | Ollama (`llama3.1:8b`) |

| Embeddings | Ollama (`nomic-embed-text`) |

| Vector Database | ChromaDB |

| Transcript Fetching | youtube-transcript-api, yt-dlp |

| UI | Streamlit |



\## 🚀 Setup



\*\*Prerequisites:\*\* Python 3.10+, \[Ollama](https://ollama.com/download) installed



1\. Clone this repository



git clone https://github.com/mugdhas04/youtube-course-rag

cd youtube-rag-project



2\. Create a virtual environment and install dependencies



python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt



3\. Pull the required Ollama models



ollama pull llama3.1:8b

ollama pull nomic-embed-text



4\. Run the app



streamlit run app.py



5\. Paste a YouTube playlist URL in the sidebar, click \*\*Process this playlist\*\*, and start asking questions once it's done.



\## 📊 Evaluation



To validate retrieval quality, 10 test questions were manually created, each paired with the video topic expected to contain the answer. Retrieval was evaluated using \*\*Hit Rate @ Top-3\*\* — whether the correct source video appeared among the top 3 retrieved chunks.



\*\*Result: 10/10 (100%) on the DSA course dataset (20 videos, 1,101 chunks)\*\*



| Question | Expected Topic | Hit? |

|---|---|---|

| What is time complexity? | Time Complexity | ✓ |

| What is Big O notation? | Big O | ✓ |

| How do you insert an element in an array? | Insertion | ✓ |

| What is a linked list? | Linked List | ✓ |

| Difference between linear and binary search? | Search | ✓ |

| How to delete a node from a linked list? | Deletion | ✓ |

| What is asymptotic notation? | Asymptotic Notation | ✓ |

| What is a circular linked list? | Circular Linked List | ✓ |

| Best case and worst case analysis? | Best/Worst Case | ✓ |

| What is an abstract data type? | Abstract Data Type | ✓ |



\*Note: this result reflects a well-scoped dataset with clearly distinct topics per video. Retrieval accuracy is expected to be lower on larger, more topically overlapping datasets — a natural next step is expanding this evaluation to the full 92-video playlist.\*



Evaluation script: \[`evaluate.py`](evaluate.py)

\## 🔮 Future Improvements



\- Support for playlists without transcripts (via Whisper-based transcription)

\- Hybrid search (keyword + semantic) for improved retrieval on exact terms

\- Multi-language answer support beyond English



\## 📸 Demo



\*(Add screenshots here)\*

