import streamlit as st
import ollama
import chromadb
from pipeline import process_playlist, get_collection_name

st.set_page_config(page_title="Course RAG Assistant", page_icon="📚")
st.title("📚 Course RAG Assistant")
st.caption("Paste any YouTube course playlist, then ask questions and get answers with clickable video timestamps")

client = chromadb.PersistentClient(path="./chroma_db")

# Keep track of chat history and which collection is active
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

# --- SIDEBAR: playlist management ---
with st.sidebar:
    st.header("📂 Playlists")

    # List already-processed playlists (existing ChromaDB collections)
    existing_collections = [c.name for c in client.list_collections()]

    if existing_collections:
        st.subheader("Switch to a processed playlist")
        selected = st.selectbox("Choose:", ["-- select --"] + existing_collections)
        if selected != "-- select --" and st.button("Load this playlist"):
            st.session_state.active_collection = selected
            st.session_state.messages = []  # reset chat when switching
            st.success(f"Loaded: {selected}")

    st.divider()
    st.subheader("Add a new playlist")
    new_url = st.text_input("YouTube playlist URL")
    max_videos = st.number_input("Max videos to process (start small for testing)", min_value=1, max_value=100, value=10)

    if st.button("Process this playlist"):
        if not new_url.strip():
            st.error("Please paste a playlist URL first")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, current, total):
                status_text.text(message)
                if total > 0:
                    progress_bar.progress(min(current / total, 1.0))

            with st.spinner("Processing playlist... this will take a few minutes"):
                result = process_playlist(new_url, max_videos=int(max_videos), progress_callback=update_progress)

            if result['num_chunks'] == 0:
                st.error(f"No usable content found. {result['num_skipped']} video(s) were skipped (likely no transcripts available in supported languages).")
            else:
                st.success(f"Done! Processed {result['num_videos']} videos, {result['num_chunks']} chunks ready.")
                if result['num_skipped'] > 0:
                    st.warning(f"Note: {result['num_skipped']} video(s) were skipped (no transcript available).")
                st.session_state.active_collection = result["collection_name"]
                st.session_state.messages = []

    if st.session_state.active_collection:
        st.divider()
        st.write(f"**Active:** {st.session_state.active_collection}")

# --- MAIN AREA: chat ---
if not st.session_state.active_collection:
    st.info("👈 Add a playlist or select an existing one from the sidebar to get started.")
else:
    collection = client.get_or_create_collection(name=st.session_state.active_collection)

    def ask_question(question):
        response = ollama.embed(model="nomic-embed-text", input=question)
        question_embedding = response["embeddings"][0]

        results = collection.query(query_embeddings=[question_embedding], n_results=3)
        retrieved_chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if distances[0] > 1.3:
            return "This doesn't seem to be covered in the course material I have.", []

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have enough information about this."

Context:
{context}

Question: {question}

Answer:"""

        answer = ollama.generate(model="llama3.1:8b", prompt=prompt)
        return answer["response"], metadatas

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources:**")
                for meta in msg["sources"]:
                    minutes = int(meta["start"] // 60)
                    seconds = int(meta["start"] % 60)
                    url = f"https://youtube.com/watch?v={meta['video_id']}&t={int(meta['start'])}s"
                    st.markdown(f"- [{meta['video_title']} at {minutes}:{seconds:02d}]({url})")

    question = st.chat_input("Ask something about this course...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_text, metadatas = ask_question(question)
            st.write(answer_text)
            if metadatas:
                st.markdown("**Sources:**")
                for meta in metadatas:
                    minutes = int(meta["start"] // 60)
                    seconds = int(meta["start"] % 60)
                    url = f"https://youtube.com/watch?v={meta['video_id']}&t={int(meta['start'])}s"
                    st.markdown(f"- [{meta['video_title']} at {minutes}:{seconds:02d}]({url})")

        st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": metadatas})