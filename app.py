import streamlit as st
import ollama
import chromadb
from pipeline import process_playlist, get_collection_name

st.set_page_config(page_title="Course RAG Assistant", page_icon="📚", layout="centered")

# --- Custom styling ---
st.markdown("""
<style>
    .source-box {
        background-color: rgba(128, 128, 128, 0.1);
        border-left: 3px solid #4CAF50;
        padding: 10px 15px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .source-title {
        font-weight: 600;
        font-size: 0.9em;
    }
    .source-snippet {
        font-size: 0.85em;
        opacity: 0.8;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Course RAG Assistant")
st.caption("Paste any YouTube course playlist, then ask questions and get answers with clickable video timestamps")

# --- Check Ollama is reachable before doing anything else ---
def check_ollama():
    try:
        ollama.list()
        return True
    except Exception:
        return False

if not check_ollama():
    st.error("⚠️ Can't connect to Ollama. Make sure Ollama is running on your machine, then refresh this page.")
    st.stop()

client = chromadb.PersistentClient(path="./chroma_db")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

def render_sources(metadatas):
    st.markdown("**📌 Sources:**")
    for meta in metadatas:
        minutes = int(meta["start"] // 60)
        seconds = int(meta["start"] % 60)
        url = f"https://youtube.com/watch?v={meta['video_id']}&t={int(meta['start'])}s"
        snippet = meta.get("snippet", "")
        st.markdown(f"""
        <div class="source-box">
            <div class="source-title">🎥 <a href="{url}" target="_blank">{meta['video_title']} — {minutes}:{seconds:02d}</a></div>
        </div>
        """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Playlists")

    existing_collections = [c.name for c in client.list_collections()]

    if existing_collections:
        st.subheader("Switch playlist")
        selected = st.selectbox("Choose:", ["-- select --"] + existing_collections)
        if selected != "-- select --" and st.button("Load this playlist"):
            st.session_state.active_collection = selected
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No playlists processed yet. Add one below to get started.")

    st.divider()
    st.subheader("➕ Add a new playlist")
    new_url = st.text_input("YouTube playlist URL", placeholder="https://youtube.com/playlist?list=...")
    max_videos = st.number_input("Max videos to process", min_value=1, max_value=100, value=10, help="Start small to test, increase later")

    if st.button("🚀 Process this playlist", use_container_width=True):
        if not new_url.strip():
            st.error("Please paste a playlist URL first")
        elif "list=" not in new_url:
            st.error("That doesn't look like a valid playlist URL. It should contain 'list=' in it.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, current, total):
                status_text.text(f"⏳ {message} ({current}/{total})")
                if total > 0:
                    progress_bar.progress(min(current / total, 1.0))

            try:
                with st.spinner("Processing playlist... this will take a few minutes"):
                    result = process_playlist(new_url, max_videos=int(max_videos), progress_callback=update_progress)

                if result['num_chunks'] == 0:
                    st.error(f"No usable content found. {result['num_skipped']} video(s) were skipped — this course may not have transcripts available.")
                else:
                    st.success(f"✅ Done! Processed {result['num_videos']} videos, {result['num_chunks']} chunks ready.")
                    if result['num_skipped'] > 0:
                        st.warning(f"Note: {result['num_skipped']} video(s) skipped (no transcript available).")
                    st.session_state.active_collection = result["collection_name"]
                    st.session_state.messages = []
                    st.rerun()
            except Exception as e:
                error_name = type(e).__name__
                if "Blocked" in error_name:
                    st.error("🚫 YouTube is temporarily rate-limiting requests from this network. Please wait a while and try again.")
                else:
                    st.error(f"Something went wrong: {error_name}. Try a different playlist or wait a moment.")

    if st.session_state.active_collection:
        st.divider()
        st.caption(f"**Active:** `{st.session_state.active_collection}`")

# --- MAIN CHAT AREA ---
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

        if distances[0] > 1.5:
            return "This doesn't seem to be covered in the course material I have. Try rephrasing, or ask about a topic covered in this course.", []

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
                render_sources(msg["sources"])

    question = st.chat_input("Ask something about this course...")

    if question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer_text, metadatas = ask_question(question)
                st.write(answer_text)
                if metadatas:
                    render_sources(metadatas)
            except Exception as e:
                answer_text = "Something went wrong while generating an answer. Please try again."
                metadatas = []
                st.error(f"{answer_text} ({type(e).__name__})")

        st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": metadatas})