import yt_dlp
import ollama
import chromadb
from youtube_transcript_api import YouTubeTranscriptApi
import re

CHUNK_SIZE = 400

def get_collection_name(playlist_url):
    """Turn a playlist URL into a safe, unique collection name for ChromaDB"""
    match = re.search(r"list=([a-zA-Z0-9_-]+)", playlist_url)
    playlist_id = match.group(1) if match else "unknown"
    return f"playlist_{playlist_id}"

def get_playlist_videos(playlist_url, max_videos=None):
    """Get list of {id, title} for all videos in a playlist"""
    ydl_opts = {"extract_flat": True, "quiet": True}
    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        for entry in info["entries"]:
            videos.append({"id": entry["id"], "title": entry["title"]})
    if max_videos:
        videos = videos[:max_videos]
    return videos

def fetch_transcripts(videos, progress_callback=None):
    """Fetch transcripts for a list of videos. Returns list of transcript lines."""
    api = YouTubeTranscriptApi()
    all_data = []
    skipped = []

    from youtube_transcript_api._errors import IpBlocked, RequestBlocked

    for i, video in enumerate(videos):
        try:
            transcript = api.fetch(video["id"], languages=['en-IN', 'en', 'hi'])
            for entry in transcript.to_raw_data():
                all_data.append({
                    "video_id": video["id"],
                    "video_title": video["title"],
                    "text": entry["text"],
                    "start": entry["start"]
                })
        except (IpBlocked, RequestBlocked):
            # Stop immediately - no point trying more videos if we're rate limited
            raise
        except Exception:
            skipped.append(video["title"])

        if progress_callback:
            progress_callback(i + 1, len(videos), video["title"])

    return all_data, skipped

def chunk_transcripts(all_lines):
    """Group transcript lines into ~400 character chunks"""
    chunks = []
    current_chunk_text = ""
    current_chunk_start = None
    current_video_id = None
    current_video_title = None

    for line in all_lines:
        if current_chunk_text == "":
            current_chunk_start = line["start"]
            current_video_id = line["video_id"]
            current_video_title = line["video_title"]

        if line["video_id"] != current_video_id and current_chunk_text != "":
            chunks.append({
                "video_id": current_video_id,
                "video_title": current_video_title,
                "start": current_chunk_start,
                "text": current_chunk_text.strip()
            })
            current_chunk_text = ""
            current_chunk_start = line["start"]
            current_video_id = line["video_id"]
            current_video_title = line["video_title"]

        current_chunk_text += " " + line["text"]

        if len(current_chunk_text) >= CHUNK_SIZE:
            chunks.append({
                "video_id": current_video_id,
                "video_title": current_video_title,
                "start": current_chunk_start,
                "text": current_chunk_text.strip()
            })
            current_chunk_text = ""
            current_chunk_start = None

    if current_chunk_text.strip() != "":
        chunks.append({
            "video_id": current_video_id,
            "video_title": current_video_title,
            "start": current_chunk_start,
            "text": current_chunk_text.strip()
        })

    return chunks

def build_database(chunks, collection_name, progress_callback=None):
    """Embed chunks and store them in a ChromaDB collection"""
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=collection_name)

    for i, chunk in enumerate(chunks):
        response = ollama.embed(model="nomic-embed-text", input=chunk["text"])
        embedding = response["embeddings"][0]

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "video_id": chunk["video_id"],
                "video_title": chunk["video_title"],
                "start": chunk["start"]
            }]
        )

        if progress_callback:
            progress_callback(i + 1, len(chunks))

    return collection

def process_playlist(playlist_url, max_videos=None, progress_callback=None):
    """Full pipeline: playlist URL -> ready-to-query collection"""
    collection_name = get_collection_name(playlist_url)

    if progress_callback:
        progress_callback("Fetching video list...", 0, 1)
    videos = get_playlist_videos(playlist_url, max_videos)

    def transcript_progress(current, total, title):
        if progress_callback:
            progress_callback(f"Fetching transcripts: {title[:40]}...", current, total)

    all_lines, skipped = fetch_transcripts(videos, transcript_progress)

    if progress_callback:
        progress_callback("Chunking transcripts...", 0, 1)
    chunks = chunk_transcripts(all_lines)

    def embed_progress(current, total):
        if progress_callback:
            progress_callback(f"Embedding chunks...", current, total)

    build_database(chunks, collection_name, embed_progress)

    return {
        "collection_name": collection_name,
        "num_videos": len(videos) - len(skipped),
        "num_skipped": len(skipped),
        "num_chunks": len(chunks),
        "playlist_url": playlist_url
    }