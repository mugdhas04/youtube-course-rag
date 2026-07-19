from youtube_transcript_api import YouTubeTranscriptApi

video_id = "yRpLlJmRo2w"  # the video that got blocked earlier

api = YouTubeTranscriptApi()

try:
    transcript = api.fetch(video_id, languages=['en-IN', 'en', 'hi'])
    print(f"SUCCESS: Got {len(transcript.to_raw_data())} lines - rate limit has cleared!")
except Exception as e:
    print(f"STILL BLOCKED: {type(e).__name__}")
    print(e)