import os
import subprocess
import json
import math
import argparse
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

def get_video_duration(video_path):
    """Gets the duration of a video file in seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout)

def split_video(video_path, output_dir, num_chunks=2):
    """Splits the video into a specified number of chunks."""
    duration = get_video_duration(video_path)
    chunk_duration = math.ceil(duration / num_chunks)
    chunk_paths = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_path = os.path.join(output_dir, f"chunk_{i+1}.mp4")
        chunk_paths.append(output_path)
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-ss",
            str(start_time),
            "-t",
            str(chunk_duration),
            "-c",
            "copy",
            "-y", # Overwrite output file if it exists
            output_path,
        ]
        print(f"Creating chunk {i+1}...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Chunk {i+1} created at {output_path}")

    return chunk_paths, chunk_duration

def transcribe_chunk(client, file_path):
    """Transcribes a single video chunk using Eleven Labs API."""
    print(f"Transcribing {os.path.basename(file_path)}...")
    with open(file_path, "rb") as f:
        response = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            diarize=True
        )
    print(f"Finished transcribing {os.path.basename(file_path)}.")
    # The response is already a deserialized JSON object (a Pydantic model)
    return response.dict()


def merge_transcripts(transcripts, chunk_duration):
    """Merges multiple transcript chunks, adjusting timestamps."""
    merged_transcript = {"words": [], "utterances": []}
    time_offset = 0.0

    for i, transcript_data in enumerate(transcripts):
        # Adjust word timestamps
        for word in transcript_data.get("words", []):
            word["start"] += time_offset
            word["end"] += time_offset
            merged_transcript["words"].append(word)

        # Adjust utterance timestamps
        for utterance in transcript_data.get("utterances", []):
            utterance["start"] += time_offset
            utterance["end"] += time_offset
            merged_transcript["utterances"].append(utterance)

        time_offset += chunk_duration

    return merged_transcript

def main():
    parser = argparse.ArgumentParser(description="Transcribe a video using Eleven Labs API, splitting it into chunks.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("output_dir", help="Directory to save the final transcript and chunks.")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise ValueError("ELEVEN_API_KEY not found. Please create a .env file in the project root with ELEVEN_API_KEY='your-key-here'")

    client = ElevenLabs(api_key=api_key)
    
    video_chunks_dir = os.path.join(args.output_dir, "video_chunks")

    # 1. Split the video
    chunk_paths, chunk_duration = split_video(args.video_path, video_chunks_dir)

    # 2. Transcribe each chunk
    transcripts = []
    for chunk_path in chunk_paths:
        transcript = transcribe_chunk(client, chunk_path)
        transcripts.append(transcript)

    # 3. Merge the transcripts
    merged_transcript = merge_transcripts(transcripts, chunk_duration)

    # 4. Save the final transcript
    output_file_path = os.path.join(args.output_dir, "elevenlabs_transcript.json")
    with open(output_file_path, "w") as f:
        json.dump(merged_transcript, f, indent=2)

    print(f"\\n✅ Final transcript saved to {output_file_path}")
    print(f"You can now delete the temporary chunks in: {video_chunks_dir}")


if __name__ == "__main__":
    main() 