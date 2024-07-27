import click
import os
from pytube import YouTube
from openai import OpenAI


@click.command()
@click.argument("url")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API Key")
def transcribe(url, api_key):
    """Transcribe audio from a YouTube video."""
    if not api_key:
        raise click.ClickException(
            "OpenAI API Key is required. Set OPENAI_API_KEY environment variable or use --api-key option."
        )

    # Download audio from YouTube
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_filename = audio_stream.download(filename=audio_stream.default_filename)

    try:
        # Transcribe audio using OpenAI Whisper API
        client = OpenAI(api_key=api_key)

        with open(audio_filename, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            print(transcription.text)

        # Save transcript to file
        with open("transcript.md", "w") as f:
            f.write(transcription.text)

        click.echo(f"Transcript saved to transcript.md")
    finally:
        # Clean up temporary audio file
        os.remove(audio_filename)


if __name__ == "__main__":
    transcribe()
