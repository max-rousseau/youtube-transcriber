import click
import os
from pytube import YouTube
import openai

@click.command()
@click.argument('url')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API Key')
def transcribe(url, api_key):
    """Transcribe audio from a YouTube video."""
    if not api_key:
        raise click.ClickException("OpenAI API Key is required. Set OPENAI_API_KEY environment variable or use --api-key option.")

    openai.api_key = api_key

    # Download audio from YouTube
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(filename='temp_audio')

    try:
        # Transcribe audio using OpenAI Whisper API
        with open(audio_file, 'rb') as audio:
            transcript = openai.Audio.transcribe("whisper-1", audio)

        # Save transcript to file
        with open('transcript.md', 'w') as f:
            f.write(transcript['text'])

        click.echo(f"Transcript saved to transcript.md")
    finally:
        # Clean up temporary audio file
        os.remove(audio_file)

if __name__ == '__main__':
    transcribe()
