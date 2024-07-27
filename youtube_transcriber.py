import click
import os
from pytube import YouTube
from openai import OpenAI
import json


@click.command()
@click.argument("url")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="OpenAI API Key")
@click.option(
    "--ai-audio-model", default="whisper-1", help="OpenAI audio transcription model"
)
@click.option(
    "--ai-chat-model", default="gpt-4", help="OpenAI AI chat model for post-processing"
)
def transcribe(url, api_key, ai_audio_model, ai_chat_model):
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
                model=ai_audio_model, file=audio_file
            )
            print(transcription.text)

        # Save transcript to file
        with open("transcript.md", "w") as f:
            f.write(transcription.text)

        click.echo(f"Transcript saved to transcript.md")

        # Ask user if they want to post-process the transcript
        if click.confirm("Do you want to post-process the transcript with AI?"):
            transcript_postprocess(transcription.text, api_key, ai_chat_model)
    finally:
        # Clean up temporary audio file
        os.remove(audio_filename)


def transcript_postprocess(transcript, api_key, ai_model):
    """Post-process the transcript using specified OpenAI model."""
    client = OpenAI(api_key=api_key)

    user_prompt = click.prompt("What would you like to do with the transcript?")

    system_message = (
        "You are a helpful assistant that can analyze and process transcripts."
    )
    user_message = f"Please review the following transcript and respond to the user's request below.\n\n```{transcript}```\n\nUser's request: {user_prompt}"

    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=4096,
    )

    print("\nAI Response:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    transcribe()
