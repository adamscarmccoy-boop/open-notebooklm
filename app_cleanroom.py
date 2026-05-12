"""
app_cleanroom.py
Modified version of Open NotebookLM that includes automatic CleanRoom grounding.
"""

import os
import glob
import time
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional
import gradio as gr
from loguru import logger
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from constants import (
    APP_TITLE, CHARACTER_LIMIT, ERROR_MESSAGE_NOT_PDF, ERROR_MESSAGE_NO_INPUT,
    ERROR_MESSAGE_NOT_SUPPORTED_IN_MELO_TTS, ERROR_MESSAGE_READING_PDF,
    ERROR_MESSAGE_TOO_LONG, GRADIO_CACHE_DIR, GRADIO_CLEAR_CACHE_OLDER_THAN,
    MELO_TTS_LANGUAGE_MAPPING, NOT_SUPPORTED_IN_MELO_TTS, SUNO_LANGUAGE_MAPPING,
    UI_ALLOW_FLAGGING, UI_API_NAME, UI_CACHE_EXAMPLES, UI_CONCURRENCY_LIMIT,
    UI_DESCRIPTION, UI_INPUTS, UI_OUTPUTS, UI_SHOW_API,
)
from prompts import (
    LANGUAGE_MODIFIER, LENGTH_MODIFIERS, QUESTION_MODIFIER, SYSTEM_PROMPT, TONE_MODIFIER,
)
from schema import ShortDialogue, MediumDialogue
from utils import generate_podcast_audio, generate_script, parse_url

CLEANROOM_PATHS_FILE = r"C:\Users\adams\NotebookLM_CleanRoom_Paths.txt"

def load_cleanroom_context() -> str:
    """Reads all text files from the CleanRoom paths file and returns the combined text."""
    context = "=== CLEANROOM GROUNDING DATA ===\n"
    if not os.path.exists(CLEANROOM_PATHS_FILE):
        return ""
    
    with open(CLEANROOM_PATHS_FILE, 'r') as f:
        paths = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('=')]
    
    for path in paths:
        if os.path.exists(path):
            try:
                # Only read text-based files to avoid binary bloat
                if path.endswith(('.txt', '.py', '.ps1', '.bat', '.json', '.yaml', '.yml', '.md', '.ini')):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        context += f"\n--- SOURCE: {path} ---\n{f.read()[:2000]}\n" # Cap each file at 2k chars
            except:
                continue
    return context

def generate_podcast_cleanroom(
    files: List[str],
    url: Optional[str],
    question: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    language: str,
    use_advanced_audio: bool,
    use_cleanroom: bool
) -> Tuple[str, str]:
    """Generate the audio and transcript with optional CleanRoom grounding."""
    
    text = ""
    if use_cleanroom:
        logger.info("Loading CleanRoom grounding data...")
        text += load_cleanroom_context()

    # Process PDFs if any
    if files:
        for file in files:
            if file.lower().endswith(".pdf"):
                with Path(file).open("rb") as f:
                    reader = PdfReader(f)
                    text += "\n\n" + "\n\n".join([page.extract_text() for page in reader.pages])
            else:
                # Handle other text files if uploaded
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        text += f"\n\n--- UPLOADED FILE: {file} ---\n{f.read()}\n"
                except:
                    pass

    if url:
        text += "\n\n" + parse_url(url)

    if not text and not use_cleanroom:
        raise gr.Error("Please provide some input (Files, URL, or CleanRoom).")

    # Call LLM with Gemma
    modified_system_prompt = SYSTEM_PROMPT
    if question: modified_system_prompt += f"\n\n{QUESTION_MODIFIER} {question}"
    if tone: modified_system_prompt += f"\n\n{TONE_MODIFIER} {tone}."
    if length: modified_system_prompt += f"\n\n{LENGTH_MODIFIERS[length]}"
    if language: modified_system_prompt += f"\n\n{LANGUAGE_MODIFIER} {language}."

    dialogue_format = ShortDialogue if length == "Short (1-2 min)" else MediumDialogue
    llm_output = generate_script(modified_system_prompt, text, dialogue_format)

    # Process audio... (Simplified for logic check)
    audio_segments = []
    transcript = ""
    random_voice_number = random.randint(0, 8)

    for line in llm_output.dialogue:
        speaker_label = "**Host**" if line.speaker == "Host (Jane)" else f"**{llm_output.name_of_guest}**"
        transcript += f"{speaker_label}: {line.text}\n\n"
        
        language_for_tts = SUNO_LANGUAGE_MAPPING[language]
        if not use_advanced_audio:
            language_for_tts = MELO_TTS_LANGUAGE_MAPPING.get(language_for_tts.lower(), "EN")

        audio_file_path = generate_podcast_audio(
            line.text, line.speaker, language_for_tts, use_advanced_audio, random_voice_number
        )
        audio_segments.append(AudioSegment.from_file(audio_file_path))

    combined_audio = sum(audio_segments)
    temp_file = NamedTemporaryFile(dir=GRADIO_CACHE_DIR, delete=False, suffix=".mp3")
    combined_audio.export(temp_file.name, format="mp3")
    
    return temp_file.name, transcript

demo = gr.Interface(
    title=f"{APP_TITLE} (CleanRoom Edition)",
    description="Local Gemma-powered podcast generator with CleanRoom grounding.",
    fn=generate_podcast_cleanroom,
    inputs=[
        gr.File(label="📄 Upload Source Files", file_count="multiple"),
        gr.Textbox(label="🔗 Source URL"),
        gr.Textbox(label="🤔 Focus Question"),
        gr.Dropdown(label="🎭 Tone", choices=["Fun", "Formal"], value="Fun"),
        gr.Dropdown(label="⏱️ Length", choices=["Short (1-2 min)", "Medium (3-5 min)"], value="Medium (3-5 min)"),
        gr.Dropdown(label="🌐 Language", choices=list(SUNO_LANGUAGE_MAPPING.keys()), value="English"),
        gr.Checkbox(label="🔄 Use Advanced Audio", value=True),
        gr.Checkbox(label="🧬 Use CleanRoom Grounding (C:\\Users\\adams\\NotebookLM_CleanRoom_Paths.txt)", value=True),
    ],
    outputs=[gr.Audio(label="🔊 Podcast"), gr.Markdown(label="📜 Transcript")],
    theme=gr.themes.Ocean(),
)

if __name__ == "__main__":
    demo.launch()
