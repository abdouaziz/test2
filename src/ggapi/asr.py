import io
from functools import lru_cache

import librosa
from transformers import pipeline

MODEL_ID = "openai/whisper-small"
SAMPLING_RATE = 16_000


@lru_cache(maxsize=1)
def get_pipeline():
    # Loaded once on first use; the weights (~1 GB) are cached by Hugging Face.
    return pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        device="cpu",
        chunk_length_s=30,
    )


def transcribe(audio: bytes, language: str | None = None) -> str:
    waveform, _ = librosa.load(io.BytesIO(audio), sr=SAMPLING_RATE, mono=True)
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = get_pipeline()(
        {"raw": waveform, "sampling_rate": SAMPLING_RATE},
        generate_kwargs=generate_kwargs,
    )
    return result["text"].strip()
