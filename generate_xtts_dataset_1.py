import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from TTS.api import TTS

# =========================
# CONFIG
# =========================
INPUT_FILE = Path(r"C:\Users\sabbi\Downloads\archive\cot_validated_partial.jsonl")
OUTPUT_DIR = Path(r"C:\Users\sabbi\Documents\Codex\xtts_real_audio_dataset_1")
SAMPLE_RATE = 24000
PRINT_INTERVAL = 25
CHECKPOINT_EVERY = 500

USER_SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
]

ASSISTANT_SPEAKER = "Ana Florence"
LANGUAGE = "en"
MAX_CHARS_PER_CHUNK = 220
# =========================

WAV_DIR = OUTPUT_DIR / "wavs"
TRANS_DIR = OUTPUT_DIR / "transcripts"
INDEX_PATH = OUTPUT_DIR / "dataset.jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WAV_DIR.mkdir(parents=True, exist_ok=True)
TRANS_DIR.mkdir(parents=True, exist_ok=True)


def number_to_words_simple(text: str) -> str:
    try:
        from num2words import num2words

        def replace_num(match):
            num_str = match.group(0).replace(",", "")
            try:
                return num2words(float(num_str)) if "." in num_str else num2words(int(num_str))
            except Exception:
                return match.group(0)

        return re.sub(r"-?\d[\d,]*\.?\d*", replace_num, text)
    except Exception:
        return text


def split_text_for_tts(text: str, max_chars: int = 220):
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for part in parts:
        if len(part) > max_chars:
            words = part.split()
            temp = ""
            for word in words:
                candidate = word if not temp else temp + " " + word
                if len(candidate) <= max_chars:
                    temp = candidate
                else:
                    if temp:
                        chunks.append(temp)
                    temp = word
            if temp:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(temp)
            continue

        candidate = part if not current else current + " " + part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def synthesize_speech(tts, text: str, speaker: str) -> np.ndarray:
    spoken_text = number_to_words_simple(text)
    chunks = split_text_for_tts(spoken_text, max_chars=MAX_CHARS_PER_CHUNK)

    all_audio = []
    silence = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)

    for chunk in chunks:
        wav = tts.tts(
            text=chunk,
            speaker=speaker,
            language=LANGUAGE,
        )
        all_audio.append(np.array(wav, dtype=np.float32))
        all_audio.append(silence)

    if not all_audio:
        return np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)

    return np.concatenate(all_audio[:-1])


def make_stereo_wav(assistant_audio: np.ndarray, user_audio: np.ndarray) -> np.ndarray:
    max_len = max(len(assistant_audio), len(user_audio))

    if len(assistant_audio) < max_len:
        assistant_audio = np.pad(assistant_audio, (0, max_len - len(assistant_audio)))
    if len(user_audio) < max_len:
        user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))

    stereo = np.stack([assistant_audio, user_audio], axis=-1)

    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.9

    return (stereo * 32767).astype(np.int16)


def load_existing_index():
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_index(dataset_index):
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        for entry in dataset_index:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading XTTS on:", device)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("XTTS loaded successfully.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} samples from {INPUT_FILE}")

    dataset_index = load_existing_index()
    completed_ids = set()

    for entry in dataset_index:
        transcript_path = Path(entry["transcript"])
        if transcript_path.exists():
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                completed_ids.add(transcript["id"])
            except Exception:
                pass

    print(f"Already completed samples found: {len(completed_ids)}")

    error_count = 0

    for i, item in enumerate(data):
        if item["id"] in completed_ids:
            continue

        try:
            user_speaker = USER_SPEAKERS[i % len(USER_SPEAKERS)]

            user_audio = synthesize_speech(tts, item["question"], user_speaker)
            assistant_audio = synthesize_speech(tts, item["generated_concise_answer"], ASSISTANT_SPEAKER)

            stereo = make_stereo_wav(assistant_audio, user_audio)

            wav_filename = f"{i:05d}.wav"
            wav_path = WAV_DIR / wav_filename
            wavfile.write(wav_path, SAMPLE_RATE, stereo)

            transcript = {
                "id": item["id"],
                "wav_file": wav_filename,
                "user_text": item["question"],
                "assistant_text": item["generated_concise_answer"],
                "reasoning_text": item["generated_reasoning"],
                "ground_truth": item["ground_truth"],
                "duration_seconds": round(len(stereo) / SAMPLE_RATE, 2),
                "source": item.get("source", "original"),
            }

            trans_path = TRANS_DIR / f"{i:05d}.json"
            with open(trans_path, "w", encoding="utf-8") as tf:
                json.dump(transcript, tf, ensure_ascii=False, indent=2)

            dataset_index.append(
                {
                    "audio": str(wav_path),
                    "transcript": str(trans_path),
                    "duration": transcript["duration_seconds"],
                }
            )
            completed_ids.add(item["id"])

        except Exception as e:
            error_count += 1
            print(f"ERROR on sample {i}: {e}")

        processed_now = len(completed_ids)

        if processed_now % PRINT_INTERVAL == 0:
            print(f"Completed total: {processed_now}/{len(data)} | Errors: {error_count}")

        if processed_now % CHECKPOINT_EVERY == 0:
            save_index(dataset_index)
            print(f"\nCHECKPOINT SAVED at {processed_now} samples -> {INDEX_PATH}")

    save_index(dataset_index)

    print("\nDone.")
    print(f"Saved {len(dataset_index)}/{len(data)} samples")
    print(f"Errors: {error_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"WAV folder: {WAV_DIR}")
    print(f"Transcript folder: {TRANS_DIR}")
    print(f"Index file: {INDEX_PATH}")


if __name__ == "__main__":
    main()
