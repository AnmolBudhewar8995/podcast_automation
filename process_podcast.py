#!/usr/bin/env python3
# process_podcast.py
import os
import argparse
from pathlib import Path
import numpy as np
from pydub import AudioSegment, effects
from utils_audio import read_audio_file, write_audio_file, reduce_noise, pydub_from_numpy, numpy_from_pydub, normalize_loudness, trim_silence_pydub
from mutagen.easyid3 import EasyID3
from tqdm import tqdm

# ensure ffmpeg available for pydub
AudioSegment.converter = "ffmpeg"  # if ffmpeg is on PATH this is redundant

def process_file(input_path, output_dir, target_lufs=-16.0, do_denoise=True, do_trim=True, do_normalize=True, do_transcribe=False, whisper_model="tiny"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem
    print(f"Processing: {input_path.name}")

    # Read with soundfile (numpy) to do noise reduction and loudness measurement
    samples, rate = read_audio_file(str(input_path))
    # Denoise
    if do_denoise:
        print(" - reducing noise...")
        samples = reduce_noise(samples, rate)
    # Normalize loudness (compute gain)
    if do_normalize:
        print(" - normalizing loudness...")
        samples, measured_lu, gain_db = normalize_loudness(samples, rate, target_lufs)
        print(f"   measured LUFS: {measured_lu:.2f} dB, applied gain: {gain_db:.2f} dB")
    # Convert to pydub for silence trimming / internal silence edits
    audio_seg = pydub_from_numpy(samples, rate)

    # Trim silence and shorten long pauses
    if do_trim:
        print(" - trimming silence...")
        audio_seg = trim_silence_pydub(audio_seg, silence_thresh=-40, min_silence_len=700, padding=150)

    # Optional: apply final limiter/normalize via pydub for final headroom
    audio_seg = effects.normalize(audio_seg)  # pydub normalize (peak)
    # Export final file as WAV (24-bit) using soundfile (no FFmpeg needed)
    out_wav = output_dir / f"{base_name}_edited.wav"
    print(f" - exporting {out_wav.name}")
    # Convert pydub segment back to numpy for soundfile export
    final_samples = numpy_from_pydub(audio_seg)[0]
    write_audio_file(str(out_wav), final_samples, audio_seg.frame_rate)
    
    # Try to export MP3 if FFmpeg is available
    out_mp3 = None
    try:
        out_mp3 = output_dir / f"{base_name}_edited.mp3"
        print(f" - exporting {out_mp3.name}")
        audio_seg.export(str(out_mp3), format="mp3", bitrate="192k")
    except Exception as e:
        print(f" - MP3 export skipped (FFmpeg not available): {e}")
        out_mp3 = None

    # Add basic ID3 tags to the MP3 (only if MP3 was created)
    if out_mp3:
        try:
            audio = EasyID3(str(out_mp3))
        except Exception:
            from mutagen.mp3 import MP3
            audio = MP3(str(out_mp3), ID3=EasyID3)
        audio["title"] = base_name
        audio["artist"] = "Host Name"
        audio["album"] = "Podcast"
        audio.save()
        print(" - tags written")

    # Optional transcription using whisper (if requested)
    if do_transcribe:
        try:
            import whisper
            print(" - transcribing with Whisper:", whisper_model)
            model = whisper.load_model(whisper_model)
            # Whisper can work with numpy arrays directly
            # Load the audio file using soundfile and convert to the format Whisper expects
            audio_data, sr = read_audio_file(str(input_path))
            # Whisper expects 16kHz mono audio
            if sr != 16000:
                try:
                    # Resample to 16kHz
                    import scipy.signal
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                    sr = 16000
                except:
                    pass
            
            result = model.transcribe(audio_data)
            transcript = result["text"]
            txt_out = output_dir / f"{base_name}_transcript.txt"
            txt_out.write_text(transcript, encoding="utf-8")
            print(" - transcript saved:", txt_out.name)
        except Exception as e:
            print(" - transcription error:", e)

    print("Done:", input_path.name)

def batch_process(input_folder, output_folder, **kwargs):
    input_folder = Path(input_folder)
    files = list(input_folder.glob("*.*"))
    audio_exts = [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"]
    to_process = [f for f in files if f.suffix.lower() in audio_exts]
    for f in tqdm(to_process, desc="Batch"):
        try:
            process_file(f, output_folder, **kwargs)
        except Exception as e:
            print("Error processing", f.name, ":", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Podcast automation processor")
    parser.add_argument("--input", "-i", required=True, help="Input file or folder")
    parser.add_argument("--output", "-o", default="output", help="Output folder")
    parser.add_argument("--no-denoise", action="store_true", help="Disable noise reduction")
    parser.add_argument("--no-trim", action="store_true", help="Disable silence trimming")
    parser.add_argument("--no-normalize", action="store_true", help="Disable loudness normalization")
    parser.add_argument("--transcribe", action="store_true", help="Run local Whisper transcription (optional)")
    parser.add_argument("--whisper-model", default="tiny", help="Whisper model name (tiny, base, small, etc.)")
    parser.add_argument("--lufs", type=float, default=-16.0, help="Target LUFS level (default -16.0)")

    args = parser.parse_args()

    is_folder = Path(args.input).is_dir()
    kwargs = {
        "do_denoise": not args.no_denoise,
        "do_trim": not args.no_trim,
        "do_normalize": not args.no_normalize,
        "do_transcribe": args.transcribe,
        "whisper_model": args.whisper_model,
        "target_lufs": args.lufs
    }
    if is_folder:
        batch_process(args.input, args.output, **kwargs)
    else:
        process_file(args.input, args.output, **kwargs)
