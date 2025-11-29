# utils_audio.py
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import noisereduce as nr
import pyloudnorm as pyln
import io

def read_audio_file(path, sr=44100):
    """Return samples (float32, mono) and sample rate."""
    data, rate = sf.read(path, dtype='float32')
    # If stereo, convert to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    # Resample if needed using librosa
    if rate != sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=rate, target_sr=sr)
            rate = sr
        except ImportError:
            pass  # If librosa not available, use original rate
    return data, rate

def write_audio_file(path, samples, rate):
    """Write numpy float32 array to WAV"""
    sf.write(path, samples, rate, subtype='PCM_24')
    return path

def pydub_from_numpy(samples, rate):
    """Convert float32 numpy (-1..1) to pydub AudioSegment"""
    # pydub expects 16-bit PCM by default
    int_samples = np.int16(np.clip(samples * 32767, -32768, 32767))
    audio = AudioSegment(
        int_samples.tobytes(),
        frame_rate=rate,
        sample_width=2,
        channels=1
    )
    return audio

def numpy_from_pydub(seg: AudioSegment):
    """Return numpy float32 normalized to -1..1 from pydub AudioSegment"""
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.sample_width == 2:
        samples /= 32768.0
    return samples, seg.frame_rate

def reduce_noise(samples, rate, prop_decrease=1.0, n_fft=2048, hop_length=None):
    # estimate noise from first 1 second
    noise_clip = samples[: min(len(samples), rate)]
    reduced = nr.reduce_noise(y=samples, sr=rate, y_noise=noise_clip,
                              prop_decrease=prop_decrease, n_fft=n_fft, hop_length=hop_length)
    return reduced

def normalize_loudness(samples, rate, target_lufs=-16.0):
    meter = pyln.Meter(rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(samples)
    gain_db = target_lufs - loudness
    # apply gain
    factor = 10.0 ** (gain_db / 20.0)
    samples_norm = samples * factor
    return samples_norm, loudness, gain_db

def trim_silence_pydub(seg: AudioSegment, silence_thresh=-40, min_silence_len=700, padding=150):
    """
    Remove leading/trailing silence and shorten long internal silences.
    min_silence_len in ms. padding kept in ms.
    """
    from pydub.silence import detect_nonsilent, split_on_silence

    # Trim leading/trailing silence by detecting non-silent ranges
    nonsilent_ranges = detect_nonsilent(seg, min_silence_len=200, silence_thresh=silence_thresh)
    if not nonsilent_ranges:
        return seg  # all silence?
    start = max(0, nonsilent_ranges[0][0] - padding)
    end = min(len(seg), nonsilent_ranges[-1][1] + padding)
    trimmed = seg[start:end]

    # Shorten long internal silences by splitting and rejoining with short gap
    chunks = split_on_silence(trimmed, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=padding)
    if not chunks:
        return trimmed
    result = chunks[0]
    short_gap = AudioSegment.silent(duration=250)  # 250ms gap between speech segments
    for c in chunks[1:]:
        result += short_gap + c
    return result
