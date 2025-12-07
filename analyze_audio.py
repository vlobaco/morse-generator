#!/usr/bin/env python3
"""
Analyze Morse code audio file timing to help calibrate decoder
"""

import sys
import numpy as np
from pydub import AudioSegment

SILENCE_THRESHOLD = -40  # dB


def load_audio(file_path):
    """Load audio file and convert to mono"""
    if file_path.endswith('.wav'):
        audio = AudioSegment.from_wav(file_path)
    elif file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
    else:
        audio = AudioSegment.from_file(file_path)

    if audio.channels > 1:
        audio = audio.set_channels(1)
    return audio


def detect_segments(audio):
    """Detect tone and silence segments in the audio"""
    samples = np.array(audio.get_array_of_samples())
    frame_rate = audio.frame_rate

    samples_per_ms = frame_rate // 1000
    num_frames = len(samples) // samples_per_ms

    segments = []
    current_is_tone = None
    current_start = 0

    for i in range(num_frames):
        start_sample = i * samples_per_ms
        end_sample = start_sample + samples_per_ms
        frame = samples[start_sample:end_sample]

        if len(frame) == 0:
            continue

        frame_squared = frame.astype(np.float64) ** 2
        mean_square = np.mean(frame_squared)

        if mean_square < 0:
            mean_square = 0

        rms = np.sqrt(mean_square)

        if rms < 1e-10:
            db = -100
        else:
            db = 20 * np.log10(rms)

        is_tone = db > SILENCE_THRESHOLD

        if current_is_tone is None:
            current_is_tone = is_tone
            current_start = i
        elif is_tone != current_is_tone:
            segments.append((current_start, i, current_is_tone))
            current_is_tone = is_tone
            current_start = i

    if current_is_tone is not None:
        segments.append((current_start, num_frames, current_is_tone))

    return segments


def analyze_timing(file_path):
    """Analyze timing statistics of the audio file"""
    print(f"Analyzing {file_path}...")
    audio = load_audio(file_path)
    segments = detect_segments(audio)

    # Filter out very short segments
    min_segment_duration = 20
    segments = [(s, e, t) for s, e, t in segments if (e - s) >= min_segment_duration]

    # Separate tone and silence segments
    tone_durations = [e - s for s, e, is_tone in segments if is_tone]
    silence_durations = [e - s for s, e, is_tone in segments if not is_tone]

    print(f"\nTotal segments: {len(segments)}")
    print(f"Tone segments: {len(tone_durations)}")
    print(f"Silence segments: {len(silence_durations)}")

    if tone_durations:
        print("\n--- TONE DURATIONS (ms) ---")
        print(f"Min: {min(tone_durations)}")
        print(f"Max: {max(tone_durations)}")
        print(f"Mean: {np.mean(tone_durations):.1f}")
        print(f"Median: {np.median(tone_durations):.1f}")
        print(f"Std Dev: {np.std(tone_durations):.1f}")

        # Show distribution
        unique_durations = sorted(set(tone_durations))
        print(f"\nTone duration distribution (showing durations that appear):")
        for duration in unique_durations[:20]:  # Show first 20
            count = tone_durations.count(duration)
            print(f"  {duration}ms: {count} times")

    if silence_durations:
        print("\n--- SILENCE DURATIONS (ms) ---")
        print(f"Min: {min(silence_durations)}")
        print(f"Max: {max(silence_durations)}")
        print(f"Mean: {np.mean(silence_durations):.1f}")
        print(f"Median: {np.median(silence_durations):.1f}")
        print(f"Std Dev: {np.std(silence_durations):.1f}")

        # Show distribution
        unique_durations = sorted(set(silence_durations))
        print(f"\nSilence duration distribution (showing durations that appear):")
        for duration in unique_durations[:20]:  # Show first 20
            count = silence_durations.count(duration)
            print(f"  {duration}ms: {count} times")

    # Suggest thresholds
    print("\n--- SUGGESTED THRESHOLDS ---")
    if tone_durations:
        # Find clustering for dot vs dash
        sorted_tones = sorted(tone_durations)
        # Try to find the gap between short and long tones
        if len(sorted_tones) > 2:
            # Look for a natural break
            diffs = [sorted_tones[i+1] - sorted_tones[i] for i in range(len(sorted_tones)-1)]
            max_diff_idx = diffs.index(max(diffs))
            suggested_dash_threshold = (sorted_tones[max_diff_idx] + sorted_tones[max_diff_idx + 1]) / 2
            print(f"DOT vs DASH threshold: {suggested_dash_threshold:.1f} ms")

            # Estimate dot duration from shorter tones
            short_tones = [t for t in tone_durations if t < suggested_dash_threshold]
            if short_tones:
                avg_dot = np.mean(short_tones)
                print(f"Average DOT duration: {avg_dot:.1f} ms")

    if silence_durations:
        sorted_silences = sorted(silence_durations)
        print(f"\nSilence timing suggestions:")
        print(f"Shortest silence: {sorted_silences[0]} ms (likely symbol gap)")
        if len(sorted_silences) > 1:
            # Find clusters in silence
            unique_silences = sorted(set(silence_durations))
            if len(unique_silences) >= 2:
                print(f"Second shortest: {unique_silences[1]} ms")
                print(f"Suggested SYMBOL_GAP_MAX: {unique_silences[1] + 20} ms")
            if len(unique_silences) >= 3:
                print(f"Third shortest: {unique_silences[2]} ms")
                print(f"Suggested LETTER_GAP_MAX: {unique_silences[2] + 20} ms")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter audio file to analyze: ").strip()

    if not file_path:
        print("Error: No file provided")
        sys.exit(1)

    try:
        analyze_timing(file_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
