#!/usr/bin/env python3
"""
Morse Code Audio Decoder
Decodes MP3 files containing Morse code audio back to text
"""

import sys
import numpy as np
from pydub import AudioSegment

# Reverse Morse code mapping (morse -> character)
MORSE_TO_CHAR = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
    '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
    '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@'
}

# Timing thresholds (in milliseconds)
DOT_DURATION = 100
DASH_MIN_DURATION = DOT_DURATION * 2.5  # Anything longer than 2.5x dot is a dash
SYMBOL_GAP_MAX = DOT_DURATION * 1.5  # Max gap within a letter
LETTER_GAP_MAX = DOT_DURATION * 5  # Max gap between letters
# Anything longer than LETTER_GAP_MAX is a word gap

# Detection threshold for audio level
SILENCE_THRESHOLD = -40  # dB


def load_audio(file_path):
    """Load audio file (MP3 or WAV) and convert to mono"""
    # Detect format from extension
    if file_path.endswith('.wav'):
        audio = AudioSegment.from_wav(file_path)
    elif file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(file_path)
    else:
        # Try to auto-detect format
        audio = AudioSegment.from_file(file_path)

    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    return audio


def detect_segments(audio):
    """
    Detect tone and silence segments in the audio
    Returns list of tuples: (start_ms, end_ms, is_tone)
    """
    # Convert to numpy array for analysis
    samples = np.array(audio.get_array_of_samples())
    frame_rate = audio.frame_rate

    # Calculate RMS energy for each millisecond
    samples_per_ms = frame_rate // 1000
    num_frames = len(samples) // samples_per_ms

    segments = []
    current_is_tone = None
    current_start = 0

    for i in range(num_frames):
        start_sample = i * samples_per_ms
        end_sample = start_sample + samples_per_ms
        frame = samples[start_sample:end_sample]

        # Skip empty frames
        if len(frame) == 0:
            continue

        # Calculate RMS and convert to dB
        frame_squared = frame.astype(np.float64) ** 2
        mean_square = np.mean(frame_squared)

        # Ensure non-negative value for sqrt
        if mean_square < 0:
            mean_square = 0

        rms = np.sqrt(mean_square)

        # Avoid log(0) by adding small epsilon
        if rms < 1e-10:
            db = -100  # Very quiet, definitely silence
        else:
            db = 20 * np.log10(rms)

        # Determine if this frame is tone or silence
        is_tone = db > SILENCE_THRESHOLD

        # Start new segment if state changed
        if current_is_tone is None:
            current_is_tone = is_tone
            current_start = i
        elif is_tone != current_is_tone:
            # Save previous segment
            segments.append((current_start, i, current_is_tone))
            current_is_tone = is_tone
            current_start = i

    # Add final segment
    if current_is_tone is not None:
        segments.append((current_start, num_frames, current_is_tone))

    return segments


def detect_timing_thresholds(segments):
    """
    Auto-detect timing thresholds from the audio segments using natural breaks algorithm
    This handles noise and multiple gap types robustly
    Returns: (dash_threshold, symbol_gap_max, letter_gap_max)
    """
    # Separate tone and silence durations
    tone_durations = [e - s for s, e, is_tone in segments if is_tone]
    silence_durations = [e - s for s, e, is_tone in segments if not is_tone]

    if not tone_durations or not silence_durations:
        # Fall back to default values
        return DASH_MIN_DURATION, SYMBOL_GAP_MAX, LETTER_GAP_MAX

    # ===== PART 1: Detect dot vs dash threshold =====
    unique_tones = sorted(set(tone_durations))

    if len(unique_tones) == 1:
        # Only one tone length - use it as dot, assume dash is 3x longer
        dash_threshold = unique_tones[0] * 2
    elif len(unique_tones) >= 2:
        # Find the largest gap between consecutive unique tone values
        # This gap is likely between dots and dashes
        gaps = [(unique_tones[i+1] - unique_tones[i], i)
                for i in range(len(unique_tones) - 1)]
        max_gap, max_gap_idx = max(gaps, key=lambda x: x[0])
        dash_threshold = (unique_tones[max_gap_idx] + unique_tones[max_gap_idx + 1]) / 2
    else:
        dash_threshold = DASH_MIN_DURATION

    # ===== PART 2: Detect silence gap thresholds =====
    unique_silences = sorted(set(silence_durations))

    if len(unique_silences) <= 3:
        # 3 or fewer unique silence values - use simple approach
        if len(unique_silences) == 3:
            symbol_gap_max = (unique_silences[0] + unique_silences[1]) / 2
            letter_gap_max = (unique_silences[1] + unique_silences[2]) / 2
        elif len(unique_silences) == 2:
            symbol_gap_max = (unique_silences[0] + unique_silences[1]) / 2
            letter_gap_max = unique_silences[1] * 1.5
        else:
            base_gap = unique_silences[0] if unique_silences else 100
            symbol_gap_max = base_gap * 1.5
            letter_gap_max = base_gap * 3
    else:
        # More than 3 unique values - use natural breaks to cluster into 3 groups
        # Find the 2 largest gaps to split the data into 3 clusters
        gaps = [(unique_silences[i+1] - unique_silences[i], i)
                for i in range(len(unique_silences) - 1)]

        # Sort gaps by size (descending) and take the 2 largest
        gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)
        largest_gap_indices = sorted([g[1] for g in gaps_sorted[:2]])

        # Split into 3 groups at these gaps
        group1 = unique_silences[:largest_gap_indices[0] + 1]
        group2 = unique_silences[largest_gap_indices[0] + 1:largest_gap_indices[1] + 1]
        group3 = unique_silences[largest_gap_indices[1] + 1:]

        # Use median of each group as representative value
        symbol_gap_representative = np.median(group1) if group1 else 100
        letter_gap_representative = np.median(group2) if group2 else 300
        word_gap_representative = np.median(group3) if group3 else 700

        # Thresholds are midpoints between representative values
        symbol_gap_max = (symbol_gap_representative + letter_gap_representative) / 2
        letter_gap_max = (letter_gap_representative + word_gap_representative) / 2

    return dash_threshold, symbol_gap_max, letter_gap_max


def segments_to_morse(segments, dash_threshold=None, symbol_gap_max=None, letter_gap_max=None):
    """Convert detected segments to Morse code string"""
    # Auto-detect thresholds if not provided
    if dash_threshold is None or symbol_gap_max is None or letter_gap_max is None:
        dash_threshold, symbol_gap_max, letter_gap_max = detect_timing_thresholds(segments)

    print(f"Using thresholds: dot/dash={dash_threshold:.1f}ms, symbol_gap={symbol_gap_max:.1f}ms, letter_gap={letter_gap_max:.1f}ms")

    morse_chars = []
    current_letter = []

    for i, (start, end, is_tone) in enumerate(segments):
        duration = end - start

        if is_tone:
            # Classify as dot or dash based on duration
            if duration < dash_threshold:
                current_letter.append('.')
            else:
                current_letter.append('-')
        else:
            # This is silence - determine what kind
            if duration < symbol_gap_max:
                # Gap within a letter, do nothing
                continue
            elif duration < letter_gap_max:
                # Gap between letters
                if current_letter:
                    morse_chars.append(''.join(current_letter))
                    current_letter = []
            else:
                # Gap between words
                if current_letter:
                    morse_chars.append(''.join(current_letter))
                    current_letter = []
                morse_chars.append(' ')

    # Add final letter if any
    if current_letter:
        morse_chars.append(''.join(current_letter))

    return ' '.join(morse_chars)


def morse_to_text(morse_code):
    """Convert Morse code string to text"""
    words = morse_code.split('   ')  # Words separated by 3 spaces
    decoded_words = []

    for word in words:
        letters = word.split(' ')
        decoded_letters = []

        for letter in letters:
            if letter in MORSE_TO_CHAR:
                decoded_letters.append(MORSE_TO_CHAR[letter])
            elif letter:
                decoded_letters.append('?')  # Unknown character

        decoded_words.append(''.join(decoded_letters))

    return ' '.join(decoded_words)


def decode_morse_audio(file_path):
    """Decode Morse code from MP3 file"""
    print(f"Loading audio from {file_path}...")
    audio = load_audio(file_path)

    print("Analyzing audio segments...")
    segments = detect_segments(audio)

    # Filter out very short segments (noise)
    min_segment_duration = 20  # ms
    segments = [(s, e, t) for s, e, t in segments if (e - s) >= min_segment_duration]

    print(f"Detected {len(segments)} segments")

    print("Converting to Morse code...")
    morse = segments_to_morse(segments)
    print(f"Morse code: {morse}")

    print("Decoding to text...")
    text = morse_to_text(morse)

    return text, morse


def main():
    """Main console application"""
    print("=" * 50)
    print("Morse Code Audio Decoder")
    print("=" * 50)

    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Enter audio file to decode (.mp3 or .wav): ").strip()

    if not input_file:
        print("Error: No file provided")
        sys.exit(1)

    # Add .mp3 extension if no audio extension provided
    if not input_file.endswith('.mp3') and not input_file.endswith('.wav'):
        input_file += '.mp3'

    try:
        text, morse = decode_morse_audio(input_file)
        print("\n" + "=" * 50)
        print("DECODED TEXT:")
        print("=" * 50)
        print(text)
        print("=" * 50)

        # Ask if user wants to save to file
        save = input("\nSave decoded text to file? (y/n): ").strip().lower()
        if save == 'y':
            output_file = input("Enter output filename (default: decoded.txt): ").strip()
            if not output_file:
                output_file = "decoded.txt"

            with open(output_file, 'w') as f:
                f.write(f"Decoded Text: {text}\n")
                f.write(f"Morse Code: {morse}\n")

            print(f"Saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
