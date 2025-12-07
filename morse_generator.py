#!/usr/bin/env python3
"""
Morse Code Audio Generator
Converts text to Morse code and generates an MP3 file with the audio
"""

import sys
from pydub import AudioSegment
from pydub.generators import Sine

# Morse code mapping
MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
    '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
    ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
    '"': '.-..-.', '$': '...-..-', '@': '.--.-.', ' ': ' '
}

# Timing constants (in milliseconds)
DOT_DURATION = 100  # Duration of a dot
DASH_DURATION = DOT_DURATION * 3  # Duration of a dash (3 times dot)
SYMBOL_GAP = DOT_DURATION  # Gap between dots and dashes within a letter
LETTER_GAP = DOT_DURATION * 3  # Gap between letters
WORD_GAP = DOT_DURATION * 7  # Gap between words

# Audio parameters
FREQUENCY = 800  # Frequency of the tone in Hz
SAMPLE_RATE = 44100  # Sample rate for audio


def text_to_morse(text):
    """Convert text to Morse code"""
    morse = []
    for char in text.upper():
        if char in MORSE_CODE:
            morse.append(MORSE_CODE[char])
        elif char == ' ':
            morse.append(' ')
    return ' '.join(morse)


def generate_tone(duration, frequency=FREQUENCY):
    """Generate a sine wave tone"""
    return Sine(frequency).to_audio_segment(duration=duration)


def generate_silence(duration):
    """Generate silence"""
    return AudioSegment.silent(duration=duration)


def morse_to_audio(morse_code):
    """Convert Morse code string to audio"""
    audio = AudioSegment.empty()

    words = morse_code.split('   ')  # Words are separated by 3 spaces in Morse

    for word_idx, word in enumerate(words):
        letters = word.split(' ')  # Letters are separated by single space

        for letter_idx, letter in enumerate(letters):
            for symbol_idx, symbol in enumerate(letter):
                if symbol == '.':
                    audio += generate_tone(DOT_DURATION)
                elif symbol == '-':
                    audio += generate_tone(DASH_DURATION)

                # Add gap between symbols within a letter
                if symbol_idx < len(letter) - 1:
                    audio += generate_silence(SYMBOL_GAP)

            # Add gap between letters
            if letter_idx < len(letters) - 1:
                audio += generate_silence(LETTER_GAP)

        # Add gap between words
        if word_idx < len(words) - 1:
            audio += generate_silence(WORD_GAP)

    return audio


def save_morse_audio(text, output_file):
    """Convert text to Morse code audio and save as MP3 or WAV"""
    print(f"Converting text to Morse code: '{text}'")

    # Convert text to Morse code
    morse = text_to_morse(text)
    print(f"Morse code: {morse}")

    # Generate audio
    print("Generating audio...")
    audio = morse_to_audio(morse)

    # Determine format from extension
    if output_file.endswith('.wav'):
        audio_format = "wav"
    else:
        audio_format = "mp3"

    # Export to file
    print(f"Saving to {output_file}...")
    audio.export(output_file, format=audio_format)
    print(f"Successfully created {output_file}")


def main():
    """Main console application"""
    print("=" * 50)
    print("Morse Code Audio Generator")
    print("=" * 50)

    # Get text input
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = input("Enter text to convert to Morse code: ")

    if not text.strip():
        print("Error: No text provided")
        sys.exit(1)

    # Get output filename
    default_output = "morse_output.mp3"
    output_file = input(f"Enter output filename [.mp3 or .wav] (default: {default_output}): ").strip()

    if not output_file:
        output_file = default_output

    # Ensure proper audio extension
    if not output_file.endswith('.mp3') and not output_file.endswith('.wav'):
        output_file += '.mp3'

    # Generate and save
    try:
        save_morse_audio(text, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
