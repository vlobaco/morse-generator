#!/usr/bin/env python3
"""
Improved timing detection using natural breaks algorithm
"""
import numpy as np


def detect_timing_thresholds_improved(tone_durations, silence_durations):
    """
    Improved timing detection that handles noise and multiple gap types
    Uses natural breaks (Jenks) algorithm to find clusters
    """

    if not tone_durations or not silence_durations:
        # Fall back to defaults
        return 250, 150, 500

    # ===== PART 1: Detect dot vs dash threshold =====
    dash_threshold = detect_tone_threshold(tone_durations)

    # ===== PART 2: Detect silence gap thresholds =====
    symbol_gap_max, letter_gap_max = detect_silence_thresholds(silence_durations)

    return dash_threshold, symbol_gap_max, letter_gap_max


def detect_tone_threshold(tone_durations):
    """Find threshold between dots and dashes"""
    if len(tone_durations) < 2:
        return 250

    unique_tones = sorted(set(tone_durations))

    if len(unique_tones) == 1:
        # Only one tone length - unusual, use it as dot
        return unique_tones[0] * 2

    # Find the largest gap between consecutive unique values
    gaps = [(unique_tones[i+1] - unique_tones[i], i)
            for i in range(len(unique_tones) - 1)]

    if not gaps:
        return np.median(tone_durations)

    # The largest gap is likely between dots and dashes
    max_gap, max_gap_idx = max(gaps, key=lambda x: x[0])

    # Threshold is midpoint of the largest gap
    threshold = (unique_tones[max_gap_idx] + unique_tones[max_gap_idx + 1]) / 2

    return threshold


def detect_silence_thresholds(silence_durations):
    """
    Find thresholds for symbol gaps, letter gaps, and word gaps
    Uses natural breaks to handle noise and multiple gap types
    """
    if len(silence_durations) < 3:
        # Not enough data
        if len(silence_durations) == 1:
            base = silence_durations[0]
            return base * 1.5, base * 3
        elif len(silence_durations) == 2:
            return (silence_durations[0] + silence_durations[1]) / 2, silence_durations[1] * 1.5
        else:
            return 150, 500

    unique_silences = sorted(set(silence_durations))

    if len(unique_silences) <= 3:
        # 3 or fewer unique values - use them directly
        if len(unique_silences) == 3:
            symbol_gap_max = (unique_silences[0] + unique_silences[1]) / 2
            letter_gap_max = (unique_silences[1] + unique_silences[2]) / 2
        elif len(unique_silences) == 2:
            symbol_gap_max = (unique_silences[0] + unique_silences[1]) / 2
            letter_gap_max = unique_silences[1] * 1.5
        else:
            symbol_gap_max = unique_silences[0] * 1.5
            letter_gap_max = unique_silences[0] * 3

        return symbol_gap_max, letter_gap_max

    # More than 3 unique values - need to cluster them into 3 groups
    # Use natural breaks: find the 2 largest gaps to split into 3 groups

    gaps = [(unique_silences[i+1] - unique_silences[i], i)
            for i in range(len(unique_silences) - 1)]

    # Sort gaps by size (descending)
    gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)

    # Take the 2 largest gaps - these divide the data into 3 groups
    largest_gaps = sorted([g[1] for g in gaps_sorted[:2]])

    # Split into 3 groups at these gaps
    group1 = unique_silences[:largest_gaps[0] + 1]
    group2 = unique_silences[largest_gaps[0] + 1:largest_gaps[1] + 1]
    group3 = unique_silences[largest_gaps[1] + 1:]

    print(f"  Group 1 (symbol gaps): {group1}")
    print(f"  Group 2 (letter gaps): {group2}")
    print(f"  Group 3 (word gaps): {group3}")

    # Use median of each group as representative value
    symbol_gap_representative = np.median(group1) if group1 else 100
    letter_gap_representative = np.median(group2) if group2 else 300
    word_gap_representative = np.median(group3) if group3 else 700

    # Thresholds are midpoints between groups
    symbol_gap_max = (symbol_gap_representative + letter_gap_representative) / 2
    letter_gap_max = (letter_gap_representative + word_gap_representative) / 2

    return symbol_gap_max, letter_gap_max


# ===== TEST CODE =====

def test_improved():
    print("=" * 70)
    print("TESTING IMPROVED ALGORITHM")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("SCENARIO 1: Noisy audio with small variations")
    print("=" * 70)
    tone_durations = [100] * 20 + [300] * 15
    silence_durations = [100, 102, 98, 101, 99, 300, 305, 298, 700, 710, 695]

    dash_th, symbol_max, letter_max = detect_timing_thresholds_improved(
        tone_durations, silence_durations
    )
    print(f"\nResults:")
    print(f"  Dot/Dash threshold: {dash_th:.1f}ms")
    print(f"  Symbol gap max: {symbol_max:.1f}ms")
    print(f"  Letter gap max: {letter_max:.1f}ms")

    print("\n" + "=" * 70)
    print("SCENARIO 2: Multiple word gap lengths")
    print("=" * 70)
    tone_durations = [200] * 30 + [600] * 25
    silence_durations = [200, 200, 200, 600, 600, 1400, 1400, 3000, 5000]

    dash_th, symbol_max, letter_max = detect_timing_thresholds_improved(
        tone_durations, silence_durations
    )
    print(f"\nResults:")
    print(f"  Dot/Dash threshold: {dash_th:.1f}ms")
    print(f"  Symbol gap max: {symbol_max:.1f}ms")
    print(f"  Letter gap max: {letter_max:.1f}ms")

    print("\n" + "=" * 70)
    print("SCENARIO 3: Random noise creating many unique values")
    print("=" * 70)
    tone_durations = [100] * 30 + [300] * 20
    silence_durations = [50, 75, 100, 125, 150, 200, 250, 500, 800, 1500, 2000]

    dash_th, symbol_max, letter_max = detect_timing_thresholds_improved(
        tone_durations, silence_durations
    )
    print(f"\nResults:")
    print(f"  Dot/Dash threshold: {dash_th:.1f}ms")
    print(f"  Symbol gap max: {symbol_max:.1f}ms")
    print(f"  Letter gap max: {letter_max:.1f}ms")

    print("\n" + "=" * 70)
    print("SCENARIO 4: Real morse.wav data")
    print("=" * 70)
    tone_durations = [243] * 39 + [723] * 20
    silence_durations = [237] * 34 + [717] * 18 + [1677] * 6

    dash_th, symbol_max, letter_max = detect_timing_thresholds_improved(
        tone_durations, silence_durations
    )
    print(f"\nResults:")
    print(f"  Dot/Dash threshold: {dash_th:.1f}ms")
    print(f"  Symbol gap max: {symbol_max:.1f}ms")
    print(f"  Letter gap max: {letter_max:.1f}ms")


if __name__ == "__main__":
    test_improved()
