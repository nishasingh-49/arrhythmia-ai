import numpy as np
def extract_beats(signal, r_locs, labels, window=100):
    """
    Converts full ECG into individual beats.
    For each R-peak:
        take 'window' samples before
        take 'window' samples after
    Returns:
        X -> beat signals
        y -> labels
    """
    beats = []
    beat_labels = []
    for r, lab in zip(r_locs, labels):
        start = r - window
        end = r + window
        if start < 0 or end >= len(signal):
            continue
        segment = signal[start:end]
        beats.append(segment)
        beat_labels.append(lab)
    X = np.array(beats)
    y = np.array(beat_labels)
    return X, y
