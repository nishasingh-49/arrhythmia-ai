# data_loader.py

import wfdb


def load_record(record_name="100"):
    """
    Loads ECG signal and annotation from MIT-BIH database.
    Returns:
        signal  -> numpy array of ECG values
        r_locs  -> positions of R peaks
        labels  -> beat labels
        fs      -> sampling frequency
    """

    print(f"Loading record {record_name} ...")

    # Load ECG waveform
    record = wfdb.rdrecord(record_name, pn_dir="mitdb")

    # Load annotations
    annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    signal = record.p_signal[:, 0]
    r_locs = annotation.sample
    labels = annotation.symbol
    fs = record.fs
    return signal, r_locs, labels, fs
