
from dataloader import load_record
from beat_extractor import extract_beats
from label_processor import binary_map
def main():
    signal, r_locs, labels, fs = load_record("100")
    X, y_letters = extract_beats(signal, r_locs, labels)
    y = binary_map(y_letters)
    print("\n--- AFTER LABEL PROCESSING ---")
    print("Unique labels:", set(y))
    print("First 20:", y[:20])
    print("Abnormal count:", sum(y))
if __name__ == "__main__":
    main()
