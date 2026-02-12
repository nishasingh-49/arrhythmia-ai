
from dataloader import load_record


def main():
    signal, r_locs, labels, fs = load_record("100")

    print("\n--- RECORD INFO ---")
    print("Signal length:", len(signal))
    print("Sampling rate:", fs)
    print("Number of beats:", len(r_locs))
    print("First 10 R locations:", r_locs[:10])
    print("First 10 labels:", labels[:10])


if __name__ == "__main__":
    main()
