import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append("/Users/minho/Documents/Dataset")

def main():
    print("--- True InfoNCE Multi-Augmentation VAE Dry Run ---")
    try:
        import run_all_adaptive_cnn_true_infonce_multi_aug as ma
        res = ma.run_evaluation("ACSF1_normal_2", epochs=1)
        if res is not None:
            print("Success! True InfoNCE Multi-Augmentation VAE runs perfectly.")
            for k, v in res.items():
                if k != "dataset_name":
                    print(f"  * {k:20s}: {v}")
        else:
            print("Failed: Result is None")
    except Exception as e:
        print(f"Error during dry run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
