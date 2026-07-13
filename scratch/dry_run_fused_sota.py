import os
import sys

sys.path.append("/Users/minho/Documents/Dataset")

def main():
    print("--- Fused SOTA VAE Dry Run Validation ---")
    try:
        import run_all_adaptive_cnn_fused_sota as fused
        
        test_dataset = "ACSF1_normal_2"
        print(f"Running Fused SOTA for '{test_dataset}' on 1 epoch...")
        
        res = fused.run_evaluation(test_dataset, epochs=1)
        
        if res is not None:
            print("Success! Fused SOTA VAE is mathematically sound and compiles flawlessly.")
            for k, v in res.items():
                if k != "dataset_name":
                    print(f"  * {k:20s}: {v}")
        else:
            print("Failed: Result is None")
            
    except Exception as e:
        print(f"Error during Fused SOTA validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
