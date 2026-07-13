import os
import sys

# Add Dataset path to sys path to import files
sys.path.append("/Users/minho/Documents/Dataset")

def test_script(module_name, func_name="run_evaluation"):
    print(f"\n--- Testing Module: {module_name} ---")
    try:
        # Import dynamically
        mod = __import__(module_name)
        func = getattr(mod, func_name)
        
        # Dry run on a simple dataset
        test_dataset = "ACSF1_normal_2"
        print(f"Running '{func_name}' for '{test_dataset}' on 1 epoch...")
        
        # We override epochs dynamically or call with 1 epoch if supported
        res = func(test_dataset, epochs=1)
        
        if res is not None:
            print(f"Success! Metrics computed:")
            for k, v in res.items():
                if k != "dataset_name":
                    print(f"  * {k:20s}: {v}")
        else:
            print("Failed: Result is None")
    except Exception as e:
        print(f"Error encountered during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    # List of files we want to test
    modules_to_test = [
        "run_all_adaptive_cnn_dynamic_quantile",
        "run_all_adaptive_cnn_kl_annealing",
        "run_all_adaptive_cnn_contrastive",
        "run_all_adaptive_transformer_vae"
    ]
    
    for mod in modules_to_test:
        test_script(mod)

if __name__ == "__main__":
    main()
