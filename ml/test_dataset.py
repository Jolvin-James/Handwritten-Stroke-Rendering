import torch
from ml.dataset import StrokeDataset

def test_pipeline():
    print("--- Initializing Dataset ---")
    dataset = StrokeDataset(data_dir="data/raw", max_seq_len=128)

    if len(dataset) == 0:
        print("Error: No strokes found. Check your 'data/raw' folder.")
        return

    print(f"\n--- Success! Dataset Size: {len(dataset)} samples ---")

    input_tensor, target_tensor = dataset[0]

    print("\n--- Sample 0 Inspection ---")
    print(f"Input Shape (Noisy):  {input_tensor.shape}")
    print(f"Target Shape (Clean): {target_tensor.shape}")
    
    print("\n--- Feature Check (First 5 points) ---")
    print("Input Features: [x, y, dt, p, v]")
    print(input_tensor[:5].numpy())
    
    expected_input_features = 5 
    expected_output_features = 2 
    
    assert input_tensor.shape[1] == expected_input_features, "Input features mismatch!"
    assert target_tensor.shape[1] == expected_output_features, "Target features mismatch!"
    print("\nData shapes match PRD requirements.")

if __name__ == "__main__":
    test_pipeline()