# ml/dataset.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class StrokeDataset(Dataset):
    """
    Dataset for ML based handwritten stroke smoothing.
    
    Generates synthetic noise on-the-fly to create (Noisy, Clean) pairs.
    """

    def __init__(self, data_dir, max_seq_len=128, noise_level=0.02):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.noise_level = noise_level
        
        self.samples = []
        self._load_data()

    def _load_data(self):
        """
        Loads all JSON files and flattens them into a list of individual strokes.
        Fixes the issue of only reading the first stroke.
        """
        files = sorted([
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith(".json")
        ])

        for file_path in files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                # Store every valid stroke from the file
                for stroke in data.get("strokes", []):
                    points = stroke.get("points", [])
                    if len(points) > 2:
                        # Pre-calculate features once to save time
                        processed = self._process_stroke(points)
                        if processed is not None:
                            self.samples.append(processed)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(self.samples)} valid strokes from {len(files)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Get the Ground Truth (Clean)
        clean_stroke = self.samples[idx] # Shape: (seq_len, 5)
        
        # 2. Generate Input (Noisy)
        noisy_stroke = self._add_synthetic_noise(clean_stroke)
        
        # 3. Return Pair (Input, Target)
        return torch.tensor(noisy_stroke, dtype=torch.float32), \
               torch.tensor(clean_stroke[:, :2], dtype=torch.float32)

    def _add_synthetic_noise(self, stroke_array):
        """
        Adds Gaussian noise to x and y coordinates to simulate hand jitter.
        """
        noisy = stroke_array.copy()
    
        # 1. Random Rotation (±15 degrees)
        theta = np.radians(np.random.uniform(-15, 15))
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
    
        # Rotate (x, y) coordinates
        noisy[:, :2] = np.dot(noisy[:, :2], rotation_matrix)

        # 2. Random Scaling (±20%)
        scale_factor = np.random.uniform(0.8, 1.2)
        noisy[:, :2] *= scale_factor
    
        # 3. Add existing Gaussian Jitter
        noise = np.random.normal(0, self.noise_level, size=(stroke_array.shape[0], 2))
        noisy[:, :2] += noise
    
        return noisy
    
    def _process_stroke(self, points):
        x = np.array([p["x"] for p in points], dtype=np.float32)
        y = np.array([p["y"] for p in points], dtype=np.float32)
        t = np.array([p["t"] for p in points], dtype=np.float32)
        
        # Default pressure to 1.0 if not present or 0
        p_list = [p.get("p", 1.0) for p in points]
        p = np.array(p_list, dtype=np.float32)

        # Bounding box normalization (Preserving Aspect Ratio) 
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Prevent division by zero
        if width < 1e-6 and height < 1e-6:
            return None
            
        # Scale by the largest dimension to keep the shape correct
        scale = max(width, height) + 1e-6
        
        x_norm = (x - min_x) / scale
        y_norm = (y - min_y) / scale

        # Center normalization (Shift to 0.0 center)
        cx = x_norm.mean()
        cy = y_norm.mean()
        x_centered = x_norm - cx
        y_centered = y_norm - cy

        # Derived Features 
        dt = np.zeros_like(t)
        dt[1:] = t[1:] - t[:-1]
        dt_max = dt.max()
        dt_norm = dt / (dt_max + 1e-6) if dt_max > 0 else dt

        # Standardize pressure
        p_mean = p.mean()
        p_std = p.std() + 1e-6
        p_norm = (p - p_mean) / p_std

        # Velocity (Calculated on normalized coords)
        dx = np.zeros_like(x_centered)
        dy = np.zeros_like(y_centered)
        dx[1:] = x_centered[1:] - x_centered[:-1]
        dy[1:] = y_centered[1:] - y_centered[:-1]
        
        speed = np.sqrt(dx**2 + dy**2)
        speed = speed / (speed.max() + 1e-6)

        features = np.stack([x_centered, y_centered, dt_norm, p_norm, speed], axis=1)

        # Resample 
        return self._resample(features, self.max_seq_len)

    def _resample(self, sequence, target_len):
        seq_len = len(sequence)
        if seq_len == target_len:
            return sequence

        indices = np.linspace(0, seq_len - 1, target_len)
        resampled = np.zeros((target_len, sequence.shape[1]), dtype=np.float32)

        for i in range(sequence.shape[1]):
            resampled[:, i] = np.interp(
                indices,
                np.arange(seq_len),
                sequence[:, i]
            )
        return resampled