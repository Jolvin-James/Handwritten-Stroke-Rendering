# ml/model.py
import torch
import torch.nn as nn

class StrokeLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=2):
        """
        LSTM-based regression model for stroke smoothing.
        
        Args:
            input_size (int): 5 features (x, y, dt, pressure, velocity)
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of stacked LSTM layers
            output_size (int): 2 coordinates (x, y) for the smoothed path
        """
        super(StrokeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True -> Input shape: (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully Connected Layer: Maps hidden states to coordinate predictions
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
        """
        # Validate input shape for robustness
        if x.dim() == 2:
            x = x.unsqueeze(0) # Handle single sample inference case

        # LSTM Output
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Map to 2D coordinates
        # predictions shape: (batch_size, seq_len, output_size)
        predictions = self.fc(lstm_out)
        
        return predictions

    def save(self, path="model.pth"):
        """Helper to save model weights."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path="model.pth", device="cpu"):
        """Helper to load model weights."""
        model = StrokeLSTM()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model