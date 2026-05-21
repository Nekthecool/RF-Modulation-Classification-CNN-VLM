import torch
import torch.nn as nn

# ====================================================================
# HYBRID DEEP MULTI-TASK CNN ARCHITECTURE
# ====================================================================
class DeepMultiTaskCNN(nn.Module):
    def __init__(self):
        super(DeepMultiTaskCNN, self).__init__()
        
        # Feature Extractor (Backbone)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flattened_dim = 256 * 4 * 4

        # Classification Heads
        self.head_mod = nn.Sequential(
            nn.Linear(self.flattened_dim, 1024), nn.ReLU(), nn.Dropout(0.4), 
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), 
            nn.Linear(256, 16)
        )
        self.head_snr = nn.Sequential(
            nn.Linear(self.flattened_dim, 128), nn.ReLU(), nn.Dropout(0.3), 
            nn.Linear(128, 5)
        )

        # Regression Heads (Huber Loss compatible)
        self.head_pn = nn.Sequential(
            nn.Linear(self.flattened_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.head_iqi = nn.Sequential(
            nn.Linear(self.flattened_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.head_amp_dist = nn.Sequential(
            nn.Linear(self.flattened_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
        self.head_jamming = nn.Sequential(
            nn.Linear(self.flattened_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        return {
            'mod': self.head_mod(x), 
            'pn': self.head_pn(x), 
            'iqi': self.head_iqi(x), 
            'snr': self.head_snr(x), 
            'jamming': self.head_jamming(x), 
            'amp_dist': self.head_amp_dist(x)
        }