import numpy as np
import torch
import os

from configs import global_config

class Smoother():
    def __init__(self, embedding_dir: str, duration: int, window_size: int = 5):
        self.embedding_dir = embedding_dir
        self.duration = duration
        self.window_size = window_size

    def smooth(self):
        mean_embedding_dir = self.embedding_dir.replace('PTI', 'mean')
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(mean_embedding_dir), exist_ok=True)

        device = torch.device(global_config.device)

        if self.window_size % 2 == 0:
            self.window_size += 1

        half_window = self.window_size // 2

        for i in range(self.duration):
            start_window = max(i-half_window, 0)
            finish_window = min(i+half_window+1, self.duration)
            
            windowed_embeddings = torch.tensor([], device=device)

            for j in range(start_window, finish_window):
                embedding_path = self.embedding_dir + f'/{j:04d}/0.pt'
                windowed_embeddings = torch.cat([windowed_embeddings, torch.load(embedding_path, map_location=device)], dim=0)
                
            
            mean_embedding = torch.mean(windowed_embeddings, dim=0).unsqueeze(dim=0)

            embedding_path = mean_embedding_dir + f'/{i:04d}/0.pt'

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

            torch.save(mean_embedding, embedding_path)

        return mean_embedding_dir
