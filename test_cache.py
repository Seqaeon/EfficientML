import torch
from transformers.cache_utils import DynamicCache

class TestCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # Evict everything except the first token
        self.layers[layer_idx].keys = keys[:, :, :1, :]
        self.layers[layer_idx].values = values[:, :, :1, :]
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

c = TestCache()
k1 = torch.zeros(1, 1, 2, 1)
v1 = torch.zeros(1, 1, 2, 1)
k2 = torch.zeros(1, 1, 3, 1)
v2 = torch.zeros(1, 1, 3, 1)

out_k1, _ = c.update(k1, v1, 0)
print(f"Update 1 shape: {out_k1.shape}, Internal shape: {c.layers[0].keys.shape}")
out_k2, _ = c.update(k2, v2, 0)
print(f"Update 2 shape: {out_k2.shape}, Internal shape: {c.layers[0].keys.shape}")
