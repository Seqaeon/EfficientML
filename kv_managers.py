import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import DynamicCache

class MiniCache(DynamicCache):
    """
    Implements a StreamingLLM-style Sliding Window + Attention Sink cache.
    Retains the first `n_sink` tokens globally, and the last `n_recent` tokens.
    Automatically evicts the middle tokens from the KV cache to maintain O(1) memory footprint.
    """
    def __init__(self, n_sink: int = 4, n_recent: int = 1024):
        super().__init__()
        self.n_sink = n_sink
        self.n_recent = n_recent

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Apply eviction policy
        current_seq_len = keys.shape[-2]
        max_capacity = self.n_sink + self.n_recent
        
        if current_seq_len > max_capacity:
            # We must evict middle tokens: from n_sink to (current_seq_len - n_recent)
            keep_indices = list(range(self.n_sink)) + list(range(current_seq_len - self.n_recent, current_seq_len))
            self.layers[layer_idx].keys = keys[..., keep_indices, :]
            self.layers[layer_idx].values = values[..., keep_indices, :]

        return self.layers[layer_idx].keys, self.layers[layer_idx].values

class ThinkKVCache(DynamicCache):
    """
    Simulates "ThinKV" or thought-adaptive KV cache compression.
    During generation, this cache analyzes token importance of recent entries
    (e.g., L2 norm of Keys) and aggressively prunes the bottom P% of intermediate 
    tokens to simulate evaporating shallow/redundant chains-of-thought.
    """
    def __init__(self, n_sink: int = 4, prune_ratio: float = 0.5, prune_every: int = 128):
        super().__init__()
        self.n_sink = n_sink
        self.prune_ratio = prune_ratio
        self.prune_every = prune_every
        self.step_counter = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        current_seq_len = keys.shape[-2]
        
        if layer_idx == 0:
            self.step_counter += key_states.shape[-2] # count new tokens
            
        if self.step_counter > self.prune_every and current_seq_len > self.n_sink + 64:
            # It's time to prune!
            # We avoid pruning the sink and the very recent 64 tokens.
            prune_start = self.n_sink
            prune_end = current_seq_len - 64
            
            if prune_end > prune_start + 10:
                # Calculate importance (e.g., L2 norm of the key vectors over the head dimension)
                importance = keys[..., prune_start:prune_end, :].norm(p=2, dim=-1).mean(dim=1) # mean across heads
                
                # We want to drop `prune_ratio` of these tokens
                num_to_drop = int((prune_end - prune_start) * self.prune_ratio)
                _, drop_indices = torch.topk(importance, k=num_to_drop, largest=False, dim=-1)
                
                mask = torch.ones(current_seq_len, dtype=torch.bool, device=key_states.device)
                mask[prune_start + drop_indices[0]] = False # batched implementation assumes batch=1 for now
                
                self.layers[layer_idx].keys = keys[:, :, mask, :]
                self.layers[layer_idx].values = values[:, :, mask, :]
                
            if layer_idx == len(self.layers) - 1:
                self.step_counter = 0

        return self.layers[layer_idx].keys, self.layers[layer_idx].values

class CommonKVCache(DynamicCache):
    """
    Implements Prefix / Common KV caching. You pre-initialize this cache with a fixed 
    prefix tensor sequence (the common prompt), and it permanently locks those positions.
    """
    def __init__(self, prefix_key_cache: List[torch.Tensor], prefix_value_cache: List[torch.Tensor]):
        super().__init__()
        self.prefix_len = prefix_key_cache[0].shape[-2] if prefix_key_cache else 0
        for k, v in zip(prefix_key_cache, prefix_value_cache):
            self.update(k, v, len(self.layers))
        
    def reset_to_prefix(self):
        """Discards all generated tokens and reverts exactly to the shared prefix."""
        for layer in self.layers:
            layer.keys = layer.keys[..., :self.prefix_len, :]
            layer.values = layer.values[..., :self.prefix_len, :]
