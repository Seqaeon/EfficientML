import json

with open("kv_cache_experiments.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Fix SELECTED_MODELS
        if "SELECTED_MODELS =" in source:
            source = source.replace(
                'SELECTED_MODELS = LOCAL_MODELS[2] if LOCAL_MODELS else ["llama3.1:8b"]',
                'SELECTED_MODELS = [LOCAL_MODELS[2]] if len(LOCAL_MODELS) > 2 else ["llama3.1:8b"]'
            )
            
            # also update METHOD_LIST just in case it is in this cell
            source = source.replace(
                'METHOD_LIST = ["fullkv", "commonkv", "minicache", "thinkk", "palu"]',
                'METHOD_LIST = ["fullkv", "commonkv", "minicache", "thinkkv", "palu"]'
            )

            cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]
            if cell["source"][-1] == "":
                cell["source"] = cell["source"][:-1]
                
        # 2. Update Adapters
        if "class FullKV(KVCompressionMethod):" in source:
            new_source = """class FullKV(KVCompressionMethod):
    name = "fullkv"

class CommonKV(KVCompressionMethod):
    name = "commonkv"

    def invoke(self, model: str, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        proxy_url = config.get(f"{self.name}_url")
        if proxy_url:
            payload = {"model": model, "prompt": prompt, "method": self.name, "params": config.get(f"{self.name}_params", {})}
            r = requests.post(proxy_url, json=payload, timeout=config.get("timeout_s", 300))
            r.raise_for_status()
            return r.json()

        url = config.get("ollama_url", "http://localhost:11434/api/generate")
        keep_ratio = config.get("commonkv_keep_ratio", 0.5)
        # We assume the first portion is a common system prompt
        split_idx = max(1, int(len(prompt) * keep_ratio))
        prefix = prompt[:split_idx]
        suffix = prompt[split_idx:]
        
        # Prefill prefix
        prefix_payload = {
            "model": model, "prompt": prefix, "stream": False,
            "options": {"num_predict": 1, **self.ollama_options(config)}
        }
        r_prefix = requests.post(url, json=prefix_payload)
        r_prefix.raise_for_status()
        context = r_prefix.json().get("context", [])
        
        # Generation with suffix and cached context
        payload = {
            "model": model, "prompt": suffix, "context": context, "stream": False,
            "options": self.ollama_options(config)
        }
        r = requests.post(url, json=payload, timeout=config.get("timeout_s", 300))
        r.raise_for_status()
        resp = r.json()
        resp["meta_commonkv"] = "Used Ollama context array for prefix caching"
        return resp

class MiniCache(KVCompressionMethod):
    name = "minicache"

    def invoke(self, model: str, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        proxy_url = config.get(f"{self.name}_url")
        if proxy_url:
            payload = {"model": model, "prompt": prompt, "method": self.name, "params": config.get(f"{self.name}_params", {})}
            r = requests.post(proxy_url, json=payload, timeout=config.get("timeout_s", 300))
            r.raise_for_status()
            return r.json()
            
        head_chars = config.get("minicache_head_chars", 2000)
        tail_chars = config.get("minicache_tail_chars", 2000)
        if len(prompt) <= head_chars + tail_chars:
            transformed = prompt
        else:
            # Seamless concatenation to simulate minicache text eviction
            transformed = prompt[:head_chars] + "\\n\\n...\\n\\n" + prompt[-tail_chars:]
            
        return super().invoke(model, transformed, config)

class ThinkKV(KVCompressionMethod):
    name = "thinkkv"

    def invoke(self, model: str, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        proxy_url = config.get(f"{self.name}_url")
        if proxy_url:
            payload = {"model": model, "prompt": prompt, "method": self.name, "params": config.get(f"{self.name}_params", {})}
            r = requests.post(proxy_url, json=payload, timeout=config.get("timeout_s", 300))
            r.raise_for_status()
            return r.json()
            
        # Natively, we can't evict KV mid-generation in Ollama.
        # We simulate ThinkKV by running it normally but with a reduced num_ctx if configured,
        # or just passing it through with a note that real eviction requires the proxy.
        options = self.ollama_options(config)
        options["num_ctx"] = config.get("thinkkv_num_ctx", 4096)
        
        url = config.get("ollama_url", "http://localhost:11434/api/generate")
        payload = {
            "model": model, "prompt": prompt, "stream": False,
            "options": options
        }
        r = requests.post(url, json=payload, timeout=config.get("timeout_s", 300))
        r.raise_for_status()
        resp = r.json()
        resp["meta_thinkkv"] = "Simulated with constrained num_ctx in Ollama. Use thinkkv_url proxy for true KV eviction."
        return resp

class PaLu(KVCompressionMethod):
    name = "palu"

    def invoke(self, model: str, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        palu_url = config.get("palu_url")
        if palu_url:
            payload = {
                "model": model,
                "prompt": prompt,
                "method": "palu",
                "params": config.get("palu_params", {}),
            }
            r = requests.post(palu_url, json=payload, timeout=config.get("timeout_s", 300))
            r.raise_for_status()
            return r.json()
        return super().invoke(model, prompt, config)

METHODS = {
    "fullkv": FullKV(),
    "commonkv": CommonKV(),
    "minicache": MiniCache(),
    "thinkkv": ThinkKV(),
    "palu": PaLu(),
}

print("Available method adapters:", sorted(METHODS.keys()))"""
            cell["source"] = [line + "\n" for line in new_source.split("\n")[:-1]] + [new_source.split("\n")[-1]]
            if cell["source"][-1] == "":
                cell["source"] = cell["source"][:-1]

with open("kv_cache_experiments.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook.")
