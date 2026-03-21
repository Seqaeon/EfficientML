import torch
import time
import argparse
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Import our custom caches
from kv_managers import MiniCache, ThinkKVCache, CommonKVCache

# --- SYNTHETIC RULER GENERATORS ---
def generate_ruler(task="ns", haystack_size=3000):
    filler = "The quick brown fox jumps over the lazy dog. "
    haystack = filler * (haystack_size // len(filler))
    text = haystack
    
    if task == "ns":
        needle_str = " The secret passkey is 83749. "
        insert_pos = len(text) // 2
        text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: What is the secret passkey? The secret passkey is", ["83749"]
        
    elif task == "nmk":
        needles = ["83749", "12948"]
        for i, pk in enumerate(needles):
            needle_str = f" The magic code {i} is {pk}. "
            insert_pos = (len(text) // 3) * (i + 1)
            text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: What are the magic codes? The magic codes are", needles
        
    elif task == "nmq":
        needles = {"Alpha": "83749", "Beta": "12948"}
        for i, (k, v) in enumerate(needles.items()):
            needle_str = f" Project {k} relies on code {v}. "
            insert_pos = (len(text) // 3) * (i + 1)
            text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: What code does Project Beta rely on? Project Beta relies on", ["12948"]
        
    elif task == "nmv":
        needle_str = " The passkeys are 111, 222, and 333. "
        insert_pos = len(text) // 2
        text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: What are the passkeys? The passkeys are", ["111", "222", "333"]
        
    elif task == "ruler_qa":
        needle_str = " The capital city of the fictional land of Gorp is Zentopia. "
        insert_pos = len(text) // 2
        text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: What is the capital of Gorp? The capital is", ["Zentopia"]
        
    elif task == "vt":
        needle_str = " Variable X was initialized to 42. "
        insert_pos = len(text) // 3
        text = text[:insert_pos] + needle_str + text[insert_pos:]
        needle_str2 = " Later, Variable X was updated to 99. "
        insert_pos2 = (len(text) // 3) * 2
        text = text[:insert_pos2] + needle_str2 + text[insert_pos2:]
        return text + "\nQuestion: What is the final value of Variable X? The value is", ["99"]
        
    elif task == "fwe":
        needle_str = " pineapple " * 10
        insert_pos = len(text) // 2
        text = text[:insert_pos] + needle_str + text[insert_pos:]
        return text + "\nQuestion: Which word appears most frequently in the inserted text? The word is", ["pineapple"]

# --- LONGBENCH HF LOADER ---
def get_longbench_task(task_name="qa", max_samples=2):
    try:
        from datasets import load_dataset
        dataset_name = "THUDM/LongBench"
        task_mapping = {
            "qa": "qasper", 
            "sum": "gov_report", 
            "code": "repobench-p", 
            "fshot": "hotpotqa",
            "synth": "passage_count"
        }
        subset = task_mapping.get(task_name, "qasper")
        print(f"Loading LongBench subset: {subset} from HuggingFace...")
        
        ds = load_dataset(dataset_name, subset, split="test", trust_remote_code=True)
        samples = []
        for i in range(min(max_samples, len(ds))):
            item = ds[i]
            prompt = f"Please read the following text and answer the question.\n\nContext:\n{item['context']}\n\nQuestion: {item['input']}\nAnswer:"
            expected = item["answers"] if "answers" in item else [item.get("answer", "")]
            samples.append((prompt, expected))
        return samples
    except Exception as e:
        print(f"Dataset load failed for {task_name}: {e}")
        return []

def evaluate(model, tokenizer, prompt, expected, cache_type="fullkv", config=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Initialize cache
    if cache_type == "fullkv":
        past_key_values = DynamicCache()
    elif cache_type == "minicache":
        past_key_values = MiniCache(n_sink=4, n_recent=config.get("minicache_window", 512))
    elif cache_type == "thinkkv":
        past_key_values = ThinkKVCache(n_sink=4, prune_ratio=0.5, prune_every=128)
    else:
        past_key_values = DynamicCache()
        
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    latency = time.perf_counter() - start_t
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    accuracy = 1.0 if any(str(exp).lower() in generated_text.lower() for exp in expected) else 0.0
    
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    final_kv_len = past_key_values.get_seq_length() if getattr(past_key_values, "get_seq_length", None) else 0
        
    return {
        "cache_type": cache_type,
        "accuracy": accuracy,
        "latency_s": latency,
        "peak_vram_mb": peak_mem_mb,
        "final_kv_tokens": final_kv_len,
        "input_tokens": inputs.input_ids.shape[-1],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-0.5B")
    parser.add_argument("--task", type=str, default="all", help="all, or specific task e.g. qa, ns")
    args = parser.parse_args()
    
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    
    LONGBENCH_TASKS = ["qa", "sum", "fshot", "synth", "code"]
    RULER_TASKS = ["ns", "nmk", "nmq", "nmv", "ruler_qa", "vt", "fwe"]
    ALL_TASKS = LONGBENCH_TASKS + RULER_TASKS
    
    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    results = []
    
    for task in tasks_to_run:
        print(f"\n[{task.upper()}] Preparing...")
        if task in LONGBENCH_TASKS:
            samples = get_longbench_task(task_name=task, max_samples=2)
        else:
            prompt, expected = generate_ruler(task=task, haystack_size=3000)
            samples = [(prompt, expected)]
            
        for i, (prompt, expected) in enumerate(samples):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if input_ids.shape[-1] > 6000:
                prompt = tokenizer.decode(input_ids[0, :6000], skip_special_tokens=True)
                
            for cache_type in ["fullkv", "minicache", "thinkkv"]:
                config = {"minicache_window": 512} if cache_type == "minicache" else None
                res = evaluate(model, tokenizer, prompt, expected, cache_type, config)
                res["task"] = task
                results.append(res)

    df = pd.DataFrame(results)
    
    # 1. Output a Pivot Table like the CommonKV paper (Cols = Tasks, Rows = Cache_Type, Val = Accuracy)
    print("\n" + "="*80)
    print("                  LONGBENCH & RULER ACCURACY (%) TABLE")
    print("="*80)
    
    # Group and calculate mean accuracy * 100 for paper-like format
    summary_df = df.groupby(["cache_type", "task"])["accuracy"].mean().unstack() * 100
    
    # Sort columns by Paper order
    paper_order = LONGBENCH_TASKS + RULER_TASKS
    existing_cols = [c for c in paper_order if c in summary_df.columns]
    summary_df = summary_df[existing_cols]
    
    # Add an Average column
    summary_df.insert(0, "Avg.", summary_df.mean(axis=1))
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_df.round(1).to_string())
    print("="*80)
    
    # 2. Output Hardware Metrics Table (Latency, Memory)
    print("\nHardware Overhead Tradeoffs (Averages across all tasks):")
    hw_df = df.groupby("cache_type")[["latency_s", "peak_vram_mb", "final_kv_tokens"]].mean()
    print(hw_df.round(2).to_string())
    
    df.to_csv("artifacts/benchmark_full_report.csv", index=False)
    print("\nSaved comprehensive report to artifacts/benchmark_full_report.csv")

if __name__ == "__main__":
    main()
