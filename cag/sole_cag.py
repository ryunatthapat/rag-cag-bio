import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Device and dtype selection
def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32

# 3. Load model and tokenizer
def load_model_and_tokenizer(model_name, hf_token):
    print(f"[CAG] Loading tokenizer and model: {model_name}")
    device, dtype = get_device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    model.to(device)
    print(f"[CAG] Model loaded on device: {device}")
    return tokenizer, model, device

_tokenizer, _model, _device = load_model_and_tokenizer(MODEL_NAME, HF_TOKEN)

# 4. Load knowledge source (document.txt)
def load_knowledge_for_cag(path="document.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

print("[CAG] Loading knowledge...")
faq_text = load_knowledge_for_cag()
print(f"[CAG] Loaded ({len(faq_text)} characters)")

# 5. Create system prompt
print("[CAG] Preparing system prompt...")
system_prompt = f"""
<|system|>
You are an assistant who provides concise factual answers.
<|user|>
Context:
{faq_text}
Question:
""".strip()

# 6. Build KV cache
def get_kv_cache(model, tokenizer, prompt: str):
    print("[CAG] Building KV cache for context...")
    t1 = time.time()
    device = model.model.embed_tokens.weight.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache()
    with torch.no_grad():
        _ = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    t2 = time.time()
    print(f"[CAG] KV cache built. Length: {input_ids.shape[-1]}. Time: {t2-t1:.2f}s")
    return cache, input_ids.shape[-1]

_kv_cache, _origin_len = get_kv_cache(_model, _tokenizer, system_prompt)

# 7. Clean up cache
def clean_up(cache: DynamicCache, origin_len: int):
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
        cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

# 8. Generate function
def generate(model, input_ids, past_key_values, max_new_tokens=300):
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(
                input_ids=next_token, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            past_key_values = out.past_key_values
            next_token = token.to(device)
            if model.config.eos_token_id is not None and token.item() == model.config.eos_token_id:
                break
    return output_ids[:, origin_len:]

# 9. CLI chatbot loop
if __name__ == "__main__":
    print("\n=== PALO IT CAG Chatbot ===")
    print("ถามอะไรก็ได้เกี่ยวกับ PALO IT (พิมพ์ 'exit' เพื่อออก)\n")
    while True:
        query = input("> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("ลาก่อน!")
            break
        if not query:
            continue
        clean_up(_kv_cache, _origin_len)
        input_ids = _tokenizer(query + "\n", return_tensors="pt").input_ids.to(_device)
        start_gen = time.time()
        output_ids = generate(_model, input_ids, _kv_cache, max_new_tokens=100)
        end_gen = time.time()
        answer = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n[AI] {answer.strip()} (ตอบใน {end_gen - start_gen:.2f} วินาที)\n")
