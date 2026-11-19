"""Fixed open-weight model. Only public assets download; text inference is on-server."""
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time

from .h3_ai_completion import SYSTEM

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
MODEL_WEIGHTS_SHA256 = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
SETTINGS = dict(device='cpu', dtype='bfloat16', max_input_tokens=2048,
                max_new_tokens=220, do_sample=False, seed=20261119, threads=2)
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_model(cache_dir=None):
    """Download a pinned public artifact before supplying any research context."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    folder = snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False,
        cache_dir=cache_dir, max_workers=2,
        allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                        'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'])
    weights_hash = sha256()
    with (Path(folder)/'model.safetensors').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            weights_hash.update(block)
    if weights_hash.hexdigest() != MODEL_WEIGHTS_SHA256:
        raise ValueError('Downloaded weights do not match the pinned model hash.')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager')
    model.to('cpu').eval()
    metadata = dict(model_id=MODEL_ID, revision=MODEL_REVISION,
        weights_sha256=weights_hash.hexdigest(),
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        settings=SETTINGS, inference_location='application_server_cpu',
        external_text_inference=False, generated_outputs_are_human_labels=False)
    return tokenizer, model, metadata


def generator(cache_dir=None):
    tokenizer, model, metadata = load_model(cache_dir)

    def generate(task):
        import torch
        payload = dict(county=task['county'], meeting_date=task['meeting_date'],
            question='What exact coalition-meeting headcount is explicitly documented?',
            evidence=[dict(chunk_id=d['chunk_id'], source_file=d['source_file'], page=d['page'],
                           text=d['text']) for d in task['evidence']])
        prompt = tokenizer.apply_chat_template([
            {'role':'system','content':SYSTEM}, {'role':'user','content':json.dumps(payload)}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] > SETTINGS['max_input_tokens']:
            # Never silently truncate evidence or the extraction contract.
            raise ValueError('Retrieved context exceeds the frozen input-token limit.')
        with _LOCK:
            threads = torch.get_num_threads()
            torch.set_num_threads(SETTINGS['threads'])
            started = time.perf_counter()
            try:
                torch.manual_seed(SETTINGS['seed'])
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=SETTINGS['max_new_tokens'],
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                        use_cache=True)
                generated = ids[0, inputs['input_ids'].shape[1]:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
            finally:
                torch.set_num_threads(threads)
        return dict(raw_response=raw, input_sha256=sha256(prompt.encode()).hexdigest(),
                    input_tokens=int(inputs['input_ids'].shape[1]), output_tokens=int(len(generated)),
                    elapsed_seconds=time.perf_counter()-started,
                    finish_reason='token_limit' if len(generated)==SETTINGS['max_new_tokens'] else 'eos')

    return generate, dict(metadata)
