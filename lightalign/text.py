"""Qwen3.5-9B middle-layer token means."""

import os

import numpy as np

from lightalign.pins import QWEN_REPO, QWEN_REVISION

MAX_LENGTH = 1024
TEXT_BATCH = 4


def report_text(findings, impressions):
    """Findings, then a fixed impression header. NaN stays the string 'nan'."""
    return f"{str(findings).strip()}\nIMPRESSION:\n{str(impressions).strip()}"


def finding_prompts(names):
    """Positive prompt, then negative prompt, in finding order."""
    return [p for c in names for p in (f"IMPRESSION:\n{c} is present.", f"IMPRESSION:\nNo {c.lower()}.")]


def load_language_model(model_id, revision):
    """Language model only. The vision tower is deleted after load."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this package does not run the text encoder on CPU")
    kwargs = {}
    if not os.path.isdir(model_id):
        kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="right", **kwargs)
    vlm = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    lm = vlm.model.language_model
    del vlm.model.visual
    return tok, lm.cuda().eval()


def embed_middle(lm, tok, texts, batch, max_length=MAX_LENGTH):
    """Mask-weighted mean of hidden_states[len(hidden_states) // 2].

    Batches follow the ascending character length of the raw strings, with right padding.
    """
    import torch

    if len(texts) == 0:
        raise ValueError("no texts to embed")
    order = np.argsort([len(s) for s in texts])
    hidden = lm.config.get_text_config().hidden_size
    mid = torch.empty(len(texts), hidden)
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = torch.from_numpy(order[s:s + batch])
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to("cuda")
            hs = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    output_hidden_states=True).hidden_states
            m = enc.attention_mask[..., None].float()
            mid[idx] = ((hs[len(hs) // 2].float() * m).sum(1) / m.sum(1)).cpu()
    out = mid.numpy()
    if out.ndim != 2 or not np.isfinite(out).all():
        raise RuntimeError("non-finite text embedding")
    return out


def embed_directory(out, model_id, revision, batch):
    """Write text.npy (one row per cases.jsonl row) and prompts.npy from labels.json."""
    import json

    import pandas as pd

    cases_path = os.path.join(out, "cases.jsonl")
    labels_path = os.path.join(out, "labels.json")
    if not os.path.isfile(cases_path):
        raise FileNotFoundError(f"missing {cases_path}; build the case list first")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"missing {labels_path}; build the case list first")
    cases = pd.read_json(cases_path, lines=True)
    if "report" not in cases.columns:
        raise KeyError("cases.jsonl has no report column")
    prompts = json.load(open(labels_path, encoding="utf-8"))["prompts"]
    if len(prompts) == 0:
        raise ValueError("labels.json has no prompts")
    tok, lm = load_language_model(model_id, revision)
    reports = [str(t) for t in cases["report"].tolist()]
    text = embed_middle(lm, tok, reports, batch)
    if len(text) != len(cases):
        raise RuntimeError(f"text rows {len(text)} != cases {len(cases)}")
    prompt_rows = embed_middle(lm, tok, [str(p) for p in prompts], batch)
    if len(prompt_rows) != len(prompts):
        raise RuntimeError("prompt row count changed")
    _save_npy(os.path.join(out, "text.npy"), text)
    _save_npy(os.path.join(out, "prompts.npy"), prompt_rows)
    print(f"texts n={len(cases)} dim={text.shape[1]} prompts={len(prompt_rows)}", flush=True)


def _save_npy(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)
