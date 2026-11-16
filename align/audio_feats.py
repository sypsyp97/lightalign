"""Self-supervised audio encoders for the Clotho domain, each with its official recipe. Kept
free of other align imports so it runs in its own environment (transformers 4.51.3, torchaudio),
because the EAT remote code does not load under transformers 5.

    eat       worstchan/EAT-base_epoch30_pretrain (pretrain only, no labels): Kaldi fbank (128 mels, htk, hanning, 10 ms,
              dither 0) of the mean-removed 16 kHz waveform, padded/cut to 1,024 frames per 10 s window,
              (x + 4.268) / (2 * 4.569); utterance (CLS) token of extract_features; mean over the clip's windows.
    eat_mean  the same forward pass, mean over frame tokens instead of the CLS token (saved together with eat).
    dasheng   mispeech/dasheng-base (no labels): its feature extractor on the whole 16 kHz clip; the mean over time of
              outputs.hidden_states (the model card's logits with outputdim=None are sigmoid of this mean, so we take
              the mean before the sigmoid).

    python -m align.audio_feats --src DIR --out DIR --encoder eat|dasheng --model-dir DIR [--smoke N]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

SR = 16000
EAT_FRAMES = 1024
EAT_MEAN, EAT_STD = -4.268, 4.569


def load(path):
    import soundfile as sf
    import torchaudio

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.numel() == 0 or not torch.isfinite(wav).all():
        raise ValueError(f"bad audio {path}")
    return wav


def eat_windows(wav):
    """Official EAT input for each 10 s window of the clip: [n_windows, 1, 1024, 128]."""
    import torchaudio

    n = 10 * SR
    mels = []
    for s in range(0, len(wav), n):
        w = wav[s:s + n]
        if len(w) < 400:                    # shorter than one fbank frame (a clip tail)
            if mels:
                continue
            w = torch.nn.functional.pad(w, (0, 400 - len(w)))
        w = w - w.mean()
        m = torchaudio.compliance.kaldi.fbank(w[None], htk_compat=True, sample_frequency=SR, use_energy=False,
                                              window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
        m = torch.nn.functional.pad(m, (0, 0, 0, max(0, EAT_FRAMES - len(m))))[:EAT_FRAMES]
        mels.append((m - EAT_MEAN) / (EAT_STD * 2))
    return torch.stack(mels)[:, None]


@torch.no_grad()
def main():
    from transformers import AutoFeatureExtractor, AutoModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["eat", "dasheng"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="only the first N clips, print shapes, save nothing")
    a = ap.parse_args()
    cases = pd.read_json(f"{a.out}/cases.jsonl", lines=True)
    if a.smoke:
        cases = cases.head(a.smoke)
    names = ["eat", "eat_mean"] if a.encoder == "eat" else ["dasheng"]
    for n in names:
        os.makedirs(f"{a.out}/{n}", exist_ok=True)
    if a.encoder == "dasheng":
        fe = AutoFeatureExtractor.from_pretrained(a.model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(a.model_dir, outputdim=None, trust_remote_code=True).cuda().eval()
    else:
        model = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True).cuda().eval()
    done = 0
    for key, f in zip(cases["key"], cases["file"]):
        paths = [f"{a.out}/{n}/{key}.npy" for n in names]
        if not a.smoke and all(os.path.exists(p) for p in paths):
            continue
        wav = load(f"{a.src}/{f}")
        if a.encoder == "dasheng":
            x = fe(wav[None], sampling_rate=SR, return_tensors="pt")
            zs = [model(**{k: v.cuda() for k, v in x.items()}).hidden_states[0].mean(0)]
        else:
            h = model.extract_features(eat_windows(wav).cuda())          # [windows, 1 + frames, 768]
            zs = [h[:, 0].mean(0), h[:, 1:].mean(1).mean(0)]
        for n, p, z in zip(names, paths, zs):
            z = z.float().cpu().numpy()
            if z.shape != (768,) or not np.isfinite(z).all():
                raise RuntimeError(f"{n} {key}: bad feature {z.shape}")
            if a.smoke:
                print(f"[smoke] {n} {key} len={len(wav) / SR:.1f}s dim={z.shape} norm={np.linalg.norm(z):.3f}", flush=True)
                continue
            np.save(p + ".tmp.npy", z)
            os.replace(p + ".tmp.npy", p)
        done += 1
    print(f"[audio] {a.encoder} saved {done} of {len(cases)}", flush=True)


if __name__ == "__main__":
    main()
