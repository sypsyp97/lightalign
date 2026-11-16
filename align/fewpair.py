"""Few-pair baselines on frozen encoders: with the same n image-report pairs, other ways to align the
two frozen spaces, each read by the zero-shot prompt (P) and by the report classifier (TC), plus an image classifier
trained directly on the n paired images with the labels of their reports (no alignment).

Methods give an image map f and a report map g into a shared space:
  ridge       f(x) = x W (ridge on the pairs), g = identity              (ours; P as align.official_matrix)
  procrustes  f(x) = x Q, Q = polar(Xp' Yp), g = identity
  cca         SOTAlign's CCA teacher (lam 0.1, eps 1e-6), f, g = its two projections
  hadgi       Hadgi et al. (CVPR 2025, arXiv 2503.05283): CCA on the pairs to a k = 50 dimensional shared subspace
              (k = n - 1 when n <= 50; SOTAlign's regularised CCA, as cca), then an affine map from the projected
              images to the projected reports fitted by least squares on the pairs; f(x) = [P_x(x), 1] B, g = P_y
  anchors     SOTAlign's Anchors affinity (relative representation over the n pairs, as ASIF), f, g = anchor coordinates
  sotalign    SOTAlign (ICML 2026, github.com/ExplainableML/SOTAlign @ 64c89ee): linear heads to 1,024 dims trained with
              the semi-supervised SigLIP loss on the pairs and KLOT (CCA teacher) on the unpaired remainder of the pool;
              authors' configuration (Lion 1e-4, wd 1e-5, cosine, 10 % warm-up, 2,000 steps, AMP, logit scale 20 /
              bias -10, alpha_sail 1, alpha_klot 1e-3, Sinkhorn eps 0.05 / 0.01, 100 iterations, disjoint unpaired
              images and reports). The last step is used: picking a checkpoint on extra validation pairs would spend
              pairs outside the budget.
  image_probe logistic regression per finding on the n paired image features with their report labels
  image_probe_cv  as image_probe, with the inverse regularisation C chosen per finding by stratified k-fold
              cross-validation (AUROC) on the n pairs over PROBE_CS, k = min(5, minority-class count); a finding
              with fewer than 2 minority examples keeps C = 1
  mlp_infonce MLP on each modality: LayerNorm(d), linear map to 1,024, GELU, dropout 0.1, linear map to 512;
              outputs are L2-normalised. Symmetric InfoNCE on the n pairs only (full batch, AdamW, learning rate
              1e-3, float32, cosine schedule over 1,600 epochs). Logit scale initialised at log(1/0.07), exp clamped
              at 100. Weight decay in {1e-4, 1e-2, 1e-1, 1.0, 3.0, 10.0} and the stopping epoch in {5, 10, 20, 50, 100, 200, 400, 800, 1600} are
              chosen by 5-fold cross-validation on the pairs (lowest mean validation InfoNCE; ties keep the smaller
              weight decay, then the smaller epoch). The refit on all n pairs uses that choice and the same 1,600-epoch
              cosine, stopped at the chosen epoch. No unpaired features and no labels.
P = cosine of f(x) with g(positive prompt) minus with g(negative prompt) (ridge: normalize(x W) (e+ - e-), as align.official_matrix).
TC = logistic regression on g(train reports) with the report labels, applied to f(x).

    python -m align.fewpair heldout  --emb DIR --out DIR --image voxelfm --text textmid_4B --method M
    python -m align.fewpair official --train DIR --valid DIR --ctclip CSV --out DIR --image I --text T --method M --n N
                                     [--rad DIR]

official --rad DIR also applies each seed's fitted map and report classifier, unchanged, to the RAD-ChestCT volumes of
DIR (as align.radchest_eval main: same label merge, reader TC, centring "ctrate" = the CT-RATE image mean of the fit,
mean AUC over the RAD labels) and writes one row per seed to <out>/rad_<method>_n<N>.csv. The other outputs do not change.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize, polar
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.decisive import stratified_auc, test_strata

HELDOUT_NS = (25, 50, 100, 200, 400, 800, 1600)
SOTALIGN_SRC = os.environ.get("SOTALIGN_SRC", "")
METHODS = ("ridge", "procrustes", "cca", "anchors", "sotalign", "image_probe", "image_probe_cv", "hadgi", "mlp_infonce")
PROBE_CS = np.logspace(-4, 2, 13)
HADGI_K = 50
MLP_WDS = (1e-4, 1e-2, 1e-1, 1.0, 3.0, 10.0)
MLP_EPOCHS = (5, 10, 20, 50, 100, 200, 400, 800, 1600)


def _mlp_infonce(xp, yp, seed):
    """Nonlinear maps trained with symmetric InfoNCE on the paired rows of xp, yp only."""
    import math

    import torch
    from sklearn.model_selection import KFold
    from torch import nn

    if len(xp) != len(yp):
        raise ValueError(f"pair count mismatch: {len(xp)} images, {len(yp)} reports")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    horizon = MLP_EPOCHS[-1]

    class Tower(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 512))

        def forward(self, v):
            return nn.functional.normalize(self.net(v), dim=-1)

    class PairMLP(nn.Module):
        def __init__(self, d_img, d_txt):
            super().__init__()
            self.image, self.text = Tower(d_img), Tower(d_txt)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def forward(self, x, y):
            return self.image(x), self.text(y), self.logit_scale.exp().clamp(max=100)

    def infonce(zi, zt, scale):
        if zi.shape[0] != zt.shape[0]:
            raise ValueError(f"InfoNCE batch {zi.shape[0]} vs {zt.shape[0]}")
        logits = scale * (zi @ zt.T)
        target = torch.arange(zi.shape[0], device=zi.device)
        return 0.5 * (nn.functional.cross_entropy(logits, target) + nn.functional.cross_entropy(logits.T, target))

    def eval_loss(model, x, y):
        model.eval()
        with torch.inference_mode():
            return float(infonce(*model(x, y)))

    def fit(x, y, x_val, y_val, wd, n_epochs, record):
        # Epoch t uses the t/1600 point of one cosine. An early stop keeps that prefix, on the folds and on the refit.
        if record and x_val is None:
            raise ValueError("validation features are required when recording epochs")
        torch.manual_seed(seed)
        model = PairMLP(x.shape[1], y.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        xv = yv = None
        if x_val is not None:
            xv = torch.as_tensor(x_val, dtype=torch.float32, device=dev)
            yv = torch.as_tensor(y_val, dtype=torch.float32, device=dev)
        loss0 = eval_loss(model, xt, yt)
        logged = {}
        for epoch in range(1, n_epochs + 1):
            lr = 0.5 * (1.0 + math.cos(math.pi * epoch / horizon)) * 1e-3
            for grp in opt.param_groups:
                grp["lr"] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = infonce(*model(xt, yt))
            loss.backward()
            opt.step()
            model.logit_scale.data.clamp_(max=math.log(100))
            if epoch in record:
                logged[epoch] = eval_loss(model, xv, yv)
        loss1 = eval_loss(model, xt, yt)
        return model, loss0, loss1, logged

    folds = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(len(xp))))
    candidates = []
    for wd in MLP_WDS:
        bucket = {ep: [] for ep in MLP_EPOCHS}
        for tr, va in folds:
            _, _, _, logged = fit(xp[tr], yp[tr], xp[va], yp[va], wd, horizon, MLP_EPOCHS)
            for ep in MLP_EPOCHS:
                bucket[ep].append(logged[ep])
        for ep in MLP_EPOCHS:
            if len(bucket[ep]) != 5:
                raise RuntimeError(f"fold count {len(bucket[ep])} for wd={wd} epochs={ep}")
            mean = sum(bucket[ep]) / len(bucket[ep])
            if not math.isfinite(mean):
                raise RuntimeError(f"non-finite val InfoNCE wd={wd} epochs={ep}")
            candidates.append((mean, wd, ep))
    best_mean, best_wd, best_ep = min(candidates)
    model, loss0, loss1, _ = fit(xp, yp, None, None, best_wd, best_ep, ())
    if not math.isfinite(loss0) or not math.isfinite(loss1):
        raise RuntimeError(f"non-finite train InfoNCE {loss0} -> {loss1}")
    print(f"mlp_infonce wd={best_wd} epochs={best_ep} val_infonce={best_mean:.6f} "
          f"train_loss {loss0:.6f} -> {loss1:.6f}", flush=True)

    def run(fn):
        def f(v):
            model.eval()
            with torch.inference_mode():
                return fn(torch.as_tensor(v, dtype=torch.float32, device=dev)).cpu().numpy()
        return f

    return run(model.image), run(model.text)


def _torch_space(method, xp, yp, xu, yu, seed):
    """cca, hadgi, anchors and sotalign use the SOTAlign repo (X = reports, Y = images). mlp_infonce is the pair MLP."""
    import torch
    if method == "mlp_infonce":
        return _mlp_infonce(xp, yp, seed)
    sys.path.insert(0, SOTALIGN_SRC)
    from optimal_transport.similarity import CCA, Anchors
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    if method == "cca":
        cca = CCA(n_components=None, lam_x=0.1, lam_y=0.1, eps=1e-6)
        cca.fit(t(yp), t(xp))
        proj = lambda v, mean, w: torch.nn.functional.normalize(
            (torch.nn.functional.normalize(t(v), dim=1) - mean) @ w, dim=1).cpu().numpy()
        return (lambda v: proj(v, cca.y_mean, cca.Wy)), (lambda v: proj(v, cca.x_mean, cca.Wx))
    if method == "hadgi":
        cca = CCA(n_components=min(HADGI_K, len(xp) - 1), lam_x=0.1, lam_y=0.1, eps=1e-6)
        cca.fit(t(yp), t(xp))
        proj = lambda v, mean, w: ((torch.nn.functional.normalize(t(v), dim=1) - mean) @ w).cpu().numpy().astype(np.float64)
        px, py = (lambda v: proj(v, cca.y_mean, cca.Wy)), (lambda v: proj(v, cca.x_mean, cca.Wx))
        one = lambda a: np.hstack([a, np.ones((len(a), 1))])
        b = np.linalg.lstsq(one(px(xp)), py(yp), rcond=None)[0]       # affine fit on the raw projected coordinates
        return (lambda v: normalize(one(px(v)) @ b)), (lambda v: normalize(py(v)))
    if method == "anchors":
        anc = Anchors()
        anc.fit(torch.nn.functional.normalize(t(yp), dim=1), torch.nn.functional.normalize(t(xp), dim=1))
        coord = lambda v, mean, a: torch.nn.functional.normalize(
            (torch.nn.functional.normalize(t(v), dim=1) - mean) @ a.T, dim=1).cpu().numpy()
        return (lambda v: coord(v, anc.y_mean, anc.B)), (lambda v: coord(v, anc.x_mean, anc.A))
    if method == "sotalign":
        return _sotalign(xp, yp, xu, yu, seed, t, dev)
    raise ValueError(method)


def _sotalign(xp, yp, xu, yu, seed, t, dev):
    import math
    import torch
    from torch import nn
    from optimal_transport.matching import MatchingModel
    from optimal_transport.similarity import CCA
    from training.optimizer import Lion

    torch.manual_seed(seed)

    class Heads(nn.Module):                      # AlignmentLayer with linear_type "linear" (nn.Linear + LayerNorm)
        def __init__(self, dv, dt, k=1024, scale=20.0, bias=-10.0):
            super().__init__()
            self.vision_layer_norm, self.text_layer_norm = nn.LayerNorm(dv), nn.LayerNorm(dt)
            self.vision_mapping_network, self.text_mapping_network = nn.Linear(dv, k), nn.Linear(dt, k)
            for m in (self.vision_mapping_network, self.text_mapping_network):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor([scale])))
            self.logit_bias = nn.Parameter(torch.tensor([bias]))

        def image(self, v):
            return self.vision_mapping_network(self.vision_layer_norm(v))

        def text(self, v):
            return self.text_mapping_network(self.text_layer_norm(v))

    xp_, yp_, xu_, yu_ = t(xp), t(yp), t(xu), t(yu)
    teacher = CCA(n_components=None, lam_x=0.1, lam_y=0.1, eps=1e-6)
    teacher.fit(torch.nn.functional.normalize(yp_, dim=1), torch.nn.functional.normalize(xp_, dim=1))
    loss = MatchingModel({"alpha_supervised_sail": 0.0, "alpha_supervised_implicit": 0.0,
                          "alpha_semisupervised_sail": 1.0, "alpha_semisupervised_klot": 0.001,
                          "alpha_semisupervised_softmax": 0.0, "alpha_semisupervised_cka": 0.0,
                          "epsilon_sinkhorn_student": 0.05, "n_iters_sinkhorn_student": 100,
                          "epsilon_sinkhorn_teacher": 0.01, "n_iters_sinkhorn_teacher": 100,
                          "use_stop_criterion": False})
    loss.affinity = teacher
    model = Heads(xp.shape[1], yp.shape[1]).to(dev)
    opt = Lion(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99))
    steps, warm = 2000, 200
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    per_side = min(len(xu), len(yu), 32768 - len(xp))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        lr = 1e-4 * (step + 1) / warm if step < warm else \
            0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm))) * 1e-4
        for grp in opt.param_groups:
            grp["lr"] = lr
        iu = torch.randperm(len(xu), generator=g)[:per_side].to(dev)
        tu = torch.randperm(len(yu), generator=g)[:per_side].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            total, _ = loss(X=yu_[tu], Y=xu_[iu], X_pairs=yp_, Y_pairs=xp_,
                            fX=model.text(yu_[tu]), fY=model.image(xu_[iu]),
                            fX_pairs=model.text(yp_), fY_pairs=model.image(xp_),
                            logit_scale=model.logit_scale.exp(), logit_bias=model.logit_bias)
        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()
    model.eval()

    def run(fn):
        def f(v):
            with torch.no_grad():
                return torch.nn.functional.normalize(fn(t(v)).float(), dim=1).cpu().numpy()
        return f
    return run(model.image), run(model.text)


def reader_scores(method, xp, yp, xu, yu, xe, yr, lr_, prompts, seed):
    """Scores (eval volumes x findings) of P and TC for one method; xe = evaluation image features, yr / lr_ = all
    train reports and their labels (for TC), prompts = interleaved positive / negative prompt embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tc = lambda gy, fx: np.stack([make_pipeline(StandardScaler(), LogisticRegression()).fit(gy, lr_[:, j])
                                  .decision_function(fx) for j in range(lr_.shape[1])], 1)
    if method == "ridge":                     # as align.official_matrix: P = cosine score, TC on u
        u = xe @ ridge(xp, yp).coef_.T
        return {"P": normalize(u) @ (prompts[0::2] - prompts[1::2]).T, "TC": tc(yr, u)}
    if method == "procrustes":
        f, g = (lambda v, q=polar(xp.T @ yp): normalize(v @ q)), (lambda v: normalize(v))
    else:
        f, g = _torch_space(method, xp, yp, xu, yu, seed)
    fe, gp = f(xe), g(prompts)
    return {"P": fe @ (gp[0::2] - gp[1::2]).T, "TC": tc(g(yr), fe)}


def probe_scores(xp, lp, xe, tune=False):
    """Image classifier on the n paired images; a finding with one class among them gets a constant score.
    tune: choose C per finding by stratified cross-validation on the n pairs (image_probe_cv)."""
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out, chosen = [], []
    for j in range(lp.shape[1]):
        if lp[:, j].min() == lp[:, j].max():
            out.append(np.zeros(len(xe)))
            continue
        minority = int(min(lp[:, j].sum(), len(lp) - lp[:, j].sum()))
        if tune and minority >= 2:
            clf = LogisticRegressionCV(Cs=PROBE_CS, cv=StratifiedKFold(min(5, minority), shuffle=True, random_state=0),
                                       scoring="roc_auc", max_iter=5000)
        else:
            clf = LogisticRegression()
        pipe = make_pipeline(StandardScaler(), clf).fit(xp, lp[:, j])
        out.append(pipe.decision_function(xe))
        chosen.append(float(pipe[-1].C_[0]) if hasattr(pipe[-1], "C_") else 1.0)
    if tune:
        print(json.dumps({"n": len(lp), "chosen_C": chosen}), flush=True)
    return {"IL": np.stack(out, 1)}


def _load(emb, image, text, cases):
    x = normalize(np.stack([np.load(f"{emb}/{image}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{emb}/{text}.npy").astype(np.float64))
    p = normalize(np.load(f"{emb}/{text.replace('text', 'prompts', 1)}.npy").astype(np.float64))
    assert len(x) == len(y) == len(cases)
    return x, y, p


def _scores(a, x_pool, y_pool, lab_pool, xe, prompts_raw, n, seed, pool_idx):
    perm = np.random.default_rng(seed).permutation(pool_idx)
    pairs, rest = perm[:n], perm[n:]
    mx, my = x_pool[pool_idx].mean(0), y_pool[pool_idx].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    if a.method in ("image_probe", "image_probe_cv"):
        return probe_scores(fx(x_pool[pairs]), lab_pool[pairs], fx(xe), tune=a.method == "image_probe_cv")
    half = len(rest) // 2                                            # disjoint unpaired images and reports
    return reader_scores(a.method, fx(x_pool[pairs]), fy(y_pool[pairs]), fx(x_pool[rest[:half]]),
                         fy(y_pool[rest[half:]]), fx(xe), fy(y_pool[pool_idx]), lab_pool[pool_idx],
                         fy(prompts_raw), seed)


def heldout(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x, y, pr = _load(a.emb, a.image, a.text, cases)
    lab_all = cases[names].to_numpy(float)
    lab, group = lab_all[test], test_strata(cases, lab_all[test], test)
    rows = []
    for seed in SEEDS:
        for n in HELDOUT_NS:
            for r, sc in _scores(a, x, y, lab_all, x[test], pr, n, seed, train).items():
                rows.append({"method": a.method, "image": a.image, "text": a.text, "seed": seed, "n": n, "reader": r,
                             "mean_auc": float(np.nanmean(auc_matrix(sc, lab))),
                             "strat_auc": float(np.nanmean(stratified_auc(sc, lab, group)))})
                print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/heldout_{a.method}__{a.image}__{a.text}.csv", index=False)


def official(a):
    from align.official import units
    from align.official_matrix import BOOT, DROP16, patient_resamples
    names = label_names(a.train)
    assert names == label_names(a.valid)
    tc_ = pd.read_json(f"{a.train}/cases.jsonl", lines=True)
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    xt, yt, pr = _load(a.train, a.image, a.text, tc_)
    xv = normalize(np.stack([np.load(f"{a.valid}/{a.image}/{os.path.basename(k)}.npy") for k in vc["key"]]).astype(np.float64))
    lt, lv = tc_[names].to_numpy(float), vc[names].to_numpy(float)
    keep16 = np.array([l not in DROP16 for l in names])
    ct = pd.read_csv(a.ctclip, index_col="VolumeName").loc[[f"{i}.nii.gz" for i in vc["id"]], names].to_numpy(float)
    pool = np.arange(len(tc_))
    seeds = SEEDS if a.n < len(pool) else [0]
    xe, rad_tc = xv, []
    if a.rad:                                    # RAD volumes ride behind the validation rows through the same fitted map
        from align.external import images
        from align.radchest_eval import rad_cases, rad_encoder
        rc, rnames = rad_cases(a.rad, a.train)
        xr = images(a.rad, rad_encoder(a.image), rc["key"])
        xe = np.vstack([xv, xr])
    per_seed = {}
    for seed in seeds:
        for r, sc in _scores(a, xt, yt, lt, xe, pr, a.n, seed, pool).items():
            if a.rad and r == "TC":
                rad_tc.append(sc[len(xv):])
            per_seed.setdefault(r, []).append(sc[:len(xv)])
    if "P" in per_seed and "TC" in per_seed:     # image_probe returns IL only; thresh_metrics reads P and TC
        os.makedirs(a.out, exist_ok=True)
        np.savez(f"{a.out}/official_{a.method}__{a.image}__{a.text}__n{a.n}_scores.npz",
                 ids=vc["id"].to_numpy(), P=np.stack(per_seed["P"]), TC=np.stack(per_seed["TC"]))
    patients = vc["id"].str.split("_").str[1].to_numpy()
    rows = []
    for un, mk in units(vc).items():
        idx = np.flatnonzero(mk)
        res = patient_resamples(patients[idx])
        for lname, cols in (("mean_auc", np.ones(len(names), bool)), ("mean_auc16", keep16)):
            draws = lambda sc: np.array([np.mean([np.nanmean(auc_matrix(s[idx][j], lv[idx][j])[cols]) for s in sc])
                                         for j in res])
            d_ct = draws([ct])
            for r, scs in per_seed.items():
                diff = draws(scs) - d_ct
                rows.append({"method": a.method, "image": a.image, "text": a.text, "n": a.n, "unit": un,
                             "metric": lname, "reader": r,
                             "auc": float(np.mean([np.nanmean(auc_matrix(s[idx], lv[idx])[cols]) for s in scs])),
                             "minus_ctclip_lo": float(np.percentile(diff, 2.5)),
                             "minus_ctclip_hi": float(np.percentile(diff, 97.5)), "boot": BOOT})
                print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/official_{a.method}__{a.image}__{a.text}__n{a.n}.csv", index=False)
    if a.rad:
        from align.radchest_eval import merge_columns, merge_scores
        cols = merge_columns(names, rnames)
        lr = rc[rnames].to_numpy(float)
        rad_rows = []
        for seed, sc in zip(seeds, rad_tc):
            auc = auc_matrix(merge_scores(sc, cols), lr)
            rad_rows.append({"method": a.method, "n": a.n, "seed": seed, "encoder": a.image, "model": a.text,
                             "centring": "ctrate", "volumes": len(rc), "auc": float(np.nanmean(auc)),
                             **{f"auc|{l}": float(v) for l, v in zip(rnames, auc)}})
            print(json.dumps({k: v for k, v in rad_rows[-1].items() if "|" not in k}), flush=True)
        pd.DataFrame(rad_rows).to_csv(f"{a.out}/rad_{a.method}_n{a.n}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["heldout", "official"])
    ap.add_argument("--emb")
    ap.add_argument("--train")
    ap.add_argument("--valid")
    ap.add_argument("--ctclip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--rad")
    a = ap.parse_args()
    if a.rad and a.step != "official":
        ap.error("--rad needs the official step")
    {"heldout": heldout, "official": official}[a.step](a)


if __name__ == "__main__":
    main()
