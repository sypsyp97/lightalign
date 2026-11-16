"""Official zero-pair aligner of Schnaus et al. (arXiv 2610.09411, github.com/dominik-schnaus/unpaired-rosetta @ fdfad84,
Wasserstein Procrustes with geometric initialization, authors' defaults C = 30, S = 30, b = 10,000, R = 100), run with
the authors' own split and metrics (disjoint halves, FOSCTTM by unpaired_rosetta.evaluation.retrieval_metrics).

    hf:   the authors' released COCO 2014 embeddings (positive control; their DINOv2 B/14 x MPNet reports ~0.04).
    ours: one of our embedding directories (COCO or CT-RATE): halves of the train split, scored on the test split
          (FOSCTTM, and the mean AUC of the paired zero-shot prompts, as align.budget amplify).

    UNPAIRED_ROSETTA_ROOT=DIR python -m align.rosetta hf   --vision M --language M [--train-size N] --out DIR
    python -m align.rosetta ours --emb DIR --image E --text T [--train-size N] --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

N_SEEDS = 3


def _aligner():
    """The S = 30 restarts of the geometric initialization run in one process per allocated core; the authors' code
    gives the same map for any number of workers (each restart carries its own seeds)."""
    from unpaired_rosetta import WassersteinProcrustes
    return WassersteinProcrustes(num_workers=min(30, len(os.sched_getaffinity(0))), num_threads=4)


def run_hf(a):
    from unpaired_rosetta.embeddings import embedding_path, load_rows, num_rows
    from unpaired_rosetta.evaluation import disjoint_split, retrieval_metrics
    from unpaired_rosetta.randomness import SEEDS, Randomness

    rows = []
    for seed in SEEDS[:N_SEEDS]:
        rnd = Randomness(seed)
        px, py = (embedding_path("coco_train2014", m, n) for m, n in (("vision", a.vision), ("language", a.language)))
        ix, iy = disjoint_split(num_rows(px), rnd, a.train_size)
        aligner = _aligner().fit(load_rows(px, ix), load_rows(py, iy), randomness=rnd)
        vrnd = Randomness(seed)                         # as experiments.alignment.evaluate / load_validation_data
        vpx, vpy = (embedding_path("coco_val2014", m, n) for m, n in (("vision", a.vision), ("language", a.language)))
        idx = vrnd.subset(num_rows(vpx), None)
        m = retrieval_metrics(aligner, load_rows(vpx, idx), load_rows(vpy, idx), vrnd)
        rows.append({"data": "coco_hf", "image": a.vision, "text": a.language, "seed": int(seed),
                     "half": int(len(ix)), "foscttm": m["foscttm"]})
        print(json.dumps(rows[-1]), flush=True)
    _save(a, rows, f"hf__{a.vision}__{a.language}__{a.train_size or 'all'}")


def run_ours(a):
    from align.align import foscttm, label_names
    from align.baselines import auc_matrix, zero_shot_scores
    from unpaired_rosetta.evaluation import disjoint_split, retrieval_metrics
    from unpaired_rosetta.randomness import SEEDS, Randomness, derive_seeds

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    sel = np.flatnonzero(cases.split == "sel")
    if a.ho_device == "cuda" and not (a.sign_ho and torch.cuda.is_available()):
        raise RuntimeError("--ho-device cuda needs --sign-ho and a CUDA device")
    if a.sign_ho and len(sel) == 0:
        raise ValueError(f"--sign-ho needs the sel split in {a.emb}/cases.jsonl")
    if a.rot and a.rot != a.sign_ho:
        raise ValueError(f"--rot {a.rot} needs --sign-ho with the same K (got {a.sign_ho})")
    seeds = {"dev": SEEDS[:N_SEEDS], "confirm": derive_seeds(4242, 3), "confirm2": derive_seeds(4343, 3)}[a.seed_set]
    if a.seed_index is not None:    # one seed only: its row equals that seed's row of the full run (per-seed RNG)
        seeds = [seeds[a.seed_index]]
    unit = lambda v: torch.nn.functional.normalize(torch.as_tensor(v, dtype=torch.float32), dim=-1)
    xr = np.stack([np.load(f"{a.emb}/{a.image}/{os.path.basename(k)}.npy") for k in cases["key"]])
    yr = np.load(f"{a.emb}/{a.text}.npy")
    pr = np.load(f"{a.emb}/{a.text.replace('text', 'prompts', 1)}.npy")
    info, uy = {}, None
    if a.scale is not None:
        try:
            xr, yr, pr, info, uy = _scale_shared(cases, train, xr, yr, pr, a.scale, a.random, a.symmetrize)
        except ValueError as e:     # no significant shared component: nothing to rescale
            if "no significant shared component" not in str(e):
                raise
            if a.scale != 1 or a.random or a.sign_fix:
                print(f"SKIP {a.image} x {a.text}: k = 0, the intervention is undefined", flush=True)
                return
            info = {"scale": 1.0, "k": 0, "f_image": 0.0, "f_report": 0.0}
    if a.prep and (a.sign_fix or a.orient_diag):
        raise SystemExit("--prep changes the feature coordinates; --sign-fix / --orient-diag need the raw ones")
    if a.prep:                      # pair-free locate step: each modality transformed with its own train items only
        xr, yr, pr = _prep(a.prep, xr, yr, pr, train)
        info["prep"] = a.prep
    x, y, prompts = unit(xr), unit(yr), unit(pr)
    lab = cases[names].to_numpy(float)[test]
    auc = lambda u, p: float(np.nanmean(auc_matrix(zero_shot_scores(u, p), lab))) if names else float("nan")   # label-free domains
    rows = []
    for seed in seeds:
        rnd = Randomness(seed)
        ix, iy = disjoint_split(len(train), rnd, a.train_size)
        ix, iy = torch.as_tensor(train)[ix], torch.as_tensor(train)[iy]
        aligner = _aligner().fit(x[ix], y[iy], randomness=rnd)
        m = retrieval_metrics(aligner, x[test], y[test], Randomness(seed))
        u = torch.nn.functional.normalize(aligner.transform(x[test]), dim=-1).numpy()   # row norm, as budget.amplify
        p = torch.nn.functional.normalize(prompts - aligner.mean_y, dim=-1).numpy()   # the aligner's Y coordinates
        rows.append({"data": os.path.basename(a.emb.rstrip("/")), "image": a.image, "text": a.text, "seed": int(seed),
                     "half": int(len(ix)), "foscttm": m["foscttm"],
                     "mean_auc": auc(u, p), **info})
        if a.sign_select:   # zero-pair: choose the signs of the map's top directions by the GW loss on the train halves
            rows[-1].update(_sign_select(aligner, x[ix].numpy(), y[iy].numpy(), x[test], y[test], p, auc, a.sign_select,
                                         seed))
        if a.sign_ho:       # zero-pair: six pair-free criteria over the same 2^K sign patterns
            rows[-1].update(_sign_ho(aligner, x[ix].numpy(), y[iy].numpy(), x[sel].numpy(), y[sel].numpy(),
                                     x[test], y[test], p, auc, a.sign_ho, seed, device=a.ho_device))
        if a.rot:           # zero-pair: rotation inside the shared subspace from cumulant matching, plus a refit
            rows[-1].update(_rot_fix(aligner, x[ix], y[iy], x[test], y[test], p, auc, a.rot, seed))
        if a.sign_greedy:   # zero-pair, scales past 2^k: steepest single flips of the top directions by GW loss
            rows[-1].update(_sign_greedy(aligner, x[ix].numpy(), y[iy].numpy(), x[test], y[test], p, auc,
                                         a.sign_greedy, seed))
        if a.sign_skew:     # zero-pair, closed form: match the skew sign of mapped images and reports per direction
            rows[-1].update(_sign_skew(aligner, x[ix].numpy(), y[iy].numpy(), x[test], y[test], p, auc, a.sign_skew))
        if a.sign_fix:      # oracle: flip each report shared direction whose test correlation is negative
            yt = torch.nn.functional.normalize(y[test] - aligner.mean_y, dim=-1).numpy()
            corr = [float(np.corrcoef(u @ uy[:, j], yt @ uy[:, j])[0, 1]) for j in range(uy.shape[1])]
            neg = uy[:, [c < 0 for c in corr]]
            uf = u - 2 * (u @ neg) @ neg.T
            uf /= np.linalg.norm(uf, axis=1, keepdims=True)
            rows[-1].update({"dir_corr": corr, "n_flipped": int(neg.shape[1]), "mean_auc_signfix": auc(uf, p),
                             "foscttm_signfix": foscttm(uf @ yt.T),
                             "foscttm_before_fix": foscttm(u @ yt.T)})   # same scoring as foscttm_signfix
            if a.orient_diag:
                rows[-1].update(_orient_diag(aligner, x[ix].numpy(), y[iy].numpy(), x[test], y[test], uy, seed))
        print(json.dumps(rows[-1]), flush=True)
    tag = f"{os.path.basename(a.emb.rstrip('/'))}__{a.image}__{a.text}__{a.train_size or 'all'}"
    if a.scale is not None:
        tag += f"__s{a.scale:g}{'_random' if a.random else ''}{'_signfix' if a.sign_fix else ''}{'_sym' if a.symmetrize else ''}"
    if a.prep:
        tag += "__" + a.prep.replace(":", "")
    _save(a, rows, tag + (f"__select{a.sign_select}" if a.sign_select else "")
          + (f"__greedy{a.sign_greedy}" if a.sign_greedy else "") + (f"__skew{a.sign_skew}" if a.sign_skew else "")
          + (f"__ho{a.sign_ho}" if a.sign_ho else "") + (f"__rot{a.rot}" if a.rot else "")
          + {"dev": "", "confirm": "__confirm", "confirm2": "__confirm2"}[a.seed_set]
          + (f"__seed{a.seed_index}" if a.seed_index is not None else "")
          + ("__diag" if a.orient_diag else ""))


def _sign_select(aligner, xa, yb, xt, yt, p, auc, k, seed, n=1500):
    """Zero-pair orientation: flip any subset of the top-k right singular directions of the mapped half-A images
    (2^k candidates) and keep the candidate whose hard coupling (assignment of mapped half-A images to half-B reports)
    has the lowest Gromov-Wasserstein loss (cosine distances, square loss, m points per side). The selection sees no
    pair; ties go to the earlier candidate. Only after the choice are test pairs used to score it; the test FOSCTTM of
    the best and the worst candidate are reported to show how much orientation is at stake. On --scale runs the
    features were rescaled along oracle (paired) directions, so those cells are diagnostics, not zero-pair results."""
    import itertools
    from concurrent.futures import ThreadPoolExecutor

    from scipy.optimize import linear_sum_assignment
    from threadpoolctl import threadpool_limits

    from align.align import foscttm

    rng = np.random.default_rng(seed)
    m = min(n, len(xa), len(yb))
    xs, ys = xa[rng.choice(len(xa), m, replace=False)], yb[rng.choice(len(yb), m, replace=False)]
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    tx = lambda v: unit(aligner.transform(torch.as_tensor(v, dtype=torch.float32)).numpy().astype(np.float64))
    my = aligner.mean_y.double().numpy()[0]
    ty = lambda v: unit(np.asarray(v, np.float64) - my)
    us, vs, ut, vt = tx(xs), ty(ys), tx(xt), ty(yt)
    if not 0 < k <= min(us.shape):
        raise ValueError(f"--sign-select {k} outside 1..{min(us.shape)}")
    c1, c2 = 1 - us @ us.T, 1 - vs @ vs.T          # a flip is orthogonal: c1 does not change, the assignment does
    b = np.linalg.svd(us - us.mean(0), full_matrices=False)[2][:k].T
    cands = list(itertools.product((1, -1), repeat=k))
    flips = [b[:, [s < 0 for s in signs]] for signs in cands]
    flip = lambda u, neg: unit(u - 2 * (u @ neg) @ neg.T)
    def gw_loss(neg):                                                   # GW loss of the uniform permutation coupling
        _, c = linear_sum_assignment(-(flip(us, neg) @ vs.T))
        return float(np.mean((c1 - c2[np.ix_(c, c)]) ** 2))

    # one BLAS thread per candidate thread, so the pool does not oversubscribe the cores; map keeps candidate order
    with threadpool_limits(1, user_api="blas"), ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        losses = list(pool.map(gw_loss, flips))
        best = int(np.argmin(losses))                                   # first minimum: fixed tie order
        chosen = flip(ut, flips[best])
        test_fos = list(pool.map(lambda neg: foscttm(flip(ut, neg) @ vt.T), flips))   # reporting only, after the choice
    from scipy.stats import spearmanr
    o = int(np.argmin(test_fos))
    return {"select_k": k, "select_m": m, "select_foscttm": test_fos[best], "select_auc": auc(chosen, p),
            "select_n_flipped": int(flips[best].shape[1]), "select_loss": losses[best],
            "select_loss_spread": max(losses) - losses[best], "oracle_best_foscttm": min(test_fos),
            "oracle_worst_foscttm": max(test_fos),
            # does the GW loss rank the sign patterns like their test FOSCTTM? (diagnostic, after the choice)
            "select_rho_loss_fos": float(spearmanr(losses, test_fos).statistic),
            "oracle_best_loss": losses[o], "oracle_best_loss_rank": int((np.array(losses) < losses[o]).sum()),
            "oracle_best_n_flipped": int(flips[o].shape[1])}


def _sign_ho(aligner, xa, yb, xsel, ysel, xt, yt, p, auc, k, seed, n=1500, device="cpu", chunk=64):
    """Pair-free scoring of the 2^k sign patterns of _sign_select (same basis b: top-k right singular directions of the
    mapped half-A images, same subsample, same candidates) by six criteria. ins_* use the train halves the aligner was
    fitted on; ho_* use sel images from one random half of sel and sel reports from the disjoint other half.
    *_full: GW loss of the hard coupling with cosine distances (as _sign_select). *_sub: the same loss with Euclidean
    distances of the projections onto b; a flip changes the sign of the projection coordinates, so c1 is invariant and
    only the assignment changes. *_energy: cross term of the energy distance, mean Euclidean distance between the
    flipped projected images and the projected reports (the within-sample terms are flip-invariant); lower is better.
    No criterion reads a pair. Test pairs only score the choices: per criterion the FOSCTTM and AUC of its chosen
    pattern, the number of flips, the Spearman correlation of the criterion with the test FOSCTTM over candidates and
    the rank of the oracle-best pattern under the criterion.
    device = "cuda": the same quantities in float64 torch on the GPU, `chunk` candidates per batch, the exact assignment
    by torch_linear_assignment (Triton Hungarian) on the GPU. A flip is a reflection by the orthonormal columns of b, so
    the flipped unit vectors stay unit and the flipped similarities are base - 2 (u b * neg) (b^T v^T): a rank-k
    update of the unflipped similarity, no d-wide matmul per candidate. Distances are exact (no matmul shortcut)."""
    import itertools
    from concurrent.futures import ThreadPoolExecutor

    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    from threadpoolctl import threadpool_limits

    from align.align import foscttm

    rng = np.random.default_rng(seed)
    m = min(n, len(xa), len(yb))
    xs, ys = xa[rng.choice(len(xa), m, replace=False)], yb[rng.choice(len(yb), m, replace=False)]
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    tx = lambda v: unit(aligner.transform(torch.as_tensor(v, dtype=torch.float32)).numpy().astype(np.float64))
    my = aligner.mean_y.double().numpy()[0]
    ty = lambda v: unit(np.asarray(v, np.float64) - my)
    us, vs, ut, vt = tx(xs), ty(ys), tx(xt), ty(yt)
    if not 0 < k <= min(us.shape):
        raise ValueError(f"--sign-ho {k} outside 1..{min(us.shape)}")
    b = np.linalg.svd(us - us.mean(0), full_matrices=False)[2][:k].T
    cands = list(itertools.product((1, -1), repeat=k))
    flips = [b[:, [s < 0 for s in signs]] for signs in cands]
    flip = lambda u, neg: unit(u - 2 * (u @ neg) @ neg.T)

    # held-out unpaired sel items: disjoint halves, images from one and reports from the other; own rng
    rho = np.random.default_rng([seed, 1])
    perm = rho.permutation(len(xsel))
    h = len(xsel) // 2
    ia, ib = perm[:h], perm[h:2 * h]
    m_ho = min(n, h)
    uh = tx(xsel[ia[rho.choice(h, m_ho, replace=False)]])
    vh = ty(ysel[ib[rho.choice(h, m_ho, replace=False)]])

    if device == "cpu":
        sets = {}
        for name, (u, v) in {"ins": (us, vs), "ho": (uh, vh)}.items():
            ub, vb = u @ b, v @ b
            sets[name] = {"u": u, "v": v, "ub": ub, "vb": vb, "c1": 1 - u @ u.T, "c2": 1 - v @ v.T,
                          "d2": cdist(vb, vb)}

        def gw(c1, c2, score):                                  # GW loss of the uniform permutation coupling
            _, c = linear_sum_assignment(-score)
            return float(np.mean((c1 - c2[np.ix_(c, c)]) ** 2))

        def criteria(i):
            neg, sgn = flips[i], np.array(cands[i])
            out = {}
            for name, d in sets.items():
                ubf = d["ub"] * sgn
                out[f"{name}_full"] = gw(d["c1"], d["c2"], flip(d["u"], neg) @ d["v"].T)
                out[f"{name}_sub"] = gw(cdist(ubf, ubf), d["d2"], ubf @ d["vb"].T)
                out[f"{name}_energy"] = float(cdist(ubf, d["vb"]).mean())
            return out, foscttm(flip(ut, neg) @ vt.T)           # test FOSCTTM: reporting only, after the choice

        # one BLAS thread per candidate thread, so the pool does not oversubscribe the cores; map keeps candidate order
        with threadpool_limits(1, user_api="blas"), ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
            res = list(pool.map(criteria, range(len(cands))))
    else:
        from torch_linear_assignment import batch_linear_assignment

        dev = torch.device(device)
        t = lambda v: torch.as_tensor(v, dtype=torch.float64, device=dev)
        cdist = lambda x, y: torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")
        bt, ut_t, vt_t = t(b), t(ut), t(vt)
        utb, bvt, simt = ut_t @ bt, bt.T @ vt_t.T, ut_t @ vt_t.T
        sgn_all = t(np.array(cands))
        sets = {}
        for name, (u, v) in {"ins": (us, vs), "ho": (uh, vh)}.items():
            u, v = t(u), t(v)
            ub, vb = u @ bt, v @ bt
            sets[name] = {"ub": ub, "vb": vb, "bv": bt.T @ v.T, "sim": u @ v.T, "c1": 1 - u @ u.T, "c2": 1 - v @ v.T,
                          "d2": cdist(vb, vb)}

        def gw(c1, c2, score):                              # score is consumed: negated in place for the solver
            c = batch_linear_assignment(score.neg_()).long()
            g = c2[c[:, :, None], c[:, None, :]]
            return g.sub_(c1).pow_(2).mean((1, 2))

        cols, fos = {}, []
        for i in range(0, len(cands), chunk):
            sg = sgn_all[i:i + chunk]
            neg = (sg < 0).to(torch.float64)                # 1 where the direction is flipped
            for name, d in sets.items():
                ubf = d["ub"] * sg[:, None, :]
                full = torch.baddbmm(d["sim"], d["ub"] * neg[:, None, :], d["bv"].expand(len(sg), -1, -1), alpha=-2)
                parts = {"full": gw(d["c1"], d["c2"], full),
                         "sub": gw(cdist(ubf, ubf), d["d2"], ubf @ d["vb"].T),
                         "energy": cdist(ubf, d["vb"].expand(len(sg), -1, -1)).mean((1, 2))}
                for key, val in parts.items():
                    cols.setdefault(f"{name}_{key}", []).append(val)
            for j in range(len(sg)):                        # test FOSCTTM as align.align.foscttm: ties 1/2, off-diagonal
                sim = torch.addmm(simt, utb * neg[j], bvt, alpha=-2)
                if not bool(torch.isfinite(sim).all()):
                    raise ValueError("non-finite similarity")
                dg = sim.diagonal()
                row = (sim > dg[:, None]).sum(1, dtype=sim.dtype) + 0.5 * (sim == dg[:, None]).sum(1, dtype=sim.dtype) - 0.5
                col = (sim > dg[None, :]).sum(0, dtype=sim.dtype) + 0.5 * (sim == dg[None, :]).sum(0, dtype=sim.dtype) - 0.5
                fos.append((row.mean() + col.mean()) / (2 * (len(dg) - 1)))
        cols = {key: torch.cat(v).cpu().numpy() for key, v in cols.items()}
        fos = torch.stack(fos).cpu().numpy()
        res = [({key: float(v[i]) for key, v in cols.items()}, float(fos[i])) for i in range(len(cands))]
    test_fos = [r[1] for r in res]
    o = int(np.argmin(test_fos))
    row = {"ho_k": k, "ho_m": m_ho, "ho_m_sel": len(xsel), "ho_oracle_best_foscttm": min(test_fos),
           "ho_oracle_worst_foscttm": max(test_fos)}
    for name in res[0][0]:
        vals = np.array([r[0][name] for r in res])
        best = int(np.argmin(vals))                         # first minimum: fixed tie order
        row.update({f"{name}_foscttm": test_fos[best], f"{name}_auc": auc(flip(ut, flips[best]), p),
                    f"{name}_n_flipped": int(flips[best].shape[1]),
                    f"{name}_rho": float(spearmanr(vals, test_fos).statistic),
                    f"{name}_best_rank": int((vals < vals[o]).sum())})
    return row


def _sign_greedy(aligner, xa, yb, xt, yt, p, auc, k, seed, n=1500):
    """Zero-pair orientation for many shared directions, where 2^k candidates are too many: the same subsample, hard
    coupling and GW loss as _sign_select, but over the top-k right singular directions of the mapped half-A images the
    signs are set by steepest descent: each step tries every single flip of the current pattern and keeps the one that
    lowers the GW loss most (first on ties); it stops when no single flip lowers it. No pair is used; test pairs only
    score the result."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy.optimize import linear_sum_assignment
    from threadpoolctl import threadpool_limits

    from align.align import foscttm

    rng = np.random.default_rng(seed)
    m = min(n, len(xa), len(yb))
    xs, ys = xa[rng.choice(len(xa), m, replace=False)], yb[rng.choice(len(yb), m, replace=False)]
    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    tx = lambda v: unit(aligner.transform(torch.as_tensor(v, dtype=torch.float32)).numpy().astype(np.float64))
    my = aligner.mean_y.double().numpy()[0]
    ty = lambda v: unit(np.asarray(v, np.float64) - my)
    us, vs, ut, vt = tx(xs), ty(ys), tx(xt), ty(yt)
    k = min(k, min(us.shape))
    c1, c2 = 1 - us @ us.T, 1 - vs @ vs.T
    b = np.linalg.svd(us - us.mean(0), full_matrices=False)[2][:k].T
    flip = lambda u, neg: unit(u - 2 * (u @ neg) @ neg.T)

    def gw_loss(mask):
        _, c = linear_sum_assignment(-(flip(us, b[:, mask]) @ vs.T))
        return float(np.mean((c1 - c2[np.ix_(c, c)]) ** 2))

    mask = np.zeros(k, bool)
    with threadpool_limits(1, user_api="blas"), ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        loss0 = loss = gw_loss(mask)
        steps = 0
        while True:
            trials = [mask ^ (np.arange(k) == j) for j in range(k)]
            tl = list(pool.map(gw_loss, trials))
            j = int(np.argmin(tl))
            if tl[j] >= loss:
                break
            mask, loss, steps = trials[j], tl[j], steps + 1
    chosen = flip(ut, b[:, mask])
    return {"greedy_k": k, "greedy_m": m, "greedy_foscttm": foscttm(chosen @ vt.T), "greedy_auc": auc(chosen, p),
            "greedy_n_flipped": int(mask.sum()), "greedy_steps": steps, "greedy_loss": loss, "greedy_loss0": loss0}


def _orient_diag(aligner, xa, yb, xt, yt, uy, seed, n=1500, draws=3):
    """Why a mirrored map is not repaired: does the GW loss prefer the right map at all? Three maps of the test images,
    scored by the GW loss of _sign_select (hard coupling, cosine distances) on draws independent subsamples of the
    train halves (mean and standard deviation over the draws) and by test FOSCTTM:
      found:  the aligner's map;
      sign:   the oracle sign fix (reflect the shared report directions whose test correlation is negative);
      rotate: the oracle orthogonal fix inside the shared subspace (Procrustes of the mapped test images onto their
              reports, both in the k shared report directions), which also undoes a rotation that no sign pattern
              can undo.
    The oracle fixes use test pairs; the losses use no pair. If the rotate map has a clearly lower loss than the
    found one, the objective tells right from wrong and the sign search is too narrow; if not, the geometry cannot."""
    from scipy.optimize import linear_sum_assignment

    from align.align import foscttm

    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    my = aligner.mean_y.double().numpy()[0]
    tx = lambda v: unit(aligner.transform(torch.as_tensor(v, dtype=torch.float32)).numpy().astype(np.float64))
    ty = lambda v: unit(np.asarray(v, np.float64) - my)
    ut, vt = tx(xt), ty(yt)
    q = np.linalg.qr(uy)[0]                                     # orthonormal basis of the shared report directions
    corr = np.array([np.corrcoef(ut @ q[:, j], vt @ q[:, j])[0, 1] for j in range(q.shape[1])])
    zs, zt = ut @ q, vt @ q
    w, _, vh = np.linalg.svd(zs.T @ zt)
    fixes = {"found": lambda u: u,
             "sign": lambda u: unit(u - 2 * (u @ q[:, corr < 0]) @ q[:, corr < 0].T),
             "rotate": lambda u: unit(u + ((u @ q) @ (w @ vh) - u @ q) @ q.T)}
    out = {"diag_k": int(q.shape[1])}
    for name, f in fixes.items():
        out[f"diag_fos_{name}"] = foscttm(f(ut) @ vt.T)
    for name in fixes:
        out[f"diag_loss_{name}"] = []
    for d in range(draws):
        rng = np.random.default_rng(1000 * seed + d)
        m = min(n, len(xa), len(yb))
        us, vs = tx(xa[rng.choice(len(xa), m, replace=False)]), ty(yb[rng.choice(len(yb), m, replace=False)])
        c2 = 1 - vs @ vs.T
        for name, f in fixes.items():
            u = f(us)
            _, c = linear_sum_assignment(-(u @ vs.T))
            out[f"diag_loss_{name}"].append(float(np.mean(((1 - u @ u.T) - c2[np.ix_(c, c)]) ** 2)))
    for name in fixes:
        v = out.pop(f"diag_loss_{name}")
        out[f"diag_loss_{name}"], out[f"diag_loss_{name}_sd"] = float(np.mean(v)), float(np.std(v))
    return out


def _rot_fix(aligner, xa, yb, xt, yt, p, auc, K, seed, boots=20, n_stab=8, n_rand=64):
    """Pair-free rotation of the top-k shared directions. The basis b is the top-k right singular vectors of the
    centered mapped half-A images, over all rows of xa (_sign_ho uses a subsample). The projections za = ua b and
    zb = vb b are centered and whitened per side (symmetric C^-1/2), so the two sides can differ only by an orthogonal R.
    R minimizes ||T3(wa R) - T3(wb)||^2 / noise3 + ||K4(wa R) - K4(wb)||^2 / noise4 over O(k). T3 is the third moment and
    K4 the exact fourth cumulant (sample covariance) of the whitened data; each noise is the mean squared bootstrap
    deviation of the tensors of both sides (`boots` resamples). Parametrization R = D expm(S - S^T), LBFGS from I, from
    an ICA init R0, the best of the 2^k sign diagonals D, and `n_rand` Haar orthogonals (rot_init_best names the winner:
    ica, id, sign or haar); the 8 best LBFGS results are then improved by Givens sweeps (rot_sweeps) and a final LBFGS. R0: FastICA (logcosh, no whitening) on wa and on wb separately gives orthogonal unmixings;
    components are signed to skew >= 0 and matched a to b by Hungarian on the |skew| + |excess kurtosis| differences
    (each scaled by its pooled sd), and R0 = Ua^T P Ub. rot_stab = median over `n_stab` bootstrap
    refits, each started from R*, of ||R_b - R*||_F / sqrt(2k). The map M = Wa R* Wb^-1 acts on the b-coordinates of the
    mapped images. rotref_*: the aligner refits its weight (num_iterations refinement steps) from weight @ Q with
    Q = I + b (M - I) b^T, for kk == K only. The map of the
    projections is ((u b) - ma) M + mb with ma, mb the means of za, zb; the refit ignores this mean shift (the weight
    acts linearly).
    K is the largest k: every kk in sorted({4, 6, K}) with kk <= K runs on the first kk columns of the same b, with keys
    rot<kk>_*, and rotref<K>_* for the refit.
    No pair is used, except the rot<kk>_oracle line (Procrustes of the test projections) and the scores on test pairs,
    which only report."""
    import copy
    import itertools

    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import FastICA

    from unpaired_rosetta.determinism import torch_threads
    from unpaired_rosetta.linalg import center_and_normalize
    from unpaired_rosetta.randomness import Randomness

    from align.align import foscttm

    f64 = lambda v: torch.as_tensor(v).double()
    unit = lambda v: torch.nn.functional.normalize(v, dim=-1)
    tx = lambda v, al=aligner: unit(al.transform(torch.as_tensor(v, dtype=torch.float32)).double())
    ty = lambda v: unit(f64(v) - aligner.mean_y.double()[0])
    ua, vb_, ut, vt = tx(xa), ty(yb), tx(xt), ty(yt)
    if not 0 < K <= min(ua.shape):
        raise ValueError(f"--rot {K} outside 1..{min(ua.shape)}")
    b_full = torch.linalg.svd(ua - ua.mean(0), full_matrices=False)[2][:K].T
    def one(k):
        b = b_full[:, :k]
        eye = torch.eye(k, dtype=torch.float64)

        def whiten(z):                                              # centered z -> C^-1/2 and C^1/2
            ev, V = torch.linalg.eigh(z.T @ z / len(z))
            return (V / ev.sqrt()) @ V.T, (V * ev.sqrt()) @ V.T

        def cumulants(w):                                           # rows -> third moment, exact fourth cumulant
            w = w - w.mean(0)
            n, c = len(w), w.T @ w / len(w)
            pw = (w[:, :, None] * w[:, None, :]).reshape(n, k * k)
            m4 = (pw.T @ pw / n).reshape(k, k, k, k)
            gauss = (c[:, :, None, None] * c[None, None, :, :] + c[:, None, :, None] * c[None, :, None, :]
                     + c[:, None, None, :] * c[None, :, :, None])
            return (pw.T @ w / n).reshape(k, k, k), m4 - gauss

        def rotated(t3, k4, R):
            R = R[None] if R.dim() == 2 else R                      # batch of rotations: leading axis z
            return (torch.einsum("abc,zai,zbj,zck->zijk", t3, R, R, R),
                    torch.einsum("abcd,zai,zbj,zck,zdl->zijkl", k4, R, R, R, R))

        def prepare(za, zb):                                        # whiten both sides, take the cumulants
            za, zb = za - za.mean(0), zb - zb.mean(0)
            Wa, Wb = whiten(za), whiten(zb)
            wa, wb = za @ Wa[0], zb @ Wb[0]
            return wa, wb, cumulants(wa), cumulants(wb), Wa[0], Wb[1]

        za, zb = ua @ b, vb_ @ b
        ma, mb = za.mean(0), zb.mean(0)
        wa, wb, Ta, Tb, Wa, Cb_half = prepare(za, zb)
        rng = np.random.default_rng([seed, 3])
        noise = np.zeros(2)
        for _ in range(boots):
            for w, T in ((wa, Ta), (wb, Tb)):
                tb = cumulants(w[rng.integers(len(w), size=len(w))])
                noise += [float(((tb[i] - T[i]) ** 2).sum()) for i in (0, 1)]
        noise /= boots

        def make_loss(Ta_, Tb_):
            def loss(R):
                v = sum(((x - y) ** 2).flatten(1).sum(1) / nz for x, y, nz in zip(rotated(*Ta_, R), Tb_, noise))
                return v[0] if R.dim() == 2 else v
            return loss

        def optimize(loss, D):
            S = torch.zeros(k, k, dtype=torch.float64, requires_grad=True)
            rot = lambda: D @ torch.linalg.matrix_exp(S - S.T)
            opt = torch.optim.LBFGS([S], max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                v = loss(rot())
                v.backward()
                return v
            opt.step(closure)
            with torch.no_grad():
                R = rot()
                return R, float(loss(R))

        def sweeps(loss, R, n_angle=72, max_sweep=20):
            """Givens sweeps: per pair (i < j) the best of n_angle angles over [0, pi) (angle 0 keeps R)."""
            th = torch.arange(n_angle, dtype=torch.float64) * np.pi / n_angle
            c, sn = th.cos()[:, None], th.sin()[:, None]
            cur, n_sweep = float(loss(R)), 0
            with torch.no_grad():
                while n_sweep < max_sweep:
                    before, n_sweep = cur, n_sweep + 1
                    for i in range(k):
                        for j in range(i + 1, k):
                            cand = R[None].repeat(n_angle, 1, 1)
                            cand[:, :, i] = c * R[:, i] + sn * R[:, j]
                            cand[:, :, j] = -sn * R[:, i] + c * R[:, j]
                            v = loss(cand)
                            m = int(v.argmin())
                            if float(v[m]) < cur:
                                R, cur = cand[m], float(v[m])
                    if before - cur <= 1e-9:
                        break
            return R, n_sweep

        loss = make_loss(Ta, Tb)
        with torch.no_grad():
            diags = [torch.diag(torch.tensor(s, dtype=torch.float64)) for s in itertools.product((1.0, -1.0), repeat=k)]
            d_best = diags[int(np.argmin([float(loss(d)) for d in diags]))]
            loss_id = float(loss(eye))
        gen = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, 4]).generate_state(1)[0]))
        haar = []
        for _ in range(n_rand):                                     # Haar on O(k): QR of a Gaussian, signs of diag(r) fixed
            q, r = torch.linalg.qr(torch.randn(k, k, generator=gen, dtype=torch.float64))
            haar.append(q * torch.sign(torch.diagonal(r)))

        def ica(w):                                                 # orthogonal unmixing, skew >= 0 per component
            w_np = w.numpy()
            U = FastICA(n_components=k, whiten=False, fun="logcosh", max_iter=1000,
                        random_state=int(np.random.SeedSequence([seed, 6]).generate_state(1)[0])).fit(w_np).components_
            u, _, vh = np.linalg.svd(U)
            U = torch.as_tensor(u @ vh)                             # polar factor
            src = w @ U.T
            sk = (src ** 3).mean(0) / (src ** 2).mean(0) ** 1.5
            U = U * torch.where(sk < 0, -1.0, 1.0).double()[:, None]
            src = w @ U.T
            return U, torch.abs(sk), (src ** 4).mean(0) / (src ** 2).mean(0) ** 2 - 3

        (Ua, ska, kua), (Ub, skb, kub) = ica(wa), ica(wb)
        sc = lambda x, y: torch.cat([x, y]).std()
        cost = (ska[:, None] - skb[None]).abs() / sc(ska, skb) + (kua[:, None] - kub[None]).abs() / sc(kua, kub)
        _, match = linear_sum_assignment(cost.numpy())
        perm = torch.zeros(k, k, dtype=torch.float64)
        perm[torch.arange(k), torch.as_tensor(match)] = 1.0
        inits = [("ica", Ua.T @ perm @ Ub), ("id", eye), ("sign", d_best)] + [("haar", q) for q in haar]
        fits = [optimize(loss, D) + (name, 0) for name, D in inits]
        for R, _, name, _ in sorted(fits, key=lambda t: t[1])[:8]:      # sweep + LBFGS from the 8 best inits
            R, n_sweep = sweeps(loss, R)
            fits.append(optimize(loss, R) + (name, n_sweep))
        R_star, loss_star, init_best, n_sweep = min(fits, key=lambda t: t[1])

        rb, stab = np.random.default_rng([seed, 5]), []
        for _ in range(n_stab):
            resample = lambda z: z[rb.integers(len(z), size=len(z))]
            _, _, Ta_b, Tb_b, _, _ = prepare(resample(ua @ b), resample(vb_ @ b))
            stab.append(float(torch.linalg.norm(optimize(make_loss(Ta_b, Tb_b), R_star)[0] - R_star) / np.sqrt(2 * k)))

        M = Wa @ R_star @ Cb_half                                   # Wb^-1 = C_b^1/2
        fmap = lambda u, A, ca=0.0, cb=0.0: unit(u + (((u @ b) - ca) @ A + cb - u @ b) @ b.T)
        sim = lambda u: (u @ vt.T).numpy()
        ch = fmap(ut, M, ma, mb)
        zs, zt = ut @ b, vt @ b
        w_, _, vh = torch.linalg.svd(zs.T @ zt)
        out = {f"rot{k}_loss_id": loss_id, f"rot{k}_loss": loss_star, f"rot{k}_n_init": len(inits),
               f"rot{k}_init_best": init_best, f"rot{k}_sweeps": n_sweep, f"rot{k}_stab": float(np.median(stab)),
               f"rot{k}_foscttm": foscttm(sim(ch)), f"rot{k}_auc": auc(ch.numpy(), p),
               f"rot{k}_oracle_foscttm": foscttm(sim(fmap(ut, w_ @ vh)))}

        if k == K:                                                  # the refit runs for the full k only
            Q = torch.eye(b.shape[0], dtype=torch.float64) + b @ (M - eye) @ b.T
            out[f"rotref{k}_foscttm"], out[f"rotref{k}_auc"] = refit(aligner.weight.double() @ Q)
        return out

    sx = center_and_normalize(torch.as_tensor(xa).float(), aligner.mean_x)
    sy = center_and_normalize(torch.as_tensor(yb).float(), aligner.mean_y)

    def refit(w0):
        ra = copy.copy(aligner)
        weight, rnd = w0.to(aligner.weight.dtype), Randomness(seed)
        with torch_threads(), torch.inference_mode():
            for _ in range(aligner.num_iterations):
                weight = ra.refinement_step(sx, sy, weight, None, None, rnd)
        ra.weight = weight
        u = tx(xt, ra)
        return foscttm((u @ vt.T).numpy()), auc(u.numpy(), p)

    out = {"rot_k": K}
    for kk in sorted({4, 6, K}):
        if kk <= K:
            out.update(one(kk))
    return out


def _sign_skew(aligner, xa, yb, xt, yt, p, auc, k):
    """Zero-pair orientation in closed form, from the reason a mirror is detectable at all: along a right direction the
    mapped images and the reports have third moments of the same sign; along a mirrored one, of opposite sign. Over the
    top-k right singular directions of the mapped half-A images, flip each direction whose skewness of mapped half-A
    images and of half-B reports (both projected on it) disagree in sign. Uses every half-A and half-B row, no pair;
    test pairs only score the result."""
    from scipy.stats import skew

    from align.align import foscttm

    unit = lambda v: v / np.linalg.norm(v, axis=1, keepdims=True)
    tx = lambda v: unit(aligner.transform(torch.as_tensor(v, dtype=torch.float32)).numpy().astype(np.float64))
    my = aligner.mean_y.double().numpy()[0]
    ty = lambda v: unit(np.asarray(v, np.float64) - my)
    ua, vb, ut, vt = tx(xa), ty(yb), tx(xt), ty(yt)
    k = min(k, min(ua.shape))
    b = np.linalg.svd(ua - ua.mean(0), full_matrices=False)[2][:k].T
    si, sr = skew(ua @ b, axis=0), skew(vb @ b, axis=0)
    sig = (np.abs(si) > 3 * np.sqrt(6 / len(ua))) & (np.abs(sr) > 3 * np.sqrt(6 / len(vb)))   # 3 SE of a Gaussian skew
    neg = b[:, sig & (np.sign(si) != np.sign(sr))]           # a sign is read only where both sides are clearly skewed
    chosen = unit(ut - 2 * (ut @ neg) @ neg.T)
    return {"skew_k": k, "skew_foscttm": foscttm(chosen @ vt.T), "skew_auc": auc(chosen, p),
            "skew_n_flipped": int(neg.shape[1]), "skew_n_significant": int(sig.sum()), "skew_img": si.round(4).tolist(), "skew_rep": sr.round(4).tolist()}


def _prep(spec, xr, yr, pr, train):
    """Pair-free locate step, fitted on each modality's own train rows (never on pairs):
    whiten:D  project on the top-D principal components and divide each by its standard deviation, so every direction,
              shared or private, holds the same variance share;
    ica:D     the same, followed by FastICA (D independent components, per modality);
    skewica:K:D  ica:D, then keep the K components with the largest |skewness| on the train rows (shared directions are
              skewed, so skew is the one marginal cue a modality alone carries about them)."""
    from scipy.stats import skew
    from sklearn.decomposition import PCA, FastICA

    kind, *nums = spec.split(":")
    d = int(nums[-1])
    out = []
    for z, extra in ((xr, None), (yr, pr)):
        z = z.astype(np.float64)
        f = PCA(n_components=d, whiten=True, random_state=0).fit(z[train])
        if kind in ("ica", "skewica"):
            ica = FastICA(n_components=d, whiten="unit-variance", random_state=0, max_iter=1000).fit(f.transform(z[train]))
            t = lambda v, f=f, ica=ica: ica.transform(f.transform(v))
            if kind == "skewica":
                keep = np.argsort(-np.abs(skew(t(z[train]), axis=0)), kind="stable")[: int(nums[0])]
                t = lambda v, t=t, keep=keep: t(v)[:, keep]
        elif kind == "whiten":
            t = f.transform
        else:
            raise ValueError(f"--prep {spec}: kind must be whiten, ica or skewica")
        if extra is None:
            e = None
        elif len(extra) == 0:                       # label-free domains have no prompts
            e = np.zeros((0, int(nums[0]) if kind == "skewica" else d))
        else:
            e = t(extra.astype(np.float64))
        out.append((t(z), e))
    (xw, _), (yw, pw) = out
    return xw, yw, pw


def _scale_shared(cases, train, xr, yr, pr, factor, random=False, symmetrize=False):
    """Oracle intervention as align.budget amplify: scale each side along its shared canonical directions (estimated on
    the sel patients) by factor (< 1 attenuates), before the unit normalization. Reports the shared-variance fractions
    and, as the validity check, the canonical correlations of the shared directions on train before and after."""
    from align.align import normalize, svd
    from align.budget import amplify_map, shared_directions, shared_fraction
    from align.spiked import qr_centred

    sel = np.flatnonzero(cases.split == "sel")
    xr, yr, pr = (v.astype(np.float64) for v in (xr, yr, pr))
    ux, uy, k = shared_directions(xr[sel], yr[sel], np.random.default_rng(0))
    shared_r = lambda xn, yn: svd(qr_centred(xn[train] @ ux)[0].T @ qr_centred(yn[train] @ uy)[0], compute_uv=False)
    r0 = shared_r(normalize(xr), normalize(yr))
    sx, sy = ux, uy
    if random:      # control: k random directions per side with the same variance share as the shared ones
        sx, sy = (_matched_random(normalize(v)[train], u, k) for v, u in ((xr, ux), (yr, uy)))
    gx, gy = amplify_map(sx, xr[sel].mean(0), factor), amplify_map(sy, yr[sel].mean(0), factor)
    xr, yr, pr = gx(xr), gy(yr), gy(pr)
    if symmetrize:      # skew intervention: one random sign per item reflects its image AND its report through their
        #                 train means. Covariances and cross-covariances keep their expectation, every odd moment
        #                 (skew) goes to zero in every direction, so a sign can no longer be read from the geometry.
        s = np.random.default_rng(7).choice([-1.0, 1.0], len(xr))[:, None]
        mx, my = xr[train].mean(0), yr[train].mean(0)
        xr, yr = mx + s * (xr - mx), my + s * (yr - my)
    r1 = shared_r(normalize(xr), normalize(yr))
    info = {"scale": factor, "k": k, "random": bool(random),
            "f_image_scaled": shared_fraction(normalize(xr)[train], sx), "f_report_scaled": shared_fraction(normalize(yr)[train], sy), "f_image": shared_fraction(normalize(xr)[train], ux),
            "f_report": shared_fraction(normalize(yr)[train], uy), "r1_before": float(r0[0]), "r1_after": float(r1[0]),
            "max_abs_dr": float(np.abs(r1 - r0).max())}
    print(json.dumps(info), flush=True)
    return xr, yr, pr, info, uy


def _matched_random(z, u, k, draws=500, top=64):
    """k orthonormal directions drawn at random inside the top principal subspace of z, the draw whose variance share
    is closest to that of u (the equal-energy control of the shared directions)."""
    from align.align import svd
    from align.budget import shared_fraction

    v = svd(z - z.mean(0), full_matrices=False)[2][:top].T
    target, rng = shared_fraction(z, u), np.random.default_rng(9000)
    cands = [v @ np.linalg.qr(rng.standard_normal((top, k)))[0] for _ in range(draws)]
    return min(cands, key=lambda c: abs(shared_fraction(z, c) - target))


def _save(a, rows, tag):
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{tag}.csv.tmp", index=False)   # atomic: a killed write leaves no torn CSV
    os.replace(f"{a.out}/{tag}.csv.tmp", f"{a.out}/{tag}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["hf", "ours"])
    ap.add_argument("--vision")
    ap.add_argument("--language")
    ap.add_argument("--emb")
    ap.add_argument("--image")
    ap.add_argument("--text")
    ap.add_argument("--train-size", type=int)
    ap.add_argument("--sign-select", type=int, default=0, metavar="K",
                    help="ours: zero-pair orientation over the map's top-K directions by GW loss (2^K candidates)")
    ap.add_argument("--sign-ho", type=int, default=0, metavar="K",
                    help="ours: score the 2^K sign patterns of --sign-select by six pair-free criteria "
                         "(in-sample / held-out sel x full GW / subspace GW / energy distance)")
    ap.add_argument("--rot", type=int, default=0, metavar="K",
                    help="ours --sign-ho K: also fit a rotation of the top-K shared directions by matching the 3rd and "
                         "4th cumulants of whitened image and report projections (needs the same K as --sign-ho)")
    ap.add_argument("--ho-device", choices=["cpu", "cuda"], default="cpu",
                    help="ours --sign-ho: cuda = float64 torch + GPU Hungarian (torch_linear_assignment) instead of "
                         "scipy on the CPU")
    ap.add_argument("--seed-set", choices=["dev", "confirm", "confirm2"], default="dev",
                    help="ours: dev = SEEDS[:3]; confirm = derive_seeds(4242, 3); confirm2 = derive_seeds(4343, 3)")
    ap.add_argument("--seed-index", type=int, choices=[0, 1, 2], default=None,
                    help="ours: run only this seed of the seed set; output tag gets __seed<i> (merge: "
                         "scripts/merge_seed_csvs.py)")
    ap.add_argument("--sign-greedy", type=int, default=0, metavar="K",
                    help="ours: zero-pair orientation by steepest single sign flips of the top-K directions (GW loss)")
    ap.add_argument("--sign-skew", type=int, default=0, metavar="K",
                    help="ours: zero-pair orientation in closed form: per top-K direction, match the skew signs")
    ap.add_argument("--sign-fix", action="store_true", help="ours --scale: also report the AUC after an oracle sign fix")
    ap.add_argument("--orient-diag", action="store_true",
                    help="ours --scale --sign-fix: GW loss of the found, oracle sign-fixed and oracle rotated maps")
    ap.add_argument("--prep", help="ours: pair-free locate step before the aligner: whiten:D, ica:D or skewica:K:D")
    ap.add_argument("--symmetrize", action="store_true", help="ours --scale: make the shared directions sign-symmetric")
    ap.add_argument("--random", action="store_true", help="ours --scale: scale k random directions instead (control)")
    ap.add_argument("--scale", type=float, help="ours: scale both sides along their shared directions (oracle, < 1 attenuates)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"hf": run_hf, "ours": run_ours}[a.step](a)


if __name__ == "__main__":
    main()
