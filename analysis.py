"""Reproducible diagnostics for ultrasonic flowmeters C and D.

Uses NumPy and the Python standard library only. Run from this directory:
    python analysis.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Flowmeters"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20261111
N_SPLITS = 5
N_REPEATS = 10
L2 = 0.1
CLASSES = np.array([1, 2, 3, 4])

FEATURE_NAMES = (
    ["profile_factor", "symmetry", "crossflow"]
    + [f"flow_velocity_{i}" for i in range(1, 5)]
    + [f"sound_speed_{i}" for i in range(1, 5)]
    + [f"signal_strength_{i}" for i in range(1, 9)]
    + [f"signal_quality_{i}" for i in range(1, 9)]
    + [f"gain_{i}" for i in range(1, 9)]
    + [f"transit_time_{i}" for i in range(1, 9)]
)
GROUPS = {
    "profile / flow shape": (0, 3),
    "flow velocity": (3, 7),
    "sound speed": (7, 11),
    "signal strength": (11, 19),
    "signal quality": (19, 27),
    "gain": (27, 35),
    "transit time": (35, 43),
}
COLORS = {1: "#2878b5", 2: "#e07a1f", 3: "#2c9a64", 4: "#bd4b55"}
CLASS_NAMES = {1: "Healthy", 2: "Gas", 3: "Installation", 4: "Wax"}


def load_meter(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / f"Meter {name}"
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] != 44:
        raise ValueError(f"{path.name}: expected 44 columns (43 features + class), got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: missing or non-finite values found")
    x, y = data[:, :43], data[:, 43].astype(int)
    if set(np.unique(y)) != set(CLASSES):
        raise ValueError(f"{path.name}: expected classes 1–4, found {sorted(set(y))}")
    return x, y


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(x: np.ndarray, y: np.ndarray, l2: float = L2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L2 multinomial logistic regression, optimized by backtracking gradient descent."""
    z, mean, scale = standardize_fit(x)
    n, d = z.shape
    design = np.column_stack([np.ones(n), z])
    target = (y[:, None] == CLASSES[None, :]).astype(float)
    weights = np.zeros((d + 1, len(CLASSES)), dtype=float)
    penalty = np.ones(d + 1)
    penalty[0] = 0.0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        probs = softmax(design @ w)
        loss = -np.sum(target * np.log(np.clip(probs, 1e-15, 1.0))) / n
        loss += 0.5 * l2 * np.sum((w * penalty[:, None]) ** 2)
        grad = design.T @ (probs - target) / n + l2 * w * penalty[:, None]
        return float(loss), grad

    loss, grad = objective(weights)
    step = 1.0
    for _ in range(2500):
        grad_norm = float(np.sum(grad * grad))
        if grad_norm < 1e-12:
            break
        while step > 1e-10:
            candidate = weights - step * grad
            candidate_loss, candidate_grad = objective(candidate)
            if candidate_loss <= loss - 1e-4 * step * grad_norm:
                weights, loss, grad = candidate, candidate_loss, candidate_grad
                step = min(step * 1.05, 10.0)
                break
            step *= 0.5
        else:
            break
    return weights, mean, scale


def predict(x: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, scale = fitted
    z = (x - mean) / scale
    return CLASSES[np.argmax(softmax(np.column_stack([np.ones(len(z)), z]) @ weights), axis=1)]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    recalls, f1s = [], []
    for cls in CLASSES:
        tp = int(np.sum((y == cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        recalls.append(tp / max(tp + fn, 1))
        precision = tp / max(tp + fp, 1)
        f1s.append(2 * precision * recalls[-1] / max(precision + recalls[-1], 1e-15))
    return {
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def stratified_fold_ids(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(y), dtype=int)
    for cls in CLASSES:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for fold, block in enumerate(np.array_split(idx, N_SPLITS)):
            fold_ids[block] = fold
    return fold_ids


def confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, guessed in zip(y, pred):
        matrix[int(actual) - 1, int(guessed) - 1] += 1
    return matrix


def evaluate_cv(name: str, x: np.ndarray, y: np.ndarray, seed_offset: int) -> tuple[list[dict], dict[str, float], np.ndarray]:
    fold_results, importance = [], {group: [] for group in GROUPS}
    cv_confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for repeat in range(N_REPEATS):
        fold_ids = stratified_fold_ids(y, SEED + seed_offset + repeat)
        for fold in range(N_SPLITS):
            test = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            fitted = fit_softmax(x[train], y[train])
            pred = predict(x[test], fitted)
            cv_confusion += confusion(y[test], pred)
            score = metrics(y[test], pred)
            fold_results.append({
                "meter": name, "repeat": repeat + 1, "fold": fold + 1,
                **score,
            })
            base = score["balanced_accuracy"]
            rng = np.random.default_rng(SEED + 10000 * seed_offset + repeat * N_SPLITS + fold)
            for group, (start, end) in GROUPS.items():
                drops = []
                for _ in range(3):
                    permuted = x[test].copy()
                    permuted[:, start:end] = permuted[rng.permutation(len(test)), start:end]
                    drops.append(base - metrics(y[test], predict(permuted, fitted))["balanced_accuracy"])
                importance[group].append(float(np.mean(drops)))
    summary = {
        metric: float(np.mean([row[metric] for row in fold_results]))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    summary.update({
        metric + "_sd": float(np.std([row[metric] for row in fold_results], ddof=1))
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    })
    return fold_results, {g: float(np.mean(v)) for g, v in importance.items()}, cv_confusion


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_scatter(xc: np.ndarray, yc: np.ndarray, xd: np.ndarray, yd: np.ndarray) -> str:
    w, h, left, top, plotw, ploth = 760, 500, 78, 42, 625, 340
    x = np.concatenate([xc[:, 0], xd[:, 0]])
    y = np.concatenate([xc[:, 1], xd[:, 1]])
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    padx, pady = (xmax - xmin) * 0.07, (ymax - ymin) * 0.07
    xmin, xmax, ymin, ymax = xmin - padx, xmax + padx, ymin - pady, ymax + pady
    sx = lambda a: left + (float(a) - xmin) / (xmax - xmin) * plotw
    sy = lambda a: top + ploth - (float(a) - ymin) / (ymax - ymin) * ploth
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']
    out += [f'<text x="{w/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Profile factor and symmetry by state</text>']
    for k in range(6):
        xv = xmin + (xmax - xmin) * k / 5
        yv = ymin + (ymax - ymin) * k / 5
        px, py = sx(xv), sy(yv)
        out += [f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+ploth}" stroke="#e5e7eb"/>',
                f'<line x1="{left}" y1="{py:.1f}" x2="{left+plotw}" y2="{py:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{px:.1f}" y="{top+ploth+22}" text-anchor="middle" font-family="Arial" font-size="11">{xv:.3g}</text>',
                f'<text x="{left-9}" y="{py+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{yv:.3g}</text>']
    for meter, xx, yy, shape in [("C", xc, yc, "circle"), ("D", xd, yd, "rect")]:
        for cls in CLASSES:
            for i in np.flatnonzero(yy == cls):
                cx, cy = sx(xx[i, 0]), sy(xx[i, 1])
                if shape == "circle":
                    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{COLORS[int(cls)]}" fill-opacity=".72" stroke="white" stroke-width=".5"/>')
                else:
                    out.append(f'<rect x="{cx-3.5:.1f}" y="{cy-3.5:.1f}" width="7" height="7" fill="{COLORS[int(cls)]}" fill-opacity=".72" stroke="white" stroke-width=".5"/>')
    out += [f'<line x1="{left}" y1="{top+ploth}" x2="{left+plotw}" y2="{top+ploth}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ploth}" stroke="#333"/>',
            f'<text x="{left+plotw/2}" y="{top+ploth+48}" text-anchor="middle" font-family="Arial" font-size="13">Profile factor</text>',
            f'<text x="19" y="{top+ploth/2}" transform="rotate(-90 19 {top+ploth/2})" text-anchor="middle" font-family="Arial" font-size="13">Symmetry</text>']
    legend_y = 465
    for j, cls in enumerate(CLASSES):
        xx = 160 + j * 135
        out += [f'<circle cx="{xx}" cy="{legend_y}" r="5" fill="{COLORS[int(cls)]}"/>',
                f'<text x="{xx+9}" y="{legend_y+4}" font-family="Arial" font-size="12">{CLASS_NAMES[int(cls)]}</text>']
    out += ['<circle cx="615" cy="465" r="4" fill="#777"/><text x="624" y="469" font-family="Arial" font-size="12">C</text>',
            '<rect x="652" y="461" width="8" height="8" fill="#777"/><text x="665" y="469" font-family="Arial" font-size="12">D</text>', '</svg>']
    return "\n".join(out)


def svg_confusions(matrices: dict[str, np.ndarray]) -> str:
    w, h = 790, 420
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{w/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">Instrument transfer: row-normalized confusion matrices</text>']
    cell = 53
    for panel, (title, matrix) in enumerate(matrices.items()):
        x0, y0 = 80 + panel * 380, 92
        out.append(f'<text x="{x0+2*cell}" y="66" text-anchor="middle" font-family="Arial" font-size="15">{title}</text>')
        norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        for i in range(4):
            out.append(f'<text x="{x0-12}" y="{y0+i*cell+32}" text-anchor="end" font-family="Arial" font-size="12">{i+1}</text>')
            out.append(f'<text x="{x0+i*cell+26}" y="{y0-10}" text-anchor="middle" font-family="Arial" font-size="12">{i+1}</text>')
            for j in range(4):
                v = norm[i, j]
                red = int(255 - 80 * v)
                green = int(255 - 145 * v)
                blue = int(255 - 145 * v)
                out.append(f'<rect x="{x0+j*cell}" y="{y0+i*cell}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="white"/>')
                out.append(f'<text x="{x0+j*cell+cell/2}" y="{y0+i*cell+31}" text-anchor="middle" font-family="Arial" font-size="13">{v:.2f}</text>')
        out += [f'<text x="{x0+2*cell}" y="{y0+4*cell+26}" text-anchor="middle" font-family="Arial" font-size="12">Predicted class</text>',
                f'<text x="{x0-47}" y="{y0+2*cell}" transform="rotate(-90 {x0-47} {y0+2*cell})" text-anchor="middle" font-family="Arial" font-size="12">Actual class</text>']
    out.append('</svg>')
    return "\n".join(out)


def svg_importance(rows: list[dict]) -> str:
    w, h = 800, 450
    ordered = sorted(rows, key=lambda r: r["mean_balanced_accuracy_drop"])
    left, top, chartw, barh, gap = 190, 72, 500, 35, 12
    vmax = max(max(abs(r["mean_balanced_accuracy_drop"]) for r in ordered), 0.01)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{w/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Grouped permutation importance in repeated CV</text>',
           f'<text x="{left+chartw/2}" y="50" text-anchor="middle" font-family="Arial" font-size="12">Drop in balanced accuracy after group permutation</text>']
    mid = left + chartw / 2
    out.append(f'<line x1="{mid}" y1="{top-3}" x2="{mid}" y2="{top+len(ordered)*(barh+gap)}" stroke="#555"/>')
    for i, row in enumerate(ordered):
        val = row["mean_balanced_accuracy_drop"]
        width = abs(val) / vmax * chartw / 2
        y = top + i * (barh + gap)
        x = mid - width if val < 0 else mid
        color = "#c95555" if val < 0 else "#3c8e6e"
        out += [f'<text x="{left-9}" y="{y+22}" text-anchor="end" font-family="Arial" font-size="12">{row["group"]}</text>',
                f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{barh}" fill="{color}"/>',
                f'<text x="{(x-8 if val < 0 else x+width+8):.1f}" y="{y+22}" text-anchor="{"end" if val < 0 else "start"}" font-family="Arial" font-size="11">{val:+.3f}</text>']
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    xc, yc = load_meter("C")
    xd, yd = load_meter("D")
    fold_rows, cv_summary, importance, cv_confusions = [], {}, {}, {}
    for name, x, y, offset in (("C", xc, yc, 100), ("D", xd, yd, 200)):
        rows, group_scores, cm = evaluate_cv(name, x, y, offset)
        fold_rows.extend(rows)
        cv_confusions[name] = cm
        cv_summary[name] = {
            metric: float(np.mean([r[metric] for r in rows]))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        cv_summary[name].update({
            metric + "_sd": float(np.std([r[metric] for r in rows], ddof=1))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        })
        importance[name] = group_scores
    write_csv(RESULTS / "repeated_stratified_cv.csv", fold_rows)
    classwise_rows = []
    cv_confusion_rows = []
    for name, y in (("C", yc), ("D", yd)):
        cm = cv_confusions[name]
        for i, cls in enumerate(CLASSES):
            row = {"meter": name, "class": int(cls), "name": CLASS_NAMES[int(cls)],
                   "recall": float(cm[i, i] / max(cm[i].sum(), 1)),
                   "support_across_repeats": int(cm[i].sum())}
            classwise_rows.append(row)
            cv_confusion_rows.append({
                "meter": name, "actual_class": int(cls),
                **{f"predicted_{int(CLASSES[j])}": int(cm[i, j]) for j in range(4)},
                "row_normalized": json.dumps((cm[i] / max(cm[i].sum(), 1)).round(6).tolist()),
            })
    write_csv(RESULTS / "repeated_cv_class_recall.csv", classwise_rows)
    write_csv(RESULTS / "repeated_cv_confusion.csv", cv_confusion_rows)

    transfer = {}
    transfer_rows = []
    for source, xs, ys, target, xt, yt in (("C", xc, yc, "D", xd, yd), ("D", xd, yd, "C", xc, yc)):
        pred = predict(xt, fit_softmax(xs, ys))
        score = metrics(yt, pred)
        cm = confusion(yt, pred)
        transfer[f"{source}_to_{target}"] = {
            **score, "confusion_counts": cm.tolist(),
            "class_recall": {str(cls): float(cm[cls-1, cls-1] / max(cm[cls-1].sum(), 1)) for cls in CLASSES},
        }
        transfer_rows.append({"transfer": f"{source}->{target}", **score})
        write_csv(RESULTS / f"confusion_{source}_to_{target}.csv", [
            {"actual_class": int(CLASSES[i]), **{f"predicted_{int(CLASSES[j])}": int(cm[i, j]) for j in range(4)},
             "row_normalized": json.dumps((cm[i] / max(cm[i].sum(), 1)).round(6).tolist())}
            for i in range(4)
        ])
    write_csv(RESULTS / "cross_meter_metrics.csv", transfer_rows)

    imp_rows = []
    for group in GROUPS:
        vals = [importance[m][group] for m in ("C", "D")]
        imp_rows.append({
            "group": group, "C_mean_balanced_accuracy_drop": vals[0],
            "D_mean_balanced_accuracy_drop": vals[1],
            "mean_balanced_accuracy_drop": float(np.mean(vals)),
        })
    write_csv(RESULTS / "grouped_permutation_importance.csv", imp_rows)
    write_csv(RESULTS / "feature_dictionary.csv", [
        {"column_1_based": i + 1, "feature": feature, "group": next(g for g, (a, b) in GROUPS.items() if a <= i < b)}
        for i, feature in enumerate(FEATURE_NAMES)
    ])
    metadata = {
        "seed": SEED, "splits": N_SPLITS, "repeats": N_REPEATS, "l2": L2,
        "counts": {"C": len(yc), "D": len(yd)},
        "class_counts": {
            "C": {str(c): int(np.sum(yc == c)) for c in CLASSES},
            "D": {str(c): int(np.sum(yd == c)) for c in CLASSES},
        },
        "cv": cv_summary, "cross_meter": transfer,
        "grouped_permutation_importance": importance,
    }
    (RESULTS / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (FIGURES / "profile_factor_vs_symmetry.svg").write_text(svg_scatter(xc, yc, xd, yd), encoding="utf-8")
    (FIGURES / "cross_meter_confusion.svg").write_text(svg_confusions({
        "Train C → test D": np.array(transfer["C_to_D"]["confusion_counts"]),
        "Train D → test C": np.array(transfer["D_to_C"]["confusion_counts"]),
    }), encoding="utf-8")
    (FIGURES / "grouped_importance.svg").write_text(svg_importance(imp_rows), encoding="utf-8")

    print("Repeated stratified CV (mean ± fold SD)")
    for name in ("C", "D"):
        s = cv_summary[name]
        print(f"Meter {name}: balanced accuracy {s['balanced_accuracy']:.3f} ± {s['balanced_accuracy_sd']:.3f}; "
              f"macro-F1 {s['macro_f1']:.3f} ± {s['macro_f1_sd']:.3f}; accuracy {s['accuracy']:.3f} ± {s['accuracy_sd']:.3f}")
    print("Cross-meter transfer")
    for key, s in transfer.items():
        print(f"{key}: balanced accuracy {s['balanced_accuracy']:.3f}; macro-F1 {s['macro_f1']:.3f}; accuracy {s['accuracy']:.3f}")
    print("Grouped importance (mean balanced-accuracy drop)")
    for row in sorted(imp_rows, key=lambda r: r["mean_balanced_accuracy_drop"], reverse=True):
        print(f"{row['group']}: {row['mean_balanced_accuracy_drop']:+.3f}")


if __name__ == "__main__":
    main()
