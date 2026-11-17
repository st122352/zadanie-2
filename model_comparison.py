"""Compare multiple classifiers and constant baselines for meters C and D.

Dependencies: NumPy and the local analysis.py module.
Run: python model_comparison.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analysis import CLASSES, SEED, load_meter, standardize_fit

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_SPLITS = 5
N_REPEATS = 5
RF_TREES = 25
RF_DEPTH = 6
CLASS_NAMES = {1: "Healthy", 2: "Gas", 3: "Installation", 4: "Wax"}
MODEL_LABELS = {
    "majority": "Majority baseline",
    "logistic": "Logistic",
    "knn5": "k-NN (k=5)",
    "gaussian_nb": "Gaussian NB",
    "random_forest": "Random forest",
    "always_no": "Always no fault",
    "always_yes": "Always fault",
}
COLORS = {
    "majority": "#9099a2", "logistic": "#2878b5", "knn5": "#d97827",
    "gaussian_nb": "#26936c", "random_forest": "#9163a8",
    "always_no": "#888f98", "always_yes": "#c84c50",
}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ex = np.exp(z)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_logistic(x: np.ndarray, y: np.ndarray, classes: np.ndarray, l2: float = 0.1) -> dict:
    z, mean, scale = standardize_fit(x)
    n, d = z.shape
    design = np.column_stack([np.ones(n), z])
    target = (y[:, None] == classes[None, :]).astype(float)
    weights = np.zeros((d + 1, len(classes)))
    penalty = np.ones(d + 1)
    penalty[0] = 0

    def objective(w):
        probs = softmax(design @ w)
        loss = -np.sum(target * np.log(np.clip(probs, 1e-15, 1))) / n
        loss += 0.5 * l2 * np.sum((w * penalty[:, None])**2)
        grad = design.T @ (probs - target) / n + l2 * w * penalty[:, None]
        return float(loss), grad

    loss, grad = objective(weights)
    step = 1.0
    for _ in range(600):
        norm = float(np.sum(grad**2))
        if norm < 1e-11:
            break
        for _ in range(30):
            candidate = weights - step * grad
            cand_loss, cand_grad = objective(candidate)
            if cand_loss <= loss - 1e-4 * step * norm:
                weights, loss, grad = candidate, cand_loss, cand_grad
                step = min(step * 1.05, 10.0)
                break
            step *= 0.5
    return {"weights": weights, "mean": mean, "scale": scale, "classes": classes}


def pred_logistic(x: np.ndarray, model: dict) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    scores = softmax(np.column_stack([np.ones(len(z)), z]) @ model["weights"])
    return model["classes"][np.argmax(scores, axis=1)]


def fit_knn(x: np.ndarray, y: np.ndarray) -> dict:
    z, mean, scale = standardize_fit(x)
    return {"x": z, "y": y.copy(), "mean": mean, "scale": scale}


def pred_knn(x: np.ndarray, model: dict, k: int = 5) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    result = []
    for row in z:
        dist = np.sum((model["x"] - row)**2, axis=1)
        nearest = np.argsort(dist, kind="stable")[:min(k, len(dist))]
        labels, counts = np.unique(model["y"][nearest], return_counts=True)
        winners = labels[counts == counts.max()]
        if len(winners) == 1:
            result.append(winners[0])
        else:
            result.append(next(v for v in model["y"][nearest] if v in winners))
    return np.asarray(result)


def fit_gaussian_nb(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    means, variances, priors = [], [], []
    for cls in classes:
        rows = x[y == cls]
        means.append(rows.mean(axis=0))
        variances.append(np.maximum(rows.var(axis=0), 1e-8))
        priors.append(len(rows) / len(y))
    return {"classes": classes, "means": np.asarray(means), "variances": np.asarray(variances),
            "priors": np.asarray(priors)}


def pred_gaussian_nb(x: np.ndarray, model: dict) -> np.ndarray:
    logp = []
    for mean, var, prior in zip(model["means"], model["variances"], model["priors"]):
        score = np.log(prior) - 0.5 * np.sum(np.log(2 * np.pi * var), axis=0)
        score -= 0.5 * np.sum((x - mean)**2 / var, axis=1)
        logp.append(score)
    return model["classes"][np.argmax(np.column_stack(logp), axis=1)]


def gini(counts: np.ndarray) -> float:
    total = counts.sum()
    return 0.0 if total == 0 else float(1 - np.sum((counts / total)**2))


def build_tree(x: np.ndarray, y: np.ndarray, classes: np.ndarray, rng: np.random.Generator,
               depth: int = 0, max_depth: int = RF_DEPTH, min_leaf: int = 2):
    counts = np.array([np.sum(y == cls) for cls in classes])
    majority = classes[int(np.argmax(counts))]
    if depth >= max_depth or len(y) < 2 * min_leaf or np.count_nonzero(counts) == 1:
        return {"leaf": int(majority)}
    feature_count = max(1, int(np.sqrt(x.shape[1])))
    features = rng.choice(x.shape[1], feature_count, replace=False)
    parent = gini(counts)
    best = (0.0, None, None)
    for feature in features:
        order = np.argsort(x[:, feature], kind="stable")
        vals = x[order, feature]
        labels = y[order]
        valid = np.flatnonzero((vals[:-1] < vals[1:]) & (labels[:-1] != labels[1:])) + 1
        if len(valid) == 0:
            continue
        encoded = np.searchsorted(classes, labels)
        cumulative = np.zeros((len(labels) + 1, len(classes)), dtype=int)
        for cls_idx, cls in enumerate(classes):
            cumulative[1:, cls_idx] = np.cumsum(encoded == cls_idx)
        left_counts = cumulative[valid]
        right_counts = counts[None, :] - left_counts
        left_n = valid.astype(float)
        right_n = len(y) - left_n
        score = (left_n * np.array([gini(c) for c in left_counts]) +
                 right_n * np.array([gini(c) for c in right_counts])) / len(y)
        improvement = parent - score
        allowed = (left_n >= min_leaf) & (right_n >= min_leaf)
        if not np.any(allowed):
            continue
        candidate_idx = int(np.argmax(np.where(allowed, improvement, -np.inf)))
        if improvement[candidate_idx] > best[0]:
            pos = int(valid[candidate_idx])
            threshold = float((vals[pos-1] + vals[pos]) / 2)
            best = (float(improvement[candidate_idx]), int(feature), threshold)
    if best[1] is None or best[0] <= 1e-10:
        return {"leaf": int(majority)}
    feature, threshold = best[1], best[2]
    left = x[:, feature] <= threshold
    if left.sum() < min_leaf or (~left).sum() < min_leaf:
        return {"leaf": int(majority)}
    return {"feature": feature, "threshold": threshold,
            "left": build_tree(x[left], y[left], classes, rng, depth+1, max_depth, min_leaf),
            "right": build_tree(x[~left], y[~left], classes, rng, depth+1, max_depth, min_leaf)}


def tree_predict(tree, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype=int)
    for i, row in enumerate(x):
        node = tree
        while "leaf" not in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        out[i] = node["leaf"]
    return out


def fit_forest(x: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int) -> list:
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(RF_TREES):
        sample = rng.integers(0, len(y), size=len(y))
        trees.append(build_tree(x[sample], y[sample], classes, rng))
    return trees


def pred_forest(x: np.ndarray, trees: list, classes: np.ndarray) -> np.ndarray:
    votes = np.zeros((len(x), len(classes)), dtype=int)
    for tree in trees:
        pred = tree_predict(tree, x)
        for j, cls in enumerate(classes):
            votes[:, j] += pred == cls
    return classes[np.argmax(votes, axis=1)]


def fit_model(name: str, x: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int):
    if name == "majority":
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])
    if name == "logistic":
        return fit_logistic(x, y, classes)
    if name == "knn5":
        return fit_knn(x, y)
    if name == "gaussian_nb":
        return fit_gaussian_nb(x, y, classes)
    if name == "random_forest":
        return fit_forest(x, y, classes, seed)
    raise ValueError(name)


def predict_model(name: str, x: np.ndarray, model, classes: np.ndarray) -> np.ndarray:
    if name == "majority":
        return np.full(len(x), model, dtype=int)
    if name == "logistic":
        return pred_logistic(x, model)
    if name == "knn5":
        return pred_knn(x, model, 5)
    if name == "gaussian_nb":
        return pred_gaussian_nb(x, model)
    if name == "random_forest":
        return pred_forest(x, model, classes)
    raise ValueError(name)


def stratified_folds(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    classes = np.unique(y)
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for k, block in enumerate(np.array_split(idx, N_SPLITS)):
            fold[block] = k
    return fold


def classification_metrics(y: np.ndarray, pred: np.ndarray, classes: np.ndarray) -> dict:
    recalls, f1s = [], []
    for cls in classes:
        tp = int(np.sum((y == cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        recall = tp / max(tp+fn, 1)
        precision = tp / max(tp+fp, 1)
        recalls.append(recall)
        f1s.append(2*precision*recall/max(precision+recall, 1e-15))
    return {"accuracy":float(np.mean(y==pred)), "balanced_accuracy":float(np.mean(recalls)),
            "macro_f1":float(np.mean(f1s)), **{f"recall_class_{int(cls)}":float(value) for cls,value in zip(classes,recalls)}}


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    recall = tp / max(tp+fn,1)
    specificity = tn / max(tn+fp,1)
    precision = tp / max(tp+fp,1)
    f1 = 2*precision*recall/max(precision+recall,1e-15)
    return {"accuracy":(tp+tn)/max(tp+tn+fp+fn,1),"balanced_accuracy":(recall+specificity)/2,
            "precision_fault":precision,"recall_fault":recall,"specificity_healthy":specificity,
            "f1_fault":f1,"tp":tp,"fn":fn,"fp":fp,"tn":tn}


def aggregate(rows: list[dict], task: str, meter: str) -> list[dict]:
    group = {}
    for row in rows:
        group.setdefault(row["model"], []).append(row)
    metrics = (["accuracy","balanced_accuracy","macro_f1","recall_class_1","recall_class_2","recall_class_3","recall_class_4"] if task=="multiclass" else
               ["accuracy","balanced_accuracy","precision_fault","recall_fault","specificity_healthy","f1_fault"])
    output=[]
    for model, values in group.items():
        row={"task":task,"meter":meter,"model":model,"model_label":MODEL_LABELS[model],
             "folds":len(values)}
        for metric in metrics:
            vals=np.array([v[metric] for v in values],dtype=float)
            row[metric]=float(vals.mean())
            row[metric+"_sd"]=float(vals.std(ddof=1))
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_bars(summary: list[dict], task: str, metric: str, filename: Path, title: str) -> None:
    rows=[r for r in summary if r["task"]==task]
    meters=["C","D"]
    models=list(dict.fromkeys(r["model"] for r in rows))
    width,height=940,500
    left,top,plotw,ploth=100,75,790,330
    vmax=max(1.0,max(r[metric] for r in rows)*1.12)
    slot=plotw/(len(models)*len(meters))
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>']
    for j in range(5):
        value=vmax*j/4
        y=top+ploth-value/vmax*ploth
        out += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plotw}" y2="{y:.1f}" stroke="#e6e6e6"/>',
                f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>']
    barw=slot*.65
    for i,model in enumerate(models):
        for j,meter in enumerate(meters):
            row=next((r for r in rows if r["model"]==model and r["meter"]==meter),None)
            if row is None: continue
            x=left+(i*2+j)*slot+(slot-barw)/2
            y=top+ploth-row[metric]/vmax*ploth
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{barw:.1f}" height="{top+ploth-y:.1f}" fill="{COLORS[model]}"/>')
            out.append(f'<text x="{x+barw/2:.1f}" y="{top+ploth+18}" text-anchor="middle" font-family="Arial" font-size="10">{meter}</text>')
        label_x=left+(i*2+1)*slot
        out.append(f'<text x="{label_x:.1f}" y="{top+ploth+40}" text-anchor="middle" font-family="Arial" font-size="10">{MODEL_LABELS[model]}</text>')
    out += [f'<line x1="{left}" y1="{top+ploth}" x2="{left+plotw}" y2="{top+ploth}" stroke="#333"/>',
            f'<text x="22" y="{top+ploth/2}" transform="rotate(-90 22 {top+ploth/2})" text-anchor="middle" font-family="Arial" font-size="12">{metric.replace("_"," ")}</text>']
    out.append('</svg>')
    filename.write_text("\n".join(out),encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data={name:load_meter(name) for name in ("C","D")}
    multiclass_models=["majority","logistic","knn5","gaussian_nb","random_forest"]
    binary_models=["always_no","always_yes","logistic","knn5","gaussian_nb","random_forest"]
    fold_rows=[]; summaries=[]; transfer_rows=[]
    for meter,(x,y) in data.items():
        cv_rows={"multiclass":[],"binary":[]}
        for repeat in range(N_REPEATS):
            fold=stratified_folds(y,SEED+1000*(0 if meter=="C" else 1)+repeat)
            for k in range(N_SPLITS):
                test=np.flatnonzero(fold==k); train=np.flatnonzero(fold!=k)
                for model_name in multiclass_models:
                    model=fit_model(model_name,x[train],y[train],CLASSES,SEED+repeat*20+k)
                    pred=predict_model(model_name,x[test],model,CLASSES)
                    row={"task":"multiclass","meter":meter,"repeat":repeat+1,"fold":k+1,
                         "model":model_name,**classification_metrics(y[test],pred,CLASSES)}
                    cv_rows["multiclass"].append(row);fold_rows.append(row)

                yb=(y!=1).astype(int)
                for model_name in binary_models:
                    if model_name=="always_no":
                        pred=np.zeros(len(test),dtype=int)
                    elif model_name=="always_yes":
                        pred=np.ones(len(test),dtype=int)
                    else:
                        model=fit_model(model_name,x[train],yb[train],np.array([0,1]),SEED+repeat*20+k+50)
                        pred=predict_model(model_name,x[test],model,np.array([0,1]))
                    row={"task":"binary_fault","meter":meter,"repeat":repeat+1,"fold":k+1,
                         "model":model_name,**binary_metrics(yb[test],pred)}
                    cv_rows["binary"].append(row);fold_rows.append(row)
        summaries.extend(aggregate(cv_rows["multiclass"],"multiclass",meter))
        summaries.extend(aggregate(cv_rows["binary"],"binary_fault",meter))

    # Cross-meter transfer for all learned classifiers and their constant baselines.
    for source,target in (("C","D"),("D","C")):
        xs,ys=data[source];xt,yt=data[target]
        for name in multiclass_models:
            model=fit_model(name,xs,ys,CLASSES,SEED+300)
            pred=predict_model(name,xt,model,CLASSES)
            transfer_rows.append({"task":"multiclass","transfer":source+"->"+target,"model":name,
                                  **classification_metrics(yt,pred,CLASSES)})
        ybs=(ys!=1).astype(int); ybt=(yt!=1).astype(int)
        for name in binary_models:
            if name=="always_no": pred=np.zeros(len(ybt),dtype=int)
            elif name=="always_yes": pred=np.ones(len(ybt),dtype=int)
            else:
                model=fit_model(name,xs,ybs,np.array([0,1]),SEED+400)
                pred=predict_model(name,xt,model,np.array([0,1]))
            transfer_rows.append({"task":"binary_fault","transfer":source+"->"+target,"model":name,
                                  **binary_metrics(ybt,pred)})
    write_csv(RESULTS/"model_comparison_folds.csv",fold_rows)
    write_csv(RESULTS/"model_comparison_summary.csv",summaries)
    write_csv(RESULTS/"model_transfer.csv",transfer_rows)
    svg_bars(summaries,"multiclass","balanced_accuracy",FIGURES/"multiclass_model_comparison.svg",
             "Four-class repeated stratified CV: balanced accuracy")
    svg_bars(summaries,"binary_fault","balanced_accuracy",FIGURES/"binary_model_comparison.svg",
             "Binary fault detection: balanced accuracy")
    print("CV summary (mean balanced accuracy, mean macro-F1 for multiclass / fault-F1 for binary):")
    for task in ("multiclass","binary_fault"):
        for meter in ("C","D"):
            rows=[r for r in summaries if r["task"]==task and r["meter"]==meter]
            print(task,meter)
            for r in rows:
                f1=r.get("macro_f1",r.get("f1_fault"))
                print(f"  {r['model']}: bacc={r['balanced_accuracy']:.3f}±{r['balanced_accuracy_sd']:.3f}; F1={f1:.3f}")
    print("Cross-meter transfer:")
    for row in transfer_rows:
        f1=row.get("macro_f1",row.get("f1_fault"))
        print(f"  {row['task']} {row['transfer']} {row['model']}: bacc={row['balanced_accuracy']:.3f}; F1={f1:.3f}")


if __name__=="__main__":
    main()
