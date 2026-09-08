import pandas as pd
import ast
import os
from collections import defaultdict

input_file = "exps_test/visual_only/val_res_best.csv"

df = pd.read_csv(input_file)

def safe_list_parse(x):
    try:
        v = ast.literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else None
    except Exception:
        return None

def norm_video_id(x):
    """统一 video_id：A001.mp4 -> A001"""
    x = str(x).strip()
    return os.path.splitext(x)[0]

def gt_is_coughing_pig(labels):
    """帧级 0/1 序列：只要出现过 1，就认为这头猪是咳嗽猪（pig-level GT）"""
    return any(int(v) == 1 for v in labels)

def mean_score(scores):
    """用整段 score 的平均值作为 pig-level 置信度（mean pooling）"""
    if scores is None or len(scores) == 0:
        return 0.0
    return sum(float(s) for s in scores) / len(scores)

# ==============================
# 汇总到 video 级别
# ==============================
video_rows = defaultdict(list)
bad_rows = 0

for _, row in df.iterrows():
    video_id = norm_video_id(row["video_id"])

    labels = safe_list_parse(row["labels"])
    scores = safe_list_parse(row["score"])
    if labels is None or scores is None:
        bad_rows += 1
        continue

    pig_gt = gt_is_coughing_pig(labels)
    conf = mean_score(scores)

    video_rows[video_id].append({
        "pig_gt": pig_gt,
        "conf": conf,
    })

def eval_top1_clip(video_rows_dict, prefix=None, neg_thresh=0.0):
    """
    Top-1 clip accuracy（按视频）:
    - 若视频 GT 存在咳嗽猪：选 conf 最大的猪作为 top1_pred，
      判对条件：top1_pred 对应 pig_gt=True
    - 若视频 GT 不存在咳嗽猪：
      判对条件：top1_conf <= neg_thresh
    """
    vids = [vid for vid in video_rows_dict.keys()
            if (prefix is None or vid.startswith(prefix))]

    total = len(vids)
    correct = 0

    for vid in vids:
        pigs = video_rows_dict[vid]
        if len(pigs) == 0:
            continue

        gt_has = any(p["pig_gt"] for p in pigs)

        top1 = max(pigs, key=lambda x: x["conf"])
        top1_conf = top1["conf"]
        top1_is_gt_cough = top1["pig_gt"]

        if gt_has:
            ok = (top1_is_gt_cough is True)
        else:
            ok = (top1_conf <= neg_thresh)

        if ok:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return total, correct, acc

# ✅ A=白天, N=夜晚
all_total, all_correct, all_acc = eval_top1_clip(video_rows, prefix=None)
day_total, day_correct, day_acc = eval_top1_clip(video_rows, prefix="A")
night_total, night_correct, night_acc = eval_top1_clip(video_rows, prefix="N")

print(f"Bad rows skipped: {bad_rows}")
print("\n===== VIDEO-LEVEL (TOP-1 PIG BY MEAN SCORE) =====")
print(f"ALL   : {all_correct}/{all_total}  Accuracy = {all_acc:.4f}")
print(f"DAY   : {day_correct}/{day_total}  Accuracy = {day_acc:.4f}  (A*)")
print(f"NIGHT : {night_correct}/{night_total}  Accuracy = {night_acc:.4f}  (N*)")
