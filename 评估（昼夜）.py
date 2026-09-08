# TP正确识别的咳嗽猪只数量
# FP被误判为咳嗽但实际未咳嗽的猪只数量
# FN实际发生咳嗽但未被识别出的猪只数量
# TN正确识别为未咳嗽的猪只数量

import pandas as pd
import ast

input_file = "exps_test/visual_only5/val_res_best.csv"
output_file = "exps_test/visual_only5/fp_fn_cases.csv"  # 输出文件

THRESH = 0.5

MIN_CONSEC = 10

df = pd.read_csv(input_file)

def safe_list_parse(x, name=""):
    try:
        v = ast.literal_eval(x)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"{name} not list")
        return list(v)
    except Exception:
        return None

def pred_has_cough(scores, thresh=THRESH, min_consec=MIN_CONSEC):
    cnt = 0
    for s in scores:
        if float(s) > thresh:
            cnt += 1
            if cnt >= min_consec:
                return True
        else:
            cnt = 0
    return False

def update_counts(counts, pred, gt):
    if pred and gt:
        counts["TP"] += 1
    elif pred and (not gt):
        counts["FP"] += 1
    elif (not pred) and gt:
        counts["FN"] += 1
    else:
        counts["TN"] += 1

def compute_metrics(counts):
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    er        = (FP + FN) / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall, f1, er

# 三套计数器：总/白天(A)/夜晚(N)
counts_all   = {"TP":0, "FP":0, "FN":0, "TN":0}
counts_day   = {"TP":0, "FP":0, "FN":0, "TN":0}   # A 开头
counts_night = {"TP":0, "FP":0, "FN":0, "TN":0}   # N 开头

bad_rows = 0
skipped_prefix = 0

# ====== 新增：收集 FP / FN 行 ======
keep_cols = ["video_id","pig_id","cx","cy","w","h","angle","frame_count","fps","labels","score"]
error_rows = []  # 每个元素是 dict

for _, row in df.iterrows():
    video_id = str(row["video_id"])

    labels = safe_list_parse(row["labels"], "labels")
    scores = safe_list_parse(row["score"], "score")

    if labels is None or scores is None:
        bad_rows += 1
        continue

    gt = any(int(x) == 1 for x in labels)
    pred = pred_has_cough(scores)

    # 总体统计
    update_counts(counts_all, pred, gt)

    # 分场景统计：✅ 白天 A 开头；✅ 夜晚 N 开头
    if video_id.startswith("A"):
        update_counts(counts_day, pred, gt)
    elif video_id.startswith("N"):
        update_counts(counts_night, pred, gt)
    else:
        skipped_prefix += 1

    # ====== 保存 FP / FN 行 ======
    if pred and (not gt):
        d = {c: row[c] for c in keep_cols if c in row.index}
        d["error_type"] = "FP"
        error_rows.append(d)
    elif (not pred) and gt:
        d = {c: row[c] for c in keep_cols if c in row.index}
        d["error_type"] = "FN"
        error_rows.append(d)

def print_block(title, counts):
    TP, FP, FN, TN = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    precision, recall, f1, er = compute_metrics(counts)
    total = TP + FP + FN + TN
    print(f"\n===== {title} =====")
    print(f"Rows used: {total}")
    print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"Precision={precision:.4f}")
    print(f"Recall   ={recall:.4f}")
    print(f"F1       ={f1:.4f}")
    print(f"ER       ={er:.4f}")

print(f"Total rows in CSV: {len(df)} | Bad rows skipped: {bad_rows} | Non A/N skipped in split: {skipped_prefix}")
print_block("ALL (A+N+others)", counts_all)
print_block("DAY (video_id startswith 'A')", counts_day)
print_block("NIGHT (video_id startswith 'N')", counts_night)

# 输出 FP / FN 文件
err_df = pd.DataFrame(error_rows, columns=["error_type"] + keep_cols)
err_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ FP/FN cases saved to: {output_file}  (rows={len(err_df)})")
