import pandas as pd
import ast

input_file = "exps_test/visual_only5/val_res_best.csv"

THRESH = 0.5

MIN_CONSEC = 10

df = pd.read_csv(input_file)

def safe_list_parse(x):
    try:
        v = ast.literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else None
    except Exception:
        return None

def pred_has_cough(scores, thresh, min_consec):
    cnt = 0
    for s in scores:
        if float(s) > thresh:
            cnt += 1
            if cnt >= min_consec:
                return True
        else:
            cnt = 0
    return False

# ==============================
# 三套 video-level 状态字典
# ==============================
video_all_correct   = {}   # 所有视频
video_day_correct   = {}   # A 开头（白天）
video_night_correct = {}   # N 开头（夜晚）

bad_rows = 0

for _, row in df.iterrows():
    video_id = str(row["video_id"])

    labels = safe_list_parse(row["labels"])
    scores = safe_list_parse(row["score"])
    if labels is None or scores is None:
        bad_rows += 1
        continue

    gt = any(int(x) == 1 for x in labels)
    pred = pred_has_cough(scores, THRESH, MIN_CONSEC)

    pig_correct = (pred == gt)

    # ---------- 总体 ----------
    if video_id not in video_all_correct:
        video_all_correct[video_id] = True
    if not pig_correct:
        video_all_correct[video_id] = False

    # ---------- 白天（A 开头） ----------
    if video_id.startswith("A"):
        if video_id not in video_day_correct:
            video_day_correct[video_id] = True
        if not pig_correct:
            video_day_correct[video_id] = False

    # ---------- 夜晚（N 开头） ----------
    elif video_id.startswith("N"):
        if video_id not in video_night_correct:
            video_night_correct[video_id] = True
        if not pig_correct:
            video_night_correct[video_id] = False

# ==============================
# 统计函数
# ==============================
def calc_video_acc(video_dict):
    total = len(video_dict)
    correct = sum(1 for v in video_dict.values() if v)
    acc = correct / total if total > 0 else 0.0
    return total, correct, acc

# ==============================
# 结果输出
# ==============================
all_total, all_correct, all_acc = calc_video_acc(video_all_correct)
day_total, day_correct, day_acc = calc_video_acc(video_day_correct)
night_total, night_correct, night_acc = calc_video_acc(video_night_correct)

print(f"Bad rows skipped: {bad_rows}")

print("\n===== VIDEO-LEVEL (ALL-CORRECT CRITERION) =====")
print(f"ALL   : {all_correct}/{all_total}  Accuracy = {all_acc:.4f}")
print(f"DAY   : {day_correct}/{day_total}  Accuracy = {day_acc:.4f}  (A*)")
print(f"NIGHT : {night_correct}/{night_total}  Accuracy = {night_acc:.4f}  (N*)")
