import argparse
import logging
import time
import warnings
import numpy as np
import pandas as pd
import json
import re

warnings.filterwarnings("ignore")

def _parse_array(s):
    """将字符串解析为数值数组。兼容:
       - JSON 风格: "[0,1,0]" / "[0, 1, 0]"
       - 空格分隔:  "[0 1 0]"
       - 纯逗号分隔: "0,1,0"（无方括号）
       - 纯空格分隔: "0 1 0"（无方括号）
    返回 list[float] 或 list[int]。解析失败返回 []。
    """
    if isinstance(s, (list, tuple, np.ndarray)):
        return list(s)
    if not isinstance(s, str):
        return []
    s = s.strip()
    if s == "":
        return []
    # 去掉可能的方括号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if s == "":
        return []
    # 先尝试 JSON
    try:
        arr = json.loads("[" + s + "]") if ("," in s and "[" not in s) else json.loads(s if s.startswith("[") else "["+s+"]")
        if isinstance(arr, list):
            return [float(x) for x in arr]
    except Exception:
        pass
    # 退化：统一多个空白为单空格，再按逗号优先、否则空格切
    body = re.sub(r"\s+", " ", s)
    toks = body.split(",") if ("," in body) else body.split(" ")
    out = []
    for t in toks:
        t = t.strip()
        if t == "":
            continue
        try:
            out.append(float(t))
        except Exception:
            # 非法字符丢弃
            continue
    return out

def compute_average_precision(precision, recall):
    """VOC 风格 AP（对 precision 做单调回退）"""
    if precision is None:
        return np.nan
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)

    # 首尾补点
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # 单调回退
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    # 梯形积分
    idx = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[idx] - recall[idx - 1]) * precision[idx])
    return ap

def load_csv_groundtruth(filename):
    """期望列: video_id,pig_id,cx,cy,w,h,angle,frame_count,fps,labels
       解析 labels 为 0/1 数组，生成逐帧行，保留 cx..angle 列。
    """
    df = pd.read_csv(filename, engine="python")
    required = ["video_id", "pig_id", "cx", "cy", "w", "h", "angle", "frame_count", "fps", "labels"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Groundtruth 缺少列: {miss}; 需要列: {required}")

    recs = []
    for _, r in df.iterrows():
        vid = str(r["video_id"])
        pid = str(r["pig_id"])
        cx  = str(r["cx"])
        cy  = str(r["cy"])
        w   = str(r["w"])
        h   = str(r["h"])
        ang = str(r["angle"])
        try:
            fc = int(r["frame_count"])
        except Exception:
            # 若 frame_count 读不出，用 labels 长度兜底
            labs = _parse_array(str(r["labels"]))
            fc = len(labs)

        labs = _parse_array(str(r["labels"]))
        n = min(fc, len(labs))
        if n <= 0:
            continue
        if fc != len(labs):
            print(f"[WARN][GT] frame_count({fc}) != len(labels)({len(labs)}) @ {vid},{pid},{cx},{cy},{w},{h},{ang}; use n={n}")

        for i in range(n):
            recs.append((vid, pid, cx, cy, w, h, ang, i, int(labs[i])))

    out = pd.DataFrame(recs, columns=["video_id", "pig_id", "cx", "cy", "w", "h", "angle", "frame_idx", "gt_label"])
    return out


def load_csv_predictions(filename):
    """期望列: video_id,pig_id,cx,cy,w,h,angle,frame_count,fps,score
       解析 score 为概率数组，生成逐帧行，保留 cx..angle 列。
    """
    df = pd.read_csv(filename, engine="python")
    required = ["video_id", "pig_id", "cx", "cy", "w", "h", "angle", "frame_count", "fps", "score"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Predictions 缺少列: {miss}; 需要列: {required}")

    recs = []
    for _, r in df.iterrows():
        vid = str(r["video_id"])
        pid = str(r["pig_id"])
        cx  = str(r["cx"])
        cy  = str(r["cy"])
        w   = str(r["w"])
        h   = str(r["h"])
        ang = str(r["angle"])
        try:
            fc = int(r["frame_count"])
        except Exception:
            scs = _parse_array(str(r["score"]))
            fc = len(scs)

        scs = _parse_array(str(r["score"]))
        n = min(fc, len(scs))
        if n <= 0:
            continue
        if fc != len(scs):
            print(f"[WARN][PRED] frame_count({fc}) != len(score)({len(scs)}) @ {vid},{pid},{cx},{cy},{w},{h},{ang}; use n={n}")
        for i in range(n):
            recs.append((vid, pid, cx, cy, w, h, ang, i, float(scs[i])))

    out = pd.DataFrame(recs, columns=["video_id", "pig_id", "cx", "cy", "w", "h", "angle", "frame_idx", "pred_score"])
    return out


def merge_groundtruth_and_predictions(df_groundtruth, df_predictions):
    """
    严格 1:1（闭合集）合并：
    - 键集合完全一致（无多无少），不符合直接报错
    - 两侧键唯一（validate='1:1'）
    - 仅按分数降序排序
    """
    keys = ["video_id", "pig_id", "cx", "cy", "w", "h", "angle", "frame_idx"]

    # 1) 键唯一性检查（两侧都不能有重复键）
    for name, df in [("GT", df_groundtruth), ("Pred", df_predictions)]:
        dup = df.duplicated(subset=keys).sum()
        if dup > 0:
            raise ValueError(f"[{name}] 存在重复键 {keys}，重复行数={dup}")

    # 2) 键集合完全一致检查（无多无少）
    gt_keys = set(map(tuple, df_groundtruth[keys].to_numpy()))
    pr_keys = set(map(tuple, df_predictions[keys].to_numpy()))
    only_in_gt = gt_keys - pr_keys
    only_in_pr = pr_keys - gt_keys
    if only_in_gt or only_in_pr:
        msg = []
        if only_in_gt:
            # 展示少量示例，方便定位
            msg.append(f"预测缺失 {len(only_in_gt)} 行（示例：{list(only_in_gt)[:3]}）")
        if only_in_pr:
            msg.append(f"预测多余 {len(only_in_pr)} 行（示例：{list(only_in_pr)[:3]}）")
        raise ValueError("[STRICT] GT 与预测的键集合不一致；" + "；".join(msg))

    # 3) 合并：严格一对一
    df = df_groundtruth.merge(
        df_predictions,
        on=keys,
        how="inner",
        validate="1:1",
        suffixes=("", "_pred")  # gt_label 与 pred_score 列名保留
    )

    # 4) 分数完整性检查
    if "pred_score" not in df.columns:
        raise ValueError("[STRICT] 合并后缺少 pred_score 列")
    if df["pred_score"].isna().any():
        bad = df[df["pred_score"].isna()][keys].head(3).to_dict(orient="records")
        raise ValueError(f"[STRICT] 存在缺失的 pred_score 行（示例）：{bad}")

    # 5) 排序：仅按分数降序（稳定排序），不做“正样本优先”
    df = df.sort_values(by=["pred_score"], ascending=[False], kind="mergesort").reset_index(drop=True)
    return df


def calculate_precision_recall(df_merged, all_pos):
    """按排序计算逐点精确率与召回率；all_pos 必须来自完整 GT。"""
    if all_pos == 0:
        return np.array([1.0]), np.array([0.0])

    is_tp = (df_merged["gt_label"] == 1).astype(int).to_numpy()
    tp_cum = np.cumsum(is_tp)
    idx = np.arange(1, len(df_merged) + 1, dtype=float)

    precision = tp_cum / idx
    recall = tp_cum / float(all_pos)
    return precision, recall

def run_evaluation(groundtruth, predictions):
    """展开到逐帧后评估 AP。"""
    df_gt = load_csv_groundtruth(groundtruth)
    df_pr = load_csv_predictions(predictions)

    # 真实正样本总数来自完整 GT（不是合并后的）
    all_pos = int((df_gt["gt_label"] == 1).sum())

    # （可选）预检查键的唯一性，报更清晰的错
    dup_keys = ["video_id","pig_id","cx","cy","w","h","angle","frame_idx"]
    for name, df in [("GT", df_gt), ("Pred", df_pr)]:
        dup = df.duplicated(subset=dup_keys).sum()
        if dup > 0:
            raise ValueError(f"[{name}] 存在重复键 {dup_keys} 行数={dup}")

    # 合并，严格 1:1
    df_m  = merge_groundtruth_and_predictions(df_gt, df_pr)

    # 统计匹配情况，给用户反馈（可选）
    matched_pos = int(((df_m["gt_label"] == 1) & np.isfinite(df_m["pred_score"])).sum())
    missed_pos  = int((df_m["gt_label"] == 1).sum()) - matched_pos
    if missed_pos > 0:
        print(f"[INFO] 正样本总数(all_pos)={all_pos}；其中缺失预测的正样本帧={missed_pos}，已按 pred_score=-inf 计入排序。")

    # 计算 precision / recall
    precision, recall = calculate_precision_recall(df_m, all_pos)
    mAP = 100.0 * compute_average_precision(precision, recall)
    print("average precision: %2.2f%%" % (mAP))
    return mAP



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--groundtruth", type=str, required=True, help="Groundtruth CSV")
    parser.add_argument("-p", "--predictions", type=str, required=True, help="Predictions CSV")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()

def main():
    start = time.time()
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    mAP = run_evaluation(args.groundtruth, args.predictions)
    logging.info("Computed in %s seconds", time.time() - start)
    return mAP

if __name__ == "__main__":
    main()
