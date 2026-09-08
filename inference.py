# -*- coding: utf-8 -*-
#代码python inferenc.py --video /root/data1/sed-crnn/pig_cough_data/video/A0001.mp4 --output_dir ./inference_out
"""
猪只咳嗽音视频联合推理管线 (含推理速度/参数量/FLOPs 统计)
流程: SED-CRNN咳嗽检测 -> YOLO11猪只检测 -> 逐猪裁剪视频序列
      -> SED-CRNN音频特征(T,32)对齐视频帧 -> TalkNet音视频融合推理
      -> 咳嗽猪只红框标注输出
用法:
  cd /root/data1/TalkNet-ASD-main
  python cough_inference_pipeline.py --video /path/to/video.mp4 --output_dir ./inference_out
"""
import os, sys, json, argparse, subprocess, time
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import numpy as np
import cv2
import librosa
import soundfile as sf
import joblib

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import load_model, Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

SED_MODEL     = "/root/data1/sed-crnn/models/bin_2025_12_25_10_14_52_fold_4_model.h5"
SED_SCALER    = "/root/data1/sed-crnn/pkl_pig/scaler_bin_fold4.pkl"
YOLO_WEIGHTS  = "/root/data1/YOLO11/runs/train/day_and_night/weights/best.pt"
TALKNET_MODEL = os.path.join(SCRIPT_DIR, "exps_new/2/model/model_best.model")

SR, NFFT, HOP_LEN, NB_MEL, SEQ_LEN, NUM_CH = 44100, 2048, 1024, 40, 256, 2
FRAME_DUR = HOP_LEN / SR
CLASS_LABELS = ['cough']

# ============================================================
# 性能统计工具
# ============================================================
class Timer:
    """轻量计时上下文, 记录阶段耗时(秒)"""
    def __init__(self, name, store=None):
        self.name, self.store, self.t0 = name, store, 0.0
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        if self.store is not None:
            self.store[self.name] = self.store.get(self.name, 0.0) + self.elapsed
        print("    [time] %-28s %.3f s" % (self.name, self.elapsed))

def _fmt_num(n):
    """格式化数值: 大数转 K/M/G"""
    for unit in ['', 'K', 'M', 'G']:
        if abs(n) < 1000: return "%.2f%s" % (n, unit)
        n /= 1000
    return "%.2f%s" % (n, 'T')

def get_model_complexity(net):
    """
    统计 TalkNet 的 参数量 与 FLOPs/MACs.
    优先 thop, 次选 ptflops, 都不可用则仅返回参数量并提示.
    """
    import torch
    info = {}
    # ---- 参数量(纯torch, 必有) ----
    total = sum(p.numel() for p in net.parameters())
    train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    info['params_total'] = total
    info['params_trainable'] = train
    info['params_total_fmt'] = _fmt_num(total)

    # ---- FLOPs/MACs ----
    info['flops'] = None
    info['flops_fmt'] = 'N/A (需安装 thop 或 ptflops)'
    info['flops_lib'] = None

    # 构造 1 条样本输入: audio(1, T, 32), visual(1, T, 112, 112)
    from talkNet import DEVICE
    T = 50
    audio = torch.randn(1, T, 32).to(DEVICE)
    visual = torch.randn(1, T, 112, 112).to(DEVICE)
    try:
        from thop import profile
        # thop 不能直接处理多输入分支模型, 这里分别统计 audio/visual 前端 + 融合后端
        # 用一个 wrapper 把整条管线包成一个 forward
        class _Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, a, v):
                ae = self.m.forward_audio_frontend(a)
                ve = self.m.forward_visual_frontend(v)
                out = self.m.forward_audio_visual_backend(ae, ve)
                return out
        w = _Wrapper(net.model).to(DEVICE).eval()
        macs, params = profile(w, inputs=(audio, visual), verbose=False)
        info['flops'] = float(macs * 2)  # FLOPs ≈ 2 × MACs
        info['flops_fmt'] = _fmt_num(info['flops'])
        info['flops_lib'] = 'thop'
    except Exception as e1:
        try:
            from ptflops import get_model_complexity_info
            class _Wrapper(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, a, v):
                    ae = self.m.forward_audio_frontend(a)
                    ve = self.m.forward_visual_frontend(v)
                    return self.m.forward_audio_visual_backend(ae, ve)
            w = _Wrapper(net.model).to(DEVICE).eval()
            macs, _ = get_model_complexity_info(
                w, input_res=(1, T, 32), as_strings=False,
                print_per_layer_stat=False, input_constructor=lambda x: x)
            info['flops'] = float(macs * 2)
            info['flops_fmt'] = _fmt_num(info['flops'])
            info['flops_lib'] = 'ptflops'
        except Exception as e2:
            info['flops_lib'] = f'unavailable (thop: {e1}; ptflops: {e2})'
    return info

def measure_speed(probs_dict, per_pig_times, n_frames):
    """根据逐猪推理耗时与帧数, 推算 TalkNet 推理速度"""
    total_t = sum(per_pig_times)
    fps = n_frames / total_t if total_t > 0 else 0.0
    avg_per_pig = total_t / len(per_pig_times) if per_pig_times else 0.0
    ms_per_frame = 1000.0 / fps if fps > 0 else 0.0
    return {'talknet_total_sec': round(total_t, 4),
            'talknet_fps': round(fps, 2),
            'talknet_ms_per_frame': round(ms_per_frame, 4),
            'talknet_avg_sec_per_pig': round(avg_per_pig, 4)}

# ============================================================
# 1. SED-CRNN
# ============================================================
def extract_mbe(y, sr, nfft, nb_mel):
    if y.ndim == 1:
        y = y[np.newaxis, :]
    mbe_all = []
    for ch in range(y.shape[0]):
        S = np.abs(librosa.stft(y[ch], n_fft=nfft, hop_length=HOP_LEN, center=False))
        mel = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=nb_mel)
        mbe_all.append((np.log(np.dot(mel, S) + 1e-8)).T)
    return np.concatenate(mbe_all, axis=1)

def sed_frame_probs(audio_path, threshold=0.5):
    y, sr_read = sf.read(audio_path)
    if sr_read != SR:
        y = librosa.resample(y.T if y.ndim == 2 else y, orig_sr=sr_read, target_sr=SR)
        if y.ndim == 2:
            y = y.T
    if y.ndim == 1:
        y = y[np.newaxis, :]
    elif y.shape[0] > y.shape[1]:
        y = y.T
    if y.shape[0] == 1:
        y = np.tile(y, (2, 1))
    mbe = extract_mbe(y, SR, NFFT, NB_MEL)
    mbe = joblib.load(SED_SCALER).transform(mbe)
    orig_frames = mbe.shape[0]
    pad = (SEQ_LEN - (mbe.shape[0] % SEQ_LEN)) % SEQ_LEN
    if pad > 0:
        mbe = np.vstack([mbe, np.zeros((pad, mbe.shape[1]))])
    data = mbe.reshape(mbe.shape[0] // SEQ_LEN, SEQ_LEN, mbe.shape[1])
    hop = data.shape[2] // NUM_CH
    data = np.stack([data[:, :, i * hop:(i + 1) * hop] for i in range(NUM_CH)], axis=1)
    model = load_model(SED_MODEL)
    pred = model.predict(data, verbose=0).reshape(-1, len(CLASS_LABELS))[:orig_frames]
    return pred

def extract_event_segments(pred, threshold=0.5, min_event_len=10, min_silence_len=5):
    events = []
    active = pred[:, 0] > threshold
    start, silence = None, 0
    for i, val in enumerate(active):
        if val:
            if start is None: start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_silence_len:
                end = i - silence + 1
                if (end - start) >= min_event_len:
                    seg = pred[start:end, 0]; mi = int(np.argmax(seg))
                    events.append({'start_time': round(start * FRAME_DUR, 3),
                                   'end_time': round(end * FRAME_DUR, 3),
                                   'max_time': round((start + mi) * FRAME_DUR, 3),
                                   'max_prob': round(float(seg[mi]), 4),
                                   'max_frame_idx': int(start + mi)})
                start = None
    if start is not None and (len(active) - start) >= min_event_len:
        seg = pred[start:, 0]; mi = int(np.argmax(seg))
        events.append({'start_time': round(start * FRAME_DUR, 3),
                       'end_time': round(len(active) * FRAME_DUR, 3),
                       'max_time': round((start + mi) * FRAME_DUR, 3),
                       'max_prob': round(float(seg[mi]), 4),
                       'max_frame_idx': int(start + mi)})
    return events

def pick_peak_frame(pred_2d, events):
    if events:
        best = max(events, key=lambda e: e['max_prob'])
        return best['max_frame_idx'], best['max_prob'], best['max_time']
    idx = int(np.argmax(pred_2d[:, 0]))
    return idx, float(pred_2d[idx, 0]), idx * FRAME_DUR

# ============================================================
# 2. YOLO11
# ============================================================
def detect_pigs(yolo_model, frame_bgr, conf=0.25, iou=0.5):
    """返回 boxes: (N,4,2) 旋转框4角点, confs: (N,) 置信度. OBB任务优先, 兼容 detect."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, frame_bgr)
    try:
        res = yolo_model.predict(source=tmp_path, imgsz=640, conf=conf, iou=iou,
                                 device='cpu', verbose=False)[0]
        boxes, confs = [], []
        if getattr(res, 'obb', None) is not None and len(res.obb) > 0:
            boxes = res.obb.xyxyxyxy.cpu().numpy().astype(float)  # (N,4,2)
            confs = res.obb.conf.cpu().numpy().astype(float)
        elif res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
            confs = res.boxes.conf.cpu().numpy().astype(float)
            boxes = np.stack([xyxy[:,[0,0]], xyxy[:,[1,1]],
                              xyxy[:,[2,2]], xyxy[:,[3,3]]], axis=1)  # 退化为水平4角点
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return boxes, confs


def _rotate_crop_face(frame_bgr, pts4, out_size=112):
    """pts4: (4,2) 顺时针4角点(TL,TR,BR,BL). 仿射变换摆正后裁剪到 out_size x out_size 灰度图."""
    pts = np.asarray(pts4, dtype=np.float32)
    cx, cy = pts.mean(axis=0)
    angle_deg = np.degrees(np.arctan2(pts[1,1]-pts[0,1], pts[1,0]-pts[0,0]))
    w = float(np.linalg.norm(pts[1]-pts[0]))
    h = float(np.linalg.norm(pts[2]-pts[1]))
    w, h = max(int(round(w)), 8), max(int(round(h)), 8)
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle_deg, 1.0)
    M[0,2] += -cx + w/2
    M[1,2] += -cy + h/2
    warped = cv2.warpAffine(frame_bgr, M, (w, h), borderValue=(0,0,0))
    if warped.shape[0] < 4 or warped.shape[1] < 4:
        return None
    return cv2.resize(warped, (out_size, out_size))

# ============================================================
# 3. 逐猪裁剪
# ============================================================
def crop_pig_clips(video_path, boxes, out_root, video_id, margin=0.0):
    """boxes: (N,4,2) 旋转框4角点. margin 外扩比例(相对中心点)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crops = []
    for k, pts in enumerate(boxes):
        pts = np.asarray(pts, dtype=np.float64)
        if margin > 0:
            cx, cy = pts.mean(axis=0)
            pts = cx + (pts[:,0]-cx)*(1+margin), cy + (pts[:,1]-cy)*(1+margin)
            pts = np.stack(pts, axis=1)
        pig_dir = os.path.join(out_root, "crops", video_id, f"pig{k}")
        os.makedirs(pig_dir, exist_ok=True)
        crops.append({'pig_id': f'pig{k}', 'pts4': pts.astype(np.float32).tolist(), 'dir': pig_dir})
    n_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        for c in crops:
            face = _rotate_crop_face(frame, c['pts4'], out_size=112)
            if face is None:
                face = np.zeros((112,112,3), dtype=np.uint8)
            cv2.imwrite(os.path.join(c['dir'], "%06d.jpg" % (n_frames + 1)), face)
        n_frames += 1
    cap.release()
    return crops, n_frames, fps

# ============================================================
# 4. 音频特征 + 对齐
# ============================================================
def _is_dense(layer):
    if isinstance(layer, keras.layers.Dense): return True
    if isinstance(layer, keras.layers.TimeDistributed): return isinstance(layer.layer, keras.layers.Dense)
    return False

def extract_talknet_audio_feats(audio_path):
    y, sr_read = sf.read(audio_path)
    if sr_read != SR:
        y = librosa.resample(y.T if y.ndim == 2 else y, orig_sr=sr_read, target_sr=SR)
        if y.ndim == 2: y = y.T
    if y.ndim == 1: y = y[np.newaxis, :]
    elif y.shape[0] > y.shape[1]: y = y.T
    if y.shape[0] == 1:
        y = np.tile(y, (2, 1))
    mbe = extract_mbe(y, SR, NFFT, NB_MEL)
    mbe = joblib.load(SED_SCALER).transform(mbe)
    real_frames = mbe.shape[0]
    pad = (SEQ_LEN - real_frames % SEQ_LEN) % SEQ_LEN
    if pad > 0:
        mbe = np.vstack([mbe, np.zeros((pad, mbe.shape[1]))])
    data = mbe.reshape(mbe.shape[0] // SEQ_LEN, SEQ_LEN, mbe.shape[1])
    hop = data.shape[2] // NUM_CH
    data = np.stack([data[:, :, i * hop:(i + 1) * hop] for i in range(NUM_CH)], axis=1)
    full = load_model(SED_MODEL)
    dense_idx = [i for i, l in enumerate(full.layers) if _is_dense(l)]
    feat_model = Model(full.input, full.layers[dense_idx[-2] - 1].output)
    raw = feat_model.predict(data, verbose=0)
    feat = raw.reshape(raw.shape[0] * raw.shape[1], -1)[:real_frames]
    return feat.astype(np.float32)

def align_audio_to_video(feats, n_video_frames, fps):
    idx = np.round(np.arange(n_video_frames) / fps / FRAME_DUR).astype(int)
    idx = np.clip(idx, 0, len(feats) - 1)
    return feats[idx]

# ============================================================
# 5. TalkNet 推理
# ============================================================
def load_talknet(model_path):
    from talkNet import talkNet, DEVICE
    net = talkNet(visualOnly=False, fusionMode='channel_attn')
    net.loadParameters(model_path)
    net = net.to(DEVICE)
    net.eval()
    return net

def talknet_infer(net, audio_feat_T32, pig_crop_dir, n_frames):
    import torch, glob
    face_files = sorted(glob.glob(os.path.join(pig_crop_dir, "*.jpg")),
                        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    faces = []
    for f in face_files[:n_frames]:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # 读彩色后在内存转灰度, 满足 Conv3d(1,...)
        faces.append(cv2.resize(img, (112, 112)))
    from talkNet import DEVICE
    visual = torch.FloatTensor(np.array(faces))[None].to(DEVICE)
    audio = torch.FloatTensor(audio_feat_T32[:n_frames])[None].to(DEVICE)
    with torch.no_grad():
        a = net.model.forward_audio_frontend(audio)
        v = net.model.forward_visual_frontend(visual)
        out = net.model.forward_audio_visual_backend(a, v)
        probs = net.lossAV.forward(out, labels=None)
    return np.asarray(probs)

def probs_to_intervals(probs, fps, threshold=0.5, min_len=2, min_gap=1):
    active = probs >= threshold
    intervals, start, gap = [], None, 0
    for i, v in enumerate(active):
        if v:
            if start is None: start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                end = i - gap + 1
                if end - start >= min_len:
                    intervals.append([round(start / fps, 3), round(end / fps, 3)])
                start = None
    if start is not None and len(active) - start >= min_len:
        intervals.append([round(start / fps, 3), round(len(active) / fps, 3)])
    return intervals

# ============================================================
# 6. 红框标注
# ============================================================
def annotate_video(video_path, crops, probs_dict, out_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        for c in crops:
            p = probs_dict.get(c['pig_id'])
            if p is None or t >= len(p): continue
            pts_int = np.array(c['pts4'], dtype=np.int32)
            if p[t] >= threshold:
                cv2.polylines(frame, [pts_int], True, (0, 0, 255), 3)
                cv2.putText(frame, "cough %.2f" % p[t], (int(pts_int[0,0]), max(20, int(pts_int[0,1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.polylines(frame, [pts_int], True, (255, 255, 255), 1)
        writer.write(frame); t += 1
    cap.release(); writer.release()

# ============================================================
# 主流程
# ============================================================
def _find_ffmpeg():
    """探测可用的 ffmpeg 二进制: 优先系统 ffmpeg, 回退 imageio_ffmpeg 自带二进制"""
    from shutil import which
    p = which('ffmpeg')
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "未找到 ffmpeg。请任选其一:\n"
            "  1) apt-get install -y ffmpeg\n"
            "  2) conda install -c conda-forge ffmpeg\n"
            "  3) pip install imageio-ffmpeg  (本脚本会自动用其自带的 ffmpeg 二进制)")


def extract_audio_from_video(video_path, out_wav):
    ff = _find_ffmpeg()
    cmd = [ff, "-y", "-i", video_path, "-vn", "-ar", str(SR), "-ac", "2", out_wav]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav

def main():
    ap = argparse.ArgumentParser(description="猪只咳嗽音视频联合推理")
    ap.add_argument('--video', type=str, required=True)
    ap.add_argument('--audio', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default='./inference_out')
    ap.add_argument('--sed_threshold', type=float, default=0.5)
    ap.add_argument('--talknet_threshold', type=float, default=0.5)
    ap.add_argument('--yolo_conf', type=float, default=0.6)
    ap.add_argument('--yolo_iou', type=float, default=0.7)
    ap.add_argument('--box_margin', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(args.video))[0]
    stage_times = {}   # 各阶段耗时
    t_global = time.time()

    # ---------- 1. SED-CRNN ----------
    print("[1/6] SED-CRNN 咳嗽检测 ...")
    with Timer("SED-CRNN 推理", stage_times):
        audio_path = args.audio or extract_audio_from_video(
            args.video, os.path.join(args.output_dir, "audio.wav"))
        pred_2d = sed_frame_probs(audio_path, args.sed_threshold)
        events = extract_event_segments(pred_2d, args.sed_threshold)
        peak_idx, peak_prob, peak_time = pick_peak_frame(pred_2d, events)
    print("  检测到 %d 个咳嗽区间" % len(events))
    for i, e in enumerate(events, 1):
        print("   #%d  %.2fs~%.2fs peak=%.2fs(p=%.3f)" % (i, e['start_time'], e['end_time'], e['max_time'], e['max_prob']))
    print("  咳嗽概率最大帧: t=%.3fs, p=%.4f" % (peak_time, peak_prob))

    # ---------- 2. YOLO11 ----------
    print("[2/6] YOLO11 检测峰值帧猪只 ...")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    peak_fid = int(np.clip(round(peak_time * fps), 0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_fid)
    ok, peak_frame = cap.read(); cap.release()
    if not ok: raise RuntimeError("峰值帧读取失败")
    cv2.imwrite(os.path.join(args.output_dir, "peak_frame.jpg"), peak_frame)
    from ultralytics import YOLO
    yolo = YOLO(YOLO_WEIGHTS)
    with Timer("YOLO11 检测", stage_times):
        boxes, confs = detect_pigs(yolo, peak_frame, args.yolo_conf, args.yolo_iou)
    if len(boxes) == 0:
        cv2.imwrite(os.path.join(args.output_dir, "peak_frame_no_det.jpg"), peak_frame)
        raise RuntimeError(
            "峰值帧未检测到猪只。已保存 peak_frame_no_det.jpg 供查看。\n"
            "可尝试: (1) 降低 --yolo_conf (默认已改为 0.25); "
            "(2) 检查 peak_frame.jpg 中猪只是否清晰可见; "
            "(3) 换用其他 YOLO 权重。")
    for i, (b, c) in enumerate(zip(boxes, confs)):
        pts_int = np.array(b, dtype=np.int32)
        cv2.polylines(peak_frame, [pts_int], True, (0, 255, 0), 2)
        cv2.putText(peak_frame, "pig%d %.2f" % (i, c), (int(pts_int[0,0]), int(pts_int[0,1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "peak_frame_yolo.jpg"), peak_frame)
    print("  检测到 %d 只猪 (peak video frame = %d)" % (len(boxes), peak_fid))

    # ---------- 3. 逐猪裁剪 ----------
    print("[3/6] 裁剪每只猪的视频序列 ...")
    with Timer("逐猪视频裁剪", stage_times):
        crops, n_frames, _ = crop_pig_clips(args.video, boxes, args.output_dir, video_id, margin=args.box_margin)
    print("  共 %d 帧, %d 个猪只序列" % (n_frames, len(crops)))

    # ---------- 4. 音频特征 ----------
    print("[4/6] 提取TalkNet音频特征并按时间对齐 ...")
    with Timer("音频特征提取+对齐", stage_times):
        feats = extract_talknet_audio_feats(audio_path)
        feats_aligned = align_audio_to_video(feats, n_frames, fps)
    np.save(os.path.join(args.output_dir, "audio_feat_aligned.npy"), feats_aligned)
    print("  音频特征: %s -> 对齐后: %s" % (feats.shape, feats_aligned.shape))

    # ---------- 5. TalkNet 推理 ----------
    print("[5/6] TalkNet 音视频融合推理 ...")
    net = load_talknet(TALKNET_MODEL)

    # 模型复杂度统计(参数量 + FLOPs)
    complexity = get_model_complexity(net)
    print("\n  ===== TalkNet 模型复杂度 =====")
    print("  参数量 (total)       : %s (%d)" % (complexity['params_total_fmt'], complexity['params_total']))
    print("  参数量 (trainable)   : %s (%d)" % (_fmt_num(complexity['params_trainable']), complexity['params_trainable']))
    print("  FLOPs (T=50, 单样本) : %s  [库: %s]" % (complexity['flops_fmt'], complexity['flops_lib']))

    probs_dict, summary, per_pig_times = {}, {}, []
    for c in crops:
        t0 = time.time()
        p = talknet_infer(net, feats_aligned, c['dir'], n_frames)
        dt = time.time() - t0
        per_pig_times.append(dt)
        probs_dict[c['pig_id']] = p
        iv = probs_to_intervals(p, fps, args.talknet_threshold)
        pts = np.array(c['pts4'])
        x1, y1 = pts[..., 0].min(), pts[..., 1].min()
        x2, y2 = pts[..., 0].max(), pts[..., 1].max()
        summary[c['pig_id']] = {'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
                                'pts4': c['pts4'],
                                'n_cough_frames': int((p >= args.talknet_threshold).sum()),
                                'mean_prob': round(float(p.mean()), 4),
                                'max_prob': round(float(p.max()), 4),
                                'cough_intervals_sec': iv,
                                'infer_sec': round(dt, 4)}
        print("  %s: 咳嗽帧 %d/%d, meanP=%.3f, maxP=%.3f, %.3fs" %
              (c['pig_id'], summary[c['pig_id']]['n_cough_frames'], n_frames,
               summary[c['pig_id']]['mean_prob'], summary[c['pig_id']]['max_prob'], dt))
    stage_times['TalkNet 推理(全部猪)'] = sum(per_pig_times)

    import pandas as pd
    rows = []
    for c in crops:
        pts = np.array(c['pts4'])
        x1, y1 = pts[..., 0].min(), pts[..., 1].min()
        x2, y2 = pts[..., 0].max(), pts[..., 1].max()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        # 旋转角度: 顶边(pts0->pts1) 与水平线夹角
        angle_deg = float(np.degrees(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])))
        rows.append({'video_id': video_id, 'pig_id': c['pig_id'],
                     'cx': cx, 'cy': cy, 'w': w, 'h': h, 'angle': angle_deg,
                     'frame_count': n_frames, 'fps': fps,
                     'score': json.dumps([round(float(x), 4) for x in probs_dict[c['pig_id']]], ensure_ascii=False)})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "talknet_probs.csv"), index=False)

    # ---------- 6. 红框标注 ----------
    print("[6/6] 标注咳嗽猪只并输出视频 ...")
    with Timer("红框标注输出", stage_times):
        out_video = os.path.join(args.output_dir, "annotated_%s.mp4" % video_id)
        annotate_video(args.video, crops, probs_dict, out_video, args.talknet_threshold)

    total_elapsed = time.time() - t_global
    speed = measure_speed(probs_dict, per_pig_times, n_frames)

    result = {'video': args.video, 'fps': fps, 'n_frames': n_frames,
              'sed_events': events, 'peak': {'time_sec': peak_time, 'video_frame': peak_fid, 'prob': peak_prob},
              'pigs': summary, 'annotated_video': out_video,
              'model_complexity': complexity,
              'stage_times_sec': {k: round(v, 4) for k, v in stage_times.items()},
              'talknet_speed': speed,
              'total_pipeline_sec': round(total_elapsed, 4)}
    with open(os.path.join(args.output_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ===== 性能汇总报告 =====
    print("\n" + "=" * 60)
    print("                    性能统计报告")
    print("=" * 60)
    print("[模型复杂度 TalkNet]")
    print("  参数量 (total)     : %s (%d)" % (complexity['params_total_fmt'], complexity['params_total']))
    print("  参数量 (trainable) : %s (%d)" % (_fmt_num(complexity['params_trainable']), complexity['params_trainable']))
    print("  FLOPs (T=50 样本)  : %s  [库: %s]" % (complexity['flops_fmt'], complexity['flops_lib']))
    print("\n[推理速度 TalkNet]")
    print("  总耗时             : %.4f s (%d 只猪)" % (speed['talknet_total_sec'], len(per_pig_times)))
    print("  帧率(FPS)          : %.2f fps" % speed['talknet_fps'])
    print("  每帧耗时           : %.4f ms" % speed['talknet_ms_per_frame'])
    print("  平均每只猪         : %.4f s" % speed['talknet_avg_sec_per_pig'])
    print("\n[各阶段耗时]")
    for k, v in stage_times.items():
        print("  %-28s : %.4f s" % (k, v))
    print("  %-28s : %.4f s" % ("管线总耗时", total_elapsed))
    print("=" * 60)
    print("\n[完成] 输出目录: %s" % args.output_dir)
    print("  - 标注视频: %s" % out_video)
    print("  - 峰值帧: peak_frame.jpg / peak_frame_yolo.jpg")
    print("  - 逐帧概率: talknet_probs.csv | 汇总(含性能): result.json")

if __name__ == '__main__':
    main()