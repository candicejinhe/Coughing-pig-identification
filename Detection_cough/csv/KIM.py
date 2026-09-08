# -*- coding: utf-8 -*-
"""
Kim et al. [22] non-learned baseline for pig cough detection.
Reply to Reviewer Comments 11: establish how much performance comes from
the learned temporal model by adding a non-learned baseline.

Algorithm (strictly following Kim et al.):
  1. Audio: extract mean pitch; if pitch falls into cough threshold range,
     a cough event is declared and video analysis is triggered.
  2. Video: for each pig's cropped frame sequence, compute Motion History
     Image (MHI) with temporal length = 3 frames.
  3. Compute MHI motion change: changed_MHI_motion = after_MHI - before_MHI.
  4. Compute centroid moving distance between adjacent moments.
  5. Decision rule (paper thresholds):
       change of MHI motion > 1000
       centroid moving distance < 5
     A pig satisfying both is classified as coughing.

Dataset: Detection_cough (same clips_videos + audio_feature + csv as TalkNet)
Output: per-frame cough probability curve for each pig, aligned with audio
        waveform, for direct comparison with the learned temporal model.

Usage:
  cd /root/data1/TalkNet-ASD-main
  python kim_baseline.py --video A001
  python kim_baseline.py --video A001 --pitch_low 200 --pitch_high 600 \
      --mhi_thresh 1000 --centroid_thresh 5 --mhi_len 3
"""
import os, sys, glob, csv, json, argparse
import numpy as np
import cv2
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, 'Detection_cough')
CLIPS_ROOT  = os.path.join(DATA_ROOT, 'clips_videos')
AUDIO_ROOT  = os.path.join(DATA_ROOT, 'audio_feature')
CSV_PATH    = os.path.join(DATA_ROOT, 'csv/val_loader.csv')
VIDEO_ROOT  = '/root/data1/PIG-recognition/PIG_data/clip_video'

rcParams['font.family']      = 'serif'
rcParams['font.serif']       = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

# ===================== 1. Audio: pitch-based cough detection =====================
def extract_audio_waveform(video_path, sr_target=16000):
    import subprocess, tempfile, imageio_ffmpeg
    tmp_wav = os.path.join(tempfile.gettempdir(), '_kim_audio.wav')
    ffmpeg  = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-i', video_path, '-ac', '1',
                    '-ar', str(sr_target), tmp_wav],
                   check=True, capture_output=True)
    y, sr = sf.read(tmp_wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    return y, sr

def detect_cough_by_pitch(y, sr, fps=25,
                          pitch_low=200.0, pitch_high=600.0,
                          frame_hop_sec=None):
    n_frames_total = int(len(y) / sr * fps)
    if frame_hop_sec is None:
        frame_hop_sec = 1.0 / fps
    cough_flags  = np.zeros(n_frames_total, dtype=bool)
    pitch_series = np.zeros(n_frames_total, dtype=np.float32)
    for i in range(n_frames_total):
        t0 = i * frame_hop_sec
        t1 = t0 + frame_hop_sec
        s0 = int(t0 * sr); s1 = int(t1 * sr)
        seg = y[s0:s1]
        if len(seg) < int(0.02 * sr):
            continue
        try:
            f0, voiced = librosa.piptrack(y=seg, sr=sr, fmin=80, fmax=1000)
            idx = np.argmax(f0, axis=0)
            freq = f0[idx, np.arange(f0.shape[1])]
            vmean = np.mean(freq[voiced > 0]) if np.any(voiced > 0) else 0.0
        except Exception:
            vmean = 0.0
        pitch_series[i] = vmean
        if pitch_low <= vmean <= pitch_high:
            cough_flags[i] = True
    return cough_flags, pitch_series

# ===================== 2. Video: MHI + centroid movement =====================
def load_pig_frames(pig_dir):
    files = sorted(glob.glob(os.path.join(pig_dir, 'frame_*.jpg')))
    frames = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames.append(img)
    return frames, files

def compute_mhi(frames, mhi_len=3, thresh=25):
    n = len(frames)
    H, W = frames[0].shape
    mhis = [np.zeros((H, W), dtype=np.float32) for _ in range(n)]
    for i in range(1, n):
        s = max(0, i - mhi_len)
        decay = 1.0 / max(mhi_len, 1)
        mhi = np.zeros((H, W), dtype=np.float32)
        for k in range(i - 1, s - 1, -1):
            if k < 0: break
            d = np.abs(frames[k + 1].astype(np.float32) - frames[k].astype(np.float32))
            m_k = (d > thresh).astype(np.float32)
            age = i - k
            mhi = np.maximum(mhi, m_k * max(0.0, 1.0 - (age - 1) * decay))
        mhis[i] = mhi
    return mhis

def mhi_motion_amount(mhi):
    return float(mhi.sum())

def compute_centroid(binary_motion):
    ys, xs = np.where(binary_motion > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)

def pig_cough_score(frames, mhi_len=3, mhi_thresh=25):
    n = len(frames)
    if n < 2 * mhi_len + 1:
        return {'mhi_change': np.zeros(n), 'centroid_dist': np.zeros(n)}
    mhis = compute_mhi(frames, mhi_len=mhi_len, thresh=mhi_thresh)
    mhi_change  = np.zeros(n, dtype=np.float32)
    centroid_d  = np.zeros(n, dtype=np.float32)
    for i in range(mhi_len, n - mhi_len):
        before = mhis[i - mhi_len]
        after  = mhis[i + mhi_len] if (i + mhi_len) < n else mhis[-1]
        mhi_change[i] = mhi_motion_amount(after) - mhi_motion_amount(before)
        before_c = compute_centroid((before > 0).astype(np.uint8))
        after_c  = compute_centroid((after  > 0).astype(np.uint8))
        if before_c is not None and after_c is not None:
            centroid_d[i] = float(np.linalg.norm(after_c - before_c))
    return {'mhi_change': mhi_change, 'centroid_dist': centroid_d}

def kim_decision(mhi_change, centroid_dist,
                 mhi_motion_thresh=1000.0, centroid_thresh=5.0):
    return ((mhi_change > mhi_motion_thresh) &
            (centroid_dist < centroid_thresh)).astype(np.float32)

# ===================== 3. Data loading =====================
def load_labels_from_csv(video_id):
    labels = {}
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] != video_id:
                continue
            lab = [int(x) for x in row['labels'].strip('"[]').split(',')]
            labels[row['pig_id']] = np.array(lab, dtype=np.int32)
    return labels

# ===================== 4. Main pipeline =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', type=str, default='A001')
    ap.add_argument('--split', type=str, default='val')
    ap.add_argument('--pitch_low',  type=float, default=200.0)
    ap.add_argument('--pitch_high', type=float, default=600.0)
    ap.add_argument('--mhi_len',    type=int,   default=3)
    ap.add_argument('--mhi_motion_thresh', type=float, default=1000.0)
    ap.add_argument('--centroid_thresh',   type=float, default=5.0)
    ap.add_argument('--fps', type=int, default=25)
    args = ap.parse_args()

    video_id   = args.video
    video_path = os.path.join(VIDEO_ROOT, video_id + '.mp4')
    clips_dir  = os.path.join(CLIPS_ROOT, args.split, video_id)
    if not os.path.isdir(clips_dir):
        print('找不到 clips 目录: %s' % clips_dir); sys.exit(1)

    print('=' * 64)
    print('Kim et al. [22] non-learned baseline  (%s)' % video_id)
    print('=' * 64)

    print('\n[1/4] 音频 pitch 提取与咳嗽事件检测 ...')
    y, sr = extract_audio_waveform(video_path, sr_target=16000)
    cough_flags, pitch_series = detect_cough_by_pitch(
        y, sr, fps=args.fps, pitch_low=args.pitch_low, pitch_high=args.pitch_high)
    n_audio_frames = len(cough_flags)
    print('  音频帧数: %d, 检测到咳嗽帧: %d (%.1f%%)' %
          (n_audio_frames, int(cough_flags.sum()), 100 * cough_flags.mean()))

    print('\n[2/4] 加载逐猪帧序列并计算 MHI / 质心位移 ...')
    pig_dirs = sorted([d for d in glob.glob(os.path.join(clips_dir, '*'))
                       if os.path.isdir(d)])
    pig_results = {}
    for pd in pig_dirs:
        pig_name = os.path.basename(pd).split('_')[0]
        frames, files = load_pig_frames(pd)
        if len(frames) < 2 * args.mhi_len + 1:
            print('  [跳过] %s 帧数不足: %d' % (pig_name, len(frames)))
            continue
        res = pig_cough_score(frames, mhi_len=args.mhi_len)
        pred = kim_decision(res['mhi_change'], res['centroid_dist'],
                            mhi_motion_thresh=args.mhi_motion_thresh,
                            centroid_thresh=args.centroid_thresh)
        n_frames = min(len(pred), n_audio_frames)
        pig_results[pig_name] = {
            'prob': pred[:n_frames],
            'mhi_change': res['mhi_change'][:n_frames],
            'centroid_dist': res['centroid_dist'][:n_frames],
            'n_frames': n_frames,
        }
        print('  %s: %d 帧, 预测咳嗽帧 %d' % (pig_name, n_frames, int(pred.sum())))

    print('\n[3/4] 加载真标签评估 ...')
    gt_labels = load_labels_from_csv(video_id)
    total_tp = total_fp = total_fn = 0
    for pig_name, res in pig_results.items():
        gt = gt_labels.get(pig_name)
        if gt is None:
            print('  [警告] %s 无真标签' % pig_name); continue
        n = min(len(res['prob']), len(gt))
        pred = res['prob'][:n].astype(int)
        gtb  = gt[:n].astype(int)
        tp = int(((pred == 1) & (gtb == 1)).sum())
        fp = int(((pred == 1) & (gtb == 0)).sum())
        fn = int(((pred == 0) & (gtb == 1)).sum())
        total_tp += tp; total_fp += fp; total_fn += fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print('  %s: P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d)' %
              (pig_name, prec, rec, f1, tp, fp, fn))
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print('  ---- 整体: P=%.3f R=%.3f F1=%.3f' % (prec, rec, f1))

    print('\n[4/4] 生成对比图与 JSON 报告 ...')
    _plot(video_id, y, sr, pig_results, gt_labels, args.fps)
    json_out = os.path.join(SCRIPT_DIR, 'kim_baseline_%s.json' % video_id)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump({k: {'prob': v['prob'].tolist(),
                       'mhi_change': v['mhi_change'].tolist(),
                       'centroid_dist': v['centroid_dist'].tolist()}
                   for k, v in pig_results.items()}, f)
    print('  JSON 报告: %s' % json_out)

def _plot(video_id, audio_wave, sr, pig_results, gt_labels, fps):
    n_frames = max(v['n_frames'] for v in pig_results.values())
    t_max = n_frames / fps
    t = np.linspace(0, t_max, n_frames)
    audio_t = np.linspace(0, len(audio_wave) / sr, len(audio_wave))

    cough_intervals = []
    for pig_name, gt in gt_labels.items():
        lab = np.asarray(gt)
        idx = np.where(lab == 1)[0]
        if len(idx) == 0: continue
        cough_intervals.append((idx[0] / fps, (idx[-1] + 1) / fps))

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
    fig = plt.figure(figsize=(14, 16))
    gs  = GridSpec(len(pig_results) + 2, 1,
                  height_ratios=[1.0] + [0.05] + [1.0] * len(pig_results),
                  hspace=0.05, left=0.08, right=0.97, top=0.98, bottom=0.05,
                  figure=fig)
    ax_w = fig.add_subplot(gs[0])
    ax_w.plot(audio_t, audio_wave, color='black', lw=0.5)
    for (a, b) in cough_intervals:
        ax_w.axvspan(a, b, color='red', alpha=0.20)
        ax_w.axvline(a, color='red', ls='--', lw=0.8, alpha=0.5)
        ax_w.axvline(b, color='red', ls='--', lw=0.8, alpha=0.5)
    ax_w.set_ylabel('Amplitude', fontsize=16)
    ax_w.set_xlim(0, t_max)
    ax_w.tick_params(labelbottom=False, labelsize=14)
    ax_w.text(-0.06, 1.0, '(a)', transform=ax_w.transAxes,
              fontsize=16, ha='right', va='top')
    if cough_intervals:
        mid = (cough_intervals[0][0] + cough_intervals[0][1]) / 2
        ax_w.annotate('Cough event', xy=(mid, ax_w.get_ylim()[1] * 0.9),
                    xytext=(mid, ax_w.get_ylim()[1] * 0.9),
                    color='red', fontsize=13, ha='center', style='italic')

    for i, (pig_name, res) in enumerate(pig_results.items()):
        ax = fig.add_subplot(gs[i + 2], sharex=ax_w)
        prob = res['prob']
        gt   = gt_labels.get(pig_name, np.zeros_like(prob))
        ax.plot(t[:len(prob)], prob, color=colors[i % len(colors)],
                lw=1.6, label='Kim prediction')
        if np.any(gt == 1):
            ax.fill_between(t[:len(gt)], 0, 1, where=(gt == 1),
                          color='red', alpha=0.15, label='GT cough')
        ax.axhline(0.5, color='gray', ls='--', lw=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(pig_name, fontsize=15)
        ax.tick_params(labelsize=13)
        if i < len(pig_results) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (s)', fontsize=15)
        ax.text(-0.06, 1.0, '(%c)' % chr(ord('b') + i), transform=ax.transAxes,
                fontsize=16, ha='right', va='top')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.8)

    out = os.path.join(SCRIPT_DIR, 'kim_baseline_%s.png' % video_id)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  对比图: %s' % out)

if __name__ == '__main__':
    main()