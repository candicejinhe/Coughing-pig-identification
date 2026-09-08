import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, time, numpy, os, subprocess, pandas, tqdm

from loss import lossAV, lossA, lossV
from model.talkNetModel import talkNetModel
import pandas as pd
import numpy as np
import json, subprocess, tqdm, torch
import re, json

# 自动检测 CUDA 可用性, 不可用时退回 CPU
def _try_cuda():
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device('cuda')
    except Exception as e:
        print(time.strftime("%m-%d %H:%M:%S") + f" [talkNet] CUDA 不可用, 使用 CPU: {e}")
    return torch.device('cpu')

DEVICE = _try_cuda()
class talkNet(nn.Module):
    def __init__(self, lr = 0.0001, lrDecay = 0.95, visualOnly = False, fusionMode = 'channel_attn', **kwargs):
        super(talkNet, self).__init__()   
        self.visualOnly = visualOnly
        self.fusionMode = fusionMode if not visualOnly else None
        if visualOnly:
            self.model = talkNetModel(visualOnly=True).to(DEVICE)
            self.lossMain = lossV(w_neg=1.0, w_pos=2.8, label_smoothing=0.02).to(DEVICE)
            print(time.strftime("%m-%d %H:%M:%S") + " [talkNet] 纯视觉模式，不做音视频融合。")
        else:
            # 显式传入 fusionMode（channel_attn = 使用 model/fusion.py）
            self.model = talkNetModel(visualOnly=False, fusionMode=fusionMode).to(DEVICE)
            self.lossAV = lossAV(w_neg=1, w_pos=2.8, label_smoothing=0.02).to(DEVICE)
            self.lossV  = lossV (w_neg=1.0, w_pos=1, label_smoothing=0.0).to(DEVICE)
            mode_str = "通道注意力(Channel Attention, model/fusion.py)" if fusionMode == 'channel_attn' else "直接拼接(Concat Only)"
            print(time.strftime("%m-%d %H:%M:%S") + f" [talkNet] 视听融合模式：{mode_str}")
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))
    #训练过程
    def train_network(self, loader, epoch, **kwargs): 
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr'] 
        #遍历训练数据 loader（即训练集的数据加载器）。audioFeature、visualFeature 和 labels 分别是音频特征、视觉特征和标签。       
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            #清零模型的梯度，以便下一步的反向传播。
            self.zero_grad()
            if self.visualOnly:
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outMain = self.model.forward_visual_only_backend(visualEmbed)
                labels = labels[0].reshape((-1)).cuda()
                nlossMain, prob2, logits2, prec = self.lossMain.forward(outMain, labels, detail=True)
                nloss = nlossMain
            else:
                audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                outsV = self.model.forward_visual_backend(visualEmbed)
                labels = labels[0].reshape((-1)).cuda()
                nlossAV, prob2, logits2, prec = self.lossAV.forward(outsAV, labels)
                nlossV = self.lossV.forward(outsV, labels)
                nloss = nlossAV + 0.4 * nlossV
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  
        sys.stdout.write("\n")      
        return loss/num, lr
    
    @staticmethod
    def _safe_read_evalorig(path: str) -> pd.DataFrame:
        """
        稳健读取 eval/train/val CSV：
        - 支持 10 列：video_id,pig_id,cx,cy,w,h,angle,frame_count,fps,labels
        - 允许 labels 外层带引号
        - 允许逗号或空格分隔的 0/1
        输出列: video_id, pig_id, cx,cy,w,h,angle, frame_count, fps, labels(list[int]), frame_count_num
        """
        import csv, re, pandas as pd

        def _strip_outer_quotes(s: str) -> str:
            s = s.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                return s[1:-1].strip()
            return s

        def _labels_to_list(s: str):
            s = _strip_outer_quotes(s).strip()
            if not (s.startswith('[') and s.endswith(']')):
                return None
            body = s[1:-1].strip()
            if body == '':
                return []
            body = body.replace(',', ' ')
            toks = [t for t in body.split() if t != '']
            out = []
            for t in toks:
                try:
                    out.append(int(float(t)))
                except Exception:
                    out.append(0)
            return out

        rows = []
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if not first or (len(first) >= 1 and first[0].strip().lower() != "video_id"):
                f.seek(0)
                reader = csv.reader(f)

            for r in reader:
                if not r or (len(r) >= 1 and r[0].strip().lower() == "video_id"):
                    continue
                if len(r) < 10:
                    raise ValueError(f"CSV 列数不足 10: {r}")

                video_id = r[0].strip()
                pig_id   = r[1].strip()
                cx       = r[2].strip()
                cy       = r[3].strip()
                w        = r[4].strip()
                h        = r[5].strip()
                angle    = r[6].strip()
                frame_count = r[7].strip()
                fps      = r[8].strip()
                labels   = r[9].strip()

                try:
                    fc  = int(float(frame_count))
                    fps = float(fps)
                except Exception:
                    continue

                lab_list = _labels_to_list(labels)
                if lab_list is None:
                    raise ValueError(f"labels 非方括号序列：{labels}")

                rows.append({
                    'video_id': video_id,
                    'pig_id': pig_id,
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'angle': angle,
                    'frame_count': fc,
                    'fps': fps,
                    'labels': lab_list
                })

        if not rows:
            raise ValueError("文件为空或未解析到有效数据行。")

        df = pd.DataFrame(rows)
        df['frame_count_num'] = df['frame_count'].astype(int)
        return df


    #评估过程
    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        # 1) 读取并稳健解析 evalOrig
        df = self._safe_read_evalorig(evalOrig).copy()
        df["frame_count_num"] = df["frame_count_num"].astype(int)

        preview = []
        for _, row in df.head(3).iterrows():
            lab = row["labels"]
            if isinstance(lab, list):
                lab_len = len(lab)
            else:
                try:
                    lab_len = len(json.loads(str(lab))) if str(lab).strip().startswith("[") else None
                except Exception:
                    lab_len = None
            preview.append((row["video_id"], row["pig_id"], row["frame_count_num"], lab_len))

        # print("[EVAL] Preview first 3 rows (video_id, pig_id, frame_count_num, len(labels)):")
        # for r in preview: print("       ", r)

        total_frames = int(df["frame_count_num"].sum())
        #print(f"[EVAL] total_frames from evalOrig = {total_frames}")

        predScores = []# 扁平化累积所有样本的逐帧“说话/咳嗽=1”概率
        # === 新增：统计验证集 loss ===
        total_loss, total_count = 0.0, 0
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():     
                if self.visualOnly:
                    visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                    outMain = self.model.forward_visual_only_backend(visualEmbed)
                    labels = labels[0].reshape((-1)).cuda()
                    nlossMain, predScore, logits2, _ = self.lossMain.forward(outMain, labels, detail=True)
                    nloss = nlossMain
                else:
                    audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
                    visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                    outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed)
                    outsV = self.model.forward_visual_backend(visualEmbed)
                    labels = labels[0].reshape((-1)).cuda()
                    nlossAV, predScore, logits2, _ = self.lossAV.forward(outsAV, labels)
                    nlossV = self.lossV.forward(outsV, labels)
                    nloss = nlossAV + 0.4* nlossV
                # 记录验证 loss（按样本数做平均）
                bs = len(labels)
                total_loss  += float(nloss.detach().cpu()) * bs
                total_count += bs    
                prob_pos = predScore[:,1].detach().cpu().numpy()
                predScores.extend(prob_pos.tolist())
        val_loss = (total_loss / max(total_count, 1)) if total_count > 0 else None  
          # === 按你的 evalOrig 模板读取，并把逐帧概率写回 ===
        print(f"[EVAL] collected pred frames = {len(predScores)}")

        # 3) 对齐检查与友好提示
        if total_frames != len(predScores):
            # 给出更友好的诊断：显示差值，并提示检查 DataLoader 顺序与 evalOrig
            diff = len(predScores) - total_frames
            raise ValueError(
                f"总帧数({total_frames}) 与 累计预测数({len(predScores)}) 不匹配（差值 {diff}）。\n"
                "排查建议：\n"
                "  - 确认 DataLoader 的样本顺序与 evalOrig 的 (video_id, pig_id) 行顺序一致（建议使用 SequentialSampler）。\n"
                "  - 确认每个样本的推理输出帧数与该行 frame_count 对齐（若做了丢帧/补帧，需要在两端一致处理）。\n"
                "  - 确认 evalOrig 的分隔符与编码正确，frame_count 被正确解析为数值；上方已打印前 3 行供核对。"
            )

        # 4) 依次切片回填到每一行的 labels（随后改名为 score）
        cursor, out_scores = 0, []
        for _, row in df.iterrows():
            n = int(row["frame_count_num"])
            out_scores.append(predScores[cursor: cursor + n])
            cursor += n

        df["score"] = out_scores
        df_to_save = df.copy()
        df_to_save["score"] = df_to_save["score"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df_to_save.drop(columns=["frame_count_num"], inplace=True)
        df_to_save.to_csv(evalCsvSave, index=False)
        print(f"[EVAL] Saved predictions to {evalCsvSave}")


        # 占位：调用评估脚本（等你更新脚本以适配逐帧序列）
        cmd = f"python -O utils/get_ava_active_speaker_performance.py -g {evalOrig} -p {evalCsvSave}"
        # cmd = [sys.executable, "-O", "utils/get_ava_active_speaker_performance.py",
        #        "-g", str(evalOrig),
        #        "-p", str(evalCsvSave)]
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("[Eval Script STDOUT]\n", proc.stdout)
        print("[Eval Script STDERR]\n", proc.stderr)

        mAP = None
        try:
            for tok in proc.stdout.split():
                if tok.endswith("%") and tok[:-1].replace(".", "", 1).isdigit():
                    mAP = float(tok[:-1]); break
        except Exception:
            pass
        return mAP,val_loss

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location=DEVICE)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
