#rain_loader 会读取音频和视频帧文件 → 提取 MFCC 和人脸图像序列 → 
# 做随机增强（叠加噪声/翻转/裁剪/旋转）→ 按视频长度对齐 → 
# 打包成 batch → 返回 (audioFeatures, visualFeatures, labels)
import os, torch, numpy, cv2, random, glob,re
from torchvision.transforms import RandomCrop
import numpy as np
#CSV 是“视频ID, 猪只ID, 帧数, FPS, 标签序列”。
# 原代码按 \t + 特定格式解析；这里做通用解析（支持逗号或制表符），并兼容是否有表头。
def _strip_outer_quotes(s: str) -> str:
    """去掉最外层一对 ' 或 " 引号（若存在），并裁剪空格"""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1].strip()
    return s

def _parse_labels_to_list(labels_str: str):
    """
    稳健解析 labels：
    - 去外层引号
    - 只提取方括号内的 0/1（忽略逗号、空格、杂引号等）
    返回: List[int]（仅包含 0 或 1）
    """
    s = _strip_outer_quotes(labels_str).strip()  # 去外层引号
    # 允许 labels_str 可能没有最外层 []，这里尽量从第一次 '[' 到最后一个 ']' 之间提取
    lb, rb = s.find('['), s.rfind(']')
    if lb != -1 and rb != -1 and rb > lb:
        body = s[lb+1:rb]
    else:
        # 兜底：直接用整串
        body = s
    # 只取 0/1 字符，忽略其它（包括逗号、空格、单双引号等）
    toks = re.findall(r'[01]', body)
    return [int(t) for t in toks]
def _parse_line(line: str):
    if '\t' in line:
        parts = line.split('\t', 9)  # 10列
    else:
        parts = line.split(',', 9)

    if len(parts) < 10:
        raise ValueError(f"Bad csv line (need >=10 cols): {line}")

    video_id   = parts[0].strip()
    pig_id     = parts[1].strip()
    cx         = parts[2].strip()
    cy         = parts[3].strip()
    w          = parts[4].strip()
    h          = parts[5].strip()
    angle      = parts[6].strip()
    num_frames = int(parts[7].strip())
    fps        = float(parts[8].strip())
    labels_str = _strip_outer_quotes(parts[9].strip())

    return video_id, pig_id, cx, cy, w, h, angle, num_frames, fps, labels_str



#将同一 batch所需的 video_id.npy 读入内存缓存，键值仍沿用 dataName（此处用 video_id 来充当）。
def generate_audio_set(dataPath, batchList):
    """
    dataPath: 你的音频特征根目录（例如 audio_feature）
    batchList: 本 batch 的原始行列表（字符串）
    返回: { video_id: np.ndarray (T,128) }
    """
    audioSet = {}
    needed_ids = set()
    for line in batchList:
        try:
            video_id, pig_id, cx, cy, w, h, angle, num_frames, fps, labels_str = _parse_line(line)
        except:
            # 跳过表头或异常行
            continue
        needed_ids.add(video_id)

    for vid in needed_ids:
        npy_path = os.path.join(dataPath, f"{vid}.npy")
        feats = numpy.load(npy_path)  # 期望 shape=(T,128)
        # 基本健壮性检查
        #if feats.ndim != 2 or feats.shape[1] != 128:
        #   raise ValueError(f"Audio feature shape must be (T,128), got {feats.shape} for {npy_path}")
        if feats.ndim != 2 or feats.shape[1] not in (32, 128):
            raise ValueError(f"Audio feature shape must be (T,32) or (T,128), got {feats.shape} for {npy_path}")
        audioSet[vid] = feats.astype('float32')
    return audioSet


def load_audio(data, dataPath, numFrames, audioAug, audioSet=None):
    """
    data: 通过 _parse_line 得到的 fields 列表或上层传进来的 parts
    dataPath: audio_feature 根目录
    numFrames: 本 batch 对齐长度（由 miniBatch 机制决定）
    audioAug: 保留参数（不使用）
    audioSet: generate_audio_set 返回的缓存 {video_id: (T,128)}
    """
    # 兼容旧调用：如果 data 是原始字符串，先解析
    if isinstance(data, str):
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = _parse_line(data)
    else:
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = data


    feats = audioSet[video_id]  # (T,32)
    #批内比 numFrames 更长的样本会被截断（信息丢失）；（先跑完看要不要改）
    if feats.shape[0] < numFrames:
        # 若真的比 numFrames 短，则零填充到对齐（避免 DataLoader 拼 batch 报错）
        pad = numpy.zeros((numFrames - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
        feats = numpy.concatenate([feats, pad], axis=0)
    else:
        feats = feats[:numFrames, :]
    return feats


def load_visual(data, dataPath, numFrames, visualAug): 
    if isinstance(data, str):
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = _parse_line(data)
    else:
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = data

    # ========= 新增：如果目录命名是 pig（泛指/不唯一），强制用 cx_cy_w_h_angle =========
    pig_id_norm = pig_id.strip().lower()
    force_bbox_dir = (pig_id_norm == "pig")  # 你也可以扩展：pig0 / pig_ 等

    if force_bbox_dir:
        subdir = f"{cx}_{cy}_{w}_{h}_{angle}"
        faceFolderPath = os.path.join(dataPath, video_id, subdir)
    else:
        # 先尝试 pig_id 目录（pig1/pig2/...）
        faceFolderPath = os.path.join(dataPath, video_id, pig_id)

        # pig_id 目录不存在或为空 -> fallback 到 cx_cy_w_h_angle
        if (not os.path.exists(faceFolderPath)) or (len(glob.glob("%s/*.jpg" % faceFolderPath)) == 0):
            subdir = f"{cx}_{cy}_{w}_{h}_{angle}"
            faceFolderPath = os.path.join(dataPath, video_id, subdir)

    faceFiles = glob.glob("%s/*.jpg" % faceFolderPath)

    def _numeric_key_from_path(p):
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r'(\d+)$', base) or re.search(r'(\d+)', base)
        return int(m.group(1)) if m else base

    sortedFaceFiles = sorted(faceFiles, key=_numeric_key_from_path)

    if len(sortedFaceFiles) == 0:
        raise FileNotFoundError(
            f"No frames found under: {faceFolderPath}. "
            f"Expecting images like 000001.jpg / frame_000001.jpg"
        )
    
    # augmentation 保持不变
    faces = []
    H = 112
    if visualAug:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'

    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


#原函数从 data[3] 的“[0,1,0,…]”里取；现在从第5列取，并支持空格或逗号分隔、可带方括号
def load_label(data, numFrames):
    if isinstance(data, str):
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = _parse_line(data)
    else:
        video_id, pig_id, cx, cy, w, h, angle, nframes, fps, labels_str = data

    # ★ 使用稳健解析：仅抽取 0/1，自动忽略引号、逗号、空格等杂质
    arr_list = _parse_labels_to_list(labels_str)
    arr = np.array(arr_list, dtype='int64')

    # 截断/补零对齐
    arr = arr[:numFrames]
    if arr.shape[0] < numFrames:
        pad = np.zeros((numFrames - arr.shape[0],), dtype='int64')
        arr = np.concatenate([arr, pad], axis=0)
    return arr


#训练时的数据加载器
class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, **kwargs):
        self.audioPath  = audioPath   # 这里应传入 audio_feature 根目录
        self.visualPath = visualPath  # 这里应传入视觉帧根目录，例如 crops
        self.miniBatch = []

        raw_lines = open(trialFileName, 'r', encoding='utf-8').read().splitlines()

        # 过滤掉表头：第三列不是数字的行
        clean_lines = []
        for ln in raw_lines:
            try:
                _ = _parse_line(ln)  # 能解析就是数据行
                clean_lines.append(ln)
            except:
                continue  # 跳过表头/坏行

        # 按 (num_frames, fps) 倒序排列（长序列优先，避免超大batch）
        sorted_lines = sorted(clean_lines, key=lambda ln: (
            int(_parse_line(ln)[7]),  # num_frames
            float(_parse_line(ln)[8]) # fps
        ), reverse=True)

        # 自适应分批：同一 batch 里样本的 num_frames 相同（取当前 start 处的长度）
        start = 0
        N = len(sorted_lines)
        while start < N:
            cur_len = int(_parse_line(sorted_lines[start])[7]) # 这一批起始样本的帧长
            B = max(int(batchSize / max(cur_len,1)), 1)  # 估算这一批能装多少条
            end = min(N, start + B)
            self.miniBatch.append(sorted_lines[start:end])  # 不再强制同长
            start = end

   

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        # 本 batch 的对齐长度 = 该 batch 最后一行的 num_frames（等长分批）
        numFrames  = int(_parse_line(batchList[-1])[7])
        audioFeatures, visualFeatures, labels = [], [], []
        # 只加载本 batch 需要的所有 video_id 的 .npy
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            fields = _parse_line(line)  # (video_id, pig_id, num_frames, fps, labels_str)           
            #音频特征提取
            audioFeatures.append(load_audio(fields, self.audioPath, numFrames, audioAug = False, audioSet = audioSet))  
            #视觉特征提取
            visualFeatures.append(load_visual(fields, self.visualPath,numFrames, visualAug = True))
            labels.append(load_label(fields, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        
#一个 batch 输出：这一组视频的音频特征, 图像特征, 标签
    def __len__(self):
        return len(self.miniBatch)

#验证集同样直接加载 (T,128) 的 .npy；不做增强。
class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        raw_lines = open(trialFileName, 'r', encoding='utf-8').read().splitlines()
        # 过滤可能的表头
        self.miniBatch = []
        for ln in raw_lines:
            try:
                _ = _parse_line(ln)
                self.miniBatch.append(ln)
            except:
                continue

    def __getitem__(self, index):
        # 验证阶段单条取样（保持原接口）
        line       = [self.miniBatch[index]]
        numFrames  = int(_parse_line(line[0])[7])
        audioSet   = generate_audio_set(self.audioPath, line)
        fields     = _parse_line(line[0])

        audioFeatures  = [load_audio(fields,  self.audioPath, numFrames, audioAug=False, audioSet=audioSet)]
        visualFeatures = [load_visual(fields, self.visualPath, numFrames, visualAug=False)]
        labels         = [load_label(fields,  numFrames)]

        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)
