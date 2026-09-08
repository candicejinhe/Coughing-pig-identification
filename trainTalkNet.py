import time, os, torch, argparse, warnings, glob
import numpy as np
from dataLoader import train_loader, val_loader
from utils.tools import *
from talkNet import talkNet
import matplotlib.pyplot as plt  # 导入matplotlib库
import shutil  # 新增：用于复制最佳模型
import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior.
    torch.backends.cudnn.benchmark = False  # Avoids non-deterministic algorithms in cudnn.

def main():
    set_seed(42)
    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # 训练时的超参数
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=100,    help='Maximum number of epochs')#训练轮数
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    #dataLoader中的batchSize
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    #加载数据时使用的线程，默认是4
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # 数据集路径
    parser.add_argument('--dataPathAVA',  type=str, default="Detection_cough", help='Save path of AVA dataset')
    #实验输出目录（模型、日志等会放在这里）
    parser.add_argument('--savePath',     type=str, default="exps_test/visual_only4")#保存的目录
    # Data selection    999 
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    parser.add_argument('--visualOnly',      dest='visualOnly', action='store_true', help='Use visual modality only, no audio-visual fusion')
    # ===== 融合方式（通道注意力 = model/fusion.py）=====
    parser.add_argument('--fusionMode',      type=str, default='channel_attn', choices=['channel_attn', 'concat_only'],
                        help='Fusion mode for audio-visual: "channel_attn" uses model/fusion.py (1x1Conv+BN+Sigmoid channel attention), "concat_only" for plain concat without attention.')
    # ===== Early Stopping 超参（新增）=====
    parser.add_argument('--earlyStopPatience', type=int,   default=10,help='mAP 连续多少个评测周期无提升则早停')
    parser.add_argument('--earlyStopMinEpoch', type=int,   default=10, help='至少训练多少个 epoch 后才允许早停')
    parser.add_argument('--earlyStopDelta',    type=float, default=0.01,help='提升阈值（mAP 至少提高这么多才算“更好”，单位：绝对百分点）')


    args = parser.parse_args()

    # ========== 参数一致性检查 ==========
    if args.visualOnly and args.fusionMode != 'channel_attn':
        print(f"[WARN] visualOnly 模式下不进行音视频融合，fusionMode={args.fusionMode} 被忽略，强制不融合。")
    if not args.visualOnly:
        if args.fusionMode == 'channel_attn':
            print("[FUSION] 启用 音视频通道注意力融合（使用 model/fusion.py：1×1 Conv + BN + Sigmoid 通道门控）。")
        elif args.fusionMode == 'concat_only':
            print("[FUSION] 启用 音视频直接拼接融合（无注意力，仅 concat）。")

    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()
    #加载训练数据
    #实例化自定义的 train_loader
    loader = train_loader(trialFileName = args.trainTrialAVA, \
                          audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                          visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                          **vars(args))
    #用 PyTorch 的 DataLoader 包一层：这里的 batch_size=1 指“每次迭代取出 train_loader 构造好的一个动态小批”
    # （它内部把多条样本堆成一个 batch），所以外层 DataLoader 只需批量为 1。
    # shuffle=True：打乱这些动态小批的顺序。num_workers：并行加载线程数（从命令行参数）
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread)
    #验证集数据加载
    loader = val_loader(trialFileName = args.evalTrialAVA, \
                        audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                        visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = 16)
    #仅评测流程（用预训练模型）
    if args.evaluation == True:
        download_pretrain_model_AVA()
        s = talkNet(**vars(args))
        s.loadParameters('pretrain_AVA.model')
        print("Model %s loaded from previous state!"%('pretrain_AVA.model'))
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()
    #如果目录里已经有 model_000x.model 文件，就会认为是接着训练，把 epoch 设置为 上一次训练到的 epoch + 1。
    #如果想从头训练，删除即可。
    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        #实例化talkNet模型
        s = talkNet(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = talkNet(epoch = epoch, **vars(args))

    # 记录训练损失和验证集mAP
    train_losses = []
    val_losses   = []    # 新增：验证集 loss
    val_mAPs     = []
    # （如将来要画 train mAP，可再加 train_mAPs = []）

    #训练—评测—保存主循环
    #初始化评测历史与日志文件
    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")
    # ===== Early Stopping 状态（新增）=====
    best_mAP = -1e9
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(args.modelSavePath, 'model_best.model')

    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        # 记录训练损失
        train_losses.append(loss)
        #定期评测与保存模型
        if epoch % args.testInterval == 0:        
            # s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            #mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            mAP , val_loss= s.evaluate_network(epoch=epoch, loader=valLoader, **vars(args))
            mAPs.append(mAP)
            val_mAPs.append(mAP)  # 记录验证集mAP
            val_losses.append(val_loss if val_loss is not None else float('nan'))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()
                        # ===== Early Stopping 判定（新增）=====、
            last_val_csv = os.path.join(args.savePath, 'val_res.csv')   
            # mAP 提升需超过 earlyStopDelta 才算“更好”
            if mAP > (best_mAP + args.earlyStopDelta):
                best_mAP = mAP
                best_epoch = epoch
                epochs_no_improve = 0
                # 保存最佳模型（唯一模型）
                s.saveParameters(best_model_path)
                best_val_csv = os.path.join(args.savePath, 'val_res_best.csv')
                try:
                    if os.path.isfile(last_val_csv):
                        shutil.copyfile(last_val_csv, best_val_csv)
                        print(f"[BEST] 已更新最佳推理结果：{best_val_csv}")
                except Exception as e:
                    print(f"[WARN] 复制最佳推理结果失败：{e}")    
                print(f"[BEST] epoch {epoch} 刷新最佳 mAP={best_mAP:.2f}%，已更新 {best_model_path}")
            else:
                epochs_no_improve += 1
                print(f"[EARLY-STOP] 连续 {epochs_no_improve} 次评测无显著提升（阈值 {args.earlyStopDelta:.2f}）。")

            # 满足“最小 epoch 要求”且“耐心值用尽”时早停
            if (epoch >= args.earlyStopMinEpoch) and (epochs_no_improve >= args.earlyStopPatience):
                print(f"[EARLY-STOP] 触发早停：在 {args.earlyStopPatience} 次评测内 mAP 未提升。")
                print(f"[EARLY-STOP] 最佳模型来自 epoch {best_epoch}，mAP={best_mAP:.2f}%。")
                break

        if epoch >= args.maxEpoch:
            #quit()
            break

        epoch += 1
        # ===== 训练结束后的总结（新增，可选）=====
    if best_epoch > 0:
        print(f"[SUMMARY] 训练结束：最佳 mAP={best_mAP:.2f}% @ epoch {best_epoch}")
        print(f"[SUMMARY] 最佳模型路径：{best_model_path}")
    else:
        print("[SUMMARY] 未刷新最佳模型（可能未达到 earlyStopMinEpoch 或未评测到提升）。")

    # === 图 1：训练 vs 验证 Loss ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    # val loss 是按 testInterval 记录的，横轴用对应的 epoch 列表
    val_epochs = list(range(args.testInterval, args.testInterval*len(val_losses)+1, args.testInterval))
    plt.plot(val_epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.savePath, 'loss_curve.png'))
    plt.close()

    # === 图 2：训练 vs 验证 mAP（当前只有 Val mAP；Train mAP 可选）===
    plt.figure(figsize=(10, 6))
    # 如果以后你补了 train_mAPs，这里再画一条即可：
    # train_map_epochs = val_epochs  # 若你同频率评测 train
    # plt.plot(train_map_epochs, train_mAPs, label="Train mAP")

    plt.plot(val_epochs, val_mAPs, label="Val mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP (%)")
    plt.title("Validation mAP over Epochs")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.savePath, 'map_curve.png'))
    plt.close()

    print("曲线已保存：")
    print(" -", os.path.join(args.savePath, 'loss_curve.png'))
    print(" -", os.path.join(args.savePath, 'map_curve.png'))


if __name__ == '__main__':
    main()
