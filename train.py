import os
import time
import torch
import wandb
import datetime
import warnings
from argparse import Namespace

warnings.filterwarnings("ignore")
from models.mynet.BGINet import BGINet
from utils.change_data import MyDataset
from utils.distributed_utils import set_seed
from utils.distributed_utils import ConfusionMatrix
from utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = MyDataset(args.data_path)
    val_dataset = MyDataset(args.val_path)

    num_workers = 8
    print(num_workers)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True
                                             )
    model = BGINet(3, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=5)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.ckpt_url:
        print("使用预训练模型", args.ckpt_url)
        checkpoint = torch.load(args.ckpt_url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    # 开始时间
    start_time = time.time()
    best_F1 = 0.
    Last_epoch = 0
    save_path = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_path)
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            num_classes=num_classes,
            scaler=scaler)

        confmat = evaluate(model, val_loader,
                           device=device,
                           num_classes=num_classes, print_freq=args.print_freq)

        val_info = ConfusionMatrix.todict(confmat)
        val_info_print = str(confmat)
        # 各种评价指标
        precision = float(val_info['precision'][1])
        average_row_correct = float(val_info['average row correct'][1])
        Iou = float(val_info['IoU'][1])
        recall = float(val_info['recall'][1])
        Avg_precision = val_info['Avg_precision']
        F1 = float(val_info['F1_Score'][1])
        mean_Iou = val_info['mean IoU']

        print(val_info_print)
        if F1 == "nan":
            F1 = 0
        else:
            F1 = float(F1)
        save_txt = os.path.join(save_path, results_file)
        print(save_txt)
        with open(save_txt, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"

            f.write(train_info + val_info_print + "\n\n")
        if F1 > best_F1:
            best_F1 = F1
            Last_epoch = epoch
            model_name = "best.pth"
            save_url = os.path.join(save_path, model_name)
            print(save_url)
            torch.save(model, save_url)
        print("Best:", best_F1, )
        print("Best_epoch:", Last_epoch)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=args.project_name, config=args.__dict__, name=nowtime, save_code=True)
        wandb.log(
            {'epoch': epoch, 'F1': F1, 'precision': precision, 'IoU': Iou, 'recall': recall, 'mean_Iou': mean_Iou,
             'average_row_correct': average_row_correct, 'Avg_precision': Avg_precision, "lr": lr,
             "mean_loss": mean_loss, "best_F1": best_F1})
    print("best model in {} epoch".format(Last_epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # save code
    arti_code = wandb.Artifact('python', type='code')
    arti_code.add_file('./utils/change_data.py')
    arti_code.add_file('./utils/train_and_eval.py')
    arti_code.add_file('./utils/change_data.py')
    arti_code.add_file('train.py')
    wandb.log_artifact(arti_code)
    wandb.finish()


def parse_args():
    args = Namespace(
        project_name='ContrastModel',
        batch_size=32,
        data_path=r"D:\Datasets\Data_CD\LEVIR-CD\LEVIR-CD\256\train",
        val_path=r"D:\Datasets\Data_CD\LEVIR-CD\LEVIR-CD\256\val",
        out_path="./output",
        device="cuda",
        num_classes=1,
        lr=0.0004,
        print_freq=100,
        epochs=100,
        start_epoch=0,
        save_path='checkpoint.pt',
        ckpt_url=r"",
        amp=False,
        weight_decay=1e-4,
        seed=10)
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args)
    main(args)
