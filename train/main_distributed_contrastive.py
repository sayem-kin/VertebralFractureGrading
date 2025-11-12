import argparse, os, shutil, time
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from torch import amp

from model import Seresnet50_Contrastive
import transform as custom_transform
from dataset import Vertebrae_Dataset, ContrastiveBatchSampler
from utils import CustomLogger, calculate_confusion_matrix
from losses import SupConLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', '-t', type=str, default="debug")
    parser.add_argument('--ckpt_root', '-cp', type=str, default=r"/ckpt/ckpt2")
    parser.add_argument('--tensorboard', '-tb', type=str, default=r"/ckpt/tensorboard2")
    parser.add_argument('--dataset', '-d', type=str, default=r"/dataset")
    parser.add_argument('--dataset_tag', '-dt', type=str, default="delx")
    parser.add_argument("--load_ckpt", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths (debug mode → /tmp)
    args.ckpt_path = os.path.join(args.ckpt_root, args.tag)
    if args.tag == "debug":
        print("Debug mode, won't save any checkpoint")
        args.ckpt_path = os.path.join('/tmp', 'debug_ckpt')
        args.tensorboard = os.path.join('/tmp', 'debug_tensorboard')

    os.makedirs(args.ckpt_path, exist_ok=True)
    train_code_save = os.path.join(args.ckpt_path, "train_code")
    try:
        shutil.copytree(".", train_code_save, dirs_exist_ok=True)
    except Exception:
        pass

    logger = CustomLogger(os.path.join(args.ckpt_path, "test_log.log"),
                          os.path.join(args.ckpt_path, "test_log.csv"))

    # ---------- Transforms ----------
    t = transforms.Compose([
        custom_transform.RandomMask3D(20, 2, 0.5),
        transforms.RandomApply([
            custom_transform.RandomColorScale3D(0.1),
            custom_transform.RandomNoise3D(0.05),
            custom_transform.RandomRotation3D(10),
            custom_transform.RandomZoom3D(0.2),
            custom_transform.RandomShift3D(10),
        ], p=0.7),
        custom_transform.RandomAlign3D(128),   # If still OOM → try 96
        custom_transform.RandomMask3D(20, 2, 0.5)
    ])

    train_set = Vertebrae_Dataset(args.dataset, f"train_file_list.{args.dataset_tag}.yaml", transforms=[t, t])
    batch_sampler = ContrastiveBatchSampler(train_set)
    trainloader = DataLoader(train_set, num_workers=4, batch_sampler=batch_sampler,
                             pin_memory=True, persistent_workers=False)

    t_fix = transforms.Compose([custom_transform.FixedAlign3D(128)])
    test_set = Vertebrae_Dataset(args.dataset, f"test_file_list.{args.dataset_tag}.yaml", transforms=t_fix)
    testloader = DataLoader(test_set, batch_size=8, sampler=SequentialSampler(test_set),
                            num_workers=2, pin_memory=True, persistent_workers=False)

    writer = SummaryWriter(log_dir=os.path.join(args.tensorboard, args.tag))
    test_writer = SummaryWriter(log_dir=os.path.join(args.tensorboard, f"{args.tag}_test"))

    # ---------- Model / Loss / Optimizer ----------
    net = Seresnet50_Contrastive(spatial_dims=3, in_channels=3, num_classes=4, head='linear', feat_dim=128)
    net = net.to(device)

    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_con = SupConLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[800, 900], gamma=0.1)
    scaler = amp.GradScaler('cuda')


    confusion_matrix = torch.zeros([4, 4], device=device, requires_grad=False)

    # ---------- Training ----------
    g_step = 0
    epoch_init = 0
    for epoch in range(args.epochs):
        net.train()
        for i, data in enumerate(trainloader):
            inputs, GT, _ = data
            inputs = inputs.to(device, non_blocking=True)
            GT = GT.to(device, non_blocking=True).long()  # ✅ make sure labels are long

            # Split volume along dim=3
            inputs_s1, inputs_s2 = torch.split(inputs, [128, 128], dim=3)
            inputs2 = torch.cat([inputs_s1, inputs_s2], dim=0)
            GT2 = torch.cat([GT, GT], dim=0)

            optimizer.zero_grad(set_to_none=True)
            confusion_matrix.zero_()

            # ✅ AMP ON for network + CE loss
            with amp.autocast('cuda'):
                outputs, f = net(inputs2)
                f1, f2 = torch.split(f, [12, 12], dim=0)
                f_pair = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                l1 = criterion_cls(outputs, GT2)

            # ✅ AMP OFF for SupConLoss (stabilizes training)
            with amp.autocast('cuda', enabled=False):
                l2 = criterion_con(f_pair.to(torch.float32), GT)

            loss = l1 + l2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ✅ Logging & metrics
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                calculate_confusion_matrix(GT, preds[:GT.shape[0]], confusion_matrix)
                logger.result_log_train(confusion_matrix, float(loss.detach()), epoch + epoch_init, g_step, i, writer)
                g_step += 1


                scheduler.step()

        # ---------- Evaluation ----------
        net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, GT_cpu, _ in testloader:
                inputs = inputs.to(device, non_blocking=True)
                outputs, _ = net(inputs)
                outputs = F.softmax(outputs, dim=1)
                y_true.append(GT_cpu.numpy())
                y_pred.append(outputs.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred, axis=0)
        fs = logger.result_log_test(y_true, y_pred, epoch + epoch_init, g_step, test_writer)

        # Save checkpoints (optional in debug)
        state = {
            'g_step': g_step,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        logger.save_checkpoint(state, args.ckpt_path, fs, g_step)

        print(f"Epoch {epoch+1}/{args.epochs} done. Loss={loss.item():.4f}")
        torch.cuda.empty_cache()

    print("✅ Finished Training")


if __name__ == "__main__":
    main()
