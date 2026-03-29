from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.deeplabv3plus import DeepLabV3Plus


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 参数定义 ---
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument('--datasetTrain', nargs='+', type=int, default=[2, 3, 4],
                        help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', nargs='+', type=int, default=[1],
                        help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for training the model')
    parser.add_argument('--group-num', type=int, default=1, help='group number for group normalization')
    parser.add_argument('--max-epoch', type=int, default=120, help='max epoch')
    parser.add_argument('--stop-epoch', type=int, default=120, help='stop epoch')
    parser.add_argument('--interval-validate', type=int, default=1, help='interval epoch number to valide the model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate', )
    parser.add_argument('--lr-decrease-rate', type=float, default=0.2, help='ratio multiplied to initial lr')
    parser.add_argument('--data-dir', default='./Datasets', help='data root path')
    parser.add_argument('--pretrained-model', default='./models/pytorch/fcn16s_from_caffe.pth',
                        help='pretrained model of FCN16s', )
    parser.add_argument('--out-stride', type=int, default=16, help='out-stride of deeplabv3+', )
    parser.add_argument('--gamma', type=int, default=200, help='weight of IC', )
    args = parser.parse_args([])  # 如果是命令行运行，建议改为 parser.parse_args()

    # --- 自动创建日志目录 ---
    now = datetime.now()
    args.out = osp.join('logs', 'test', str(args.datasetTest), now.strftime('%Y%m%d_%H%M%S.%f'))
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # 保存配置
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    return args


def get_loaders(args):
    # --- 变换定义 ---
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        tr.RandomFlip(),
        tr.adjust_light(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    # --- DataLoader 初始化 ---
    train_domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='train',
                                         splitid=args.datasetTrain, transform=composed_transforms_tr)
    train_loader = DataLoader(train_domain, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())

    val_domain = DL.FundusSegmentation(base_dir=args.data_dir, phase='test',
                                       splitid=args.datasetTest, transform=composed_transforms_ts)
    val_loader = DataLoader(val_domain, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


def main():
    args = get_args()
    train_loader, val_loader = get_loaders(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(1337)

    model = DeepLabV3Plus(
        encoder_name='mobilenet_v2',
        encoder_weights="imagenet",
        in_channels=3,
        classes=2).to(device)

    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    start_epoch = 0
    start_iteration = 0

    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99)
    )

    trainer = Trainer.Trainer(
        cuda=torch.cuda.is_available(),
        model=model,
        lr=args.lr,
        lr_decrease_rate=args.lr_decrease_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        gam=args.gamma,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()



if __name__ == '__main__':
    main()
