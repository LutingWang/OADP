import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, \
    add_weight_decay
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import clip
import clip.model
import torchvision.transforms as tf

import todd

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', type=str, default='data/coco')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch-size', default=56, type=int,
                    metavar='N', help='mini-batch size')

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)


# from typing import Iterable
# import torch.nn as nn
# class CLIP(nn.Module):

#     def __init__(self, model: clip.model.CLIP, classes: Iterable[str]) -> None:
#         super().__init__()
#         self._model = model.visual
#         self._class_features = nn.Parameter(self.encode_class(model, classes), requires_grad=False)
#         self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
#         self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

#     @property
#     def dtype(self) -> torch.dtype:
#         return self._model.conv1.weight.dtype

#     def encode_class(self, model: clip.model.CLIP, classes: Iterable[str]) -> torch.Tensor:
#         class_list = [f"a photo containing {class_}" for class_ in classes]
#         class_tokens = clip.tokenize(class_list)
#         class_features: torch.Tensor = model.encode_text(class_tokens.cuda())
#         return class_features / class_features.norm(dim=-1, keepdim=True)

#     def encode_image(self, input: torch.Tensor) -> torch.Tensor:
#         image_features: torch.Tensor = self._model(input.type(self.dtype))
#         return image_features / image_features.norm(dim=-1, keepdim=True)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         image_features = self.encode_image(input)
#         output = image_features @ self._class_features.T
#         return output * self._scaler - self._bias

from coop import CustomCLIP as CLIP

def main():
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    # model = create_model(args).cuda()
    model, _ = clip.load('RN50', 'cpu')
    val_pipe = tf.Compose([
        tf.Resize(args.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(args.image_size),
        lambda image: image.convert("RGB"),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    train_pipe = tf.Compose([
        tf.Resize(args.image_size, interpolation=tf.InterpolationMode.BICUBIC),
        tf.CenterCrop(args.image_size),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        lambda image: image.convert("RGB"),
        tf.ToTensor(),
        tf.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


    print('done')

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2017.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_val2017.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2017.json')
    #data_path_val = args.data
    #data_path_train = args.data
    data_path_val = f'{args.data}/val2017'  # args.data
    data_path_train = f'{args.data}/val2017'  # args.data
    data_path_train = f'{args.data}/train2017'  # args.data
    val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                val_pipe,
                                )
    train_dataset = CocoDetection(data_path_train,
                                  instances_path_train,
                                  train_pipe,
                                  )
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # Actuall Training
    model = CLIP(model, [cat['name'] for cat in train_loader.dataset.coco.cats.values()])
    # model.load_state_dict(torch.load('models/model-4-2113.ckpt'))
    model.float()
    model.train()
    model.requires_grad_()
    model.cuda()
    train_multi_label_coco(model, train_loader, val_loader, args.lr)


def train_multi_label_coco(model, train_loader, val_loader, lr):
    # ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    # Epochs = 10
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()

            # ema.update(model)
            # store information
            if i % 20 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                            #   lr,
                              loss.item()))
                print(model._scaler, model._bias)

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()

        # mAP_score = validate_multi(val_loader, model, ema)
        mAP_score = validate_multi(val_loader, model)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    # preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                # output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        # preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    # mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    # print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    print("mAP score regular {:.2f}".format(mAP_score_regular))
    # return max(mAP_score_regular, mAP_score_ema)
    return mAP_score_regular


if __name__ == '__main__':
    main()
