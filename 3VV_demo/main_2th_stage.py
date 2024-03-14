import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import autograd, optim
from UNet import Unet
from attention_unet import AttU_Net
from unetpp import NestedUNet
from fcn import get_fcn8s
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from utils1 import *
from argument import *
from torchvision.models import vgg16
import os
from loss import BinaryDiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=35)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='deeplabv3',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument('--dataset', default='MyDataset',  # dsb2018_256
                       help='dataset name')
    parse.add_argument("--log_dir", default='.../3VV_demo/log/', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    args = parse.parse_args()
    return args


def getLog(args):
    dirname = os.path.join(args.log_dir, 'last', '3VV')
    filename = dirname + '/' + args.arch + '_log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filemode='a'
    )
    return logging


def getModel(args):
    set_seed(1207)
    if args.arch == 'UNet':
        model = Unet(3, 4).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = False
        model = NestedUNet(args, 3, 1).to(device)
    if args.arch == 'Attention_UNet':
        model = AttU_Net(3, 3).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset != 'esophagus', "fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
        model = get_fcn8s(1).to(device)
    if args.arch == 'deeplabv3':
        from method_utils.deeplabv3 import DeepLabV3_gai
        model = DeepLabV3_gai(1, '.../3VV_demo/log/deeplabv3/').to(device)
    return model


def getDataset(args):
    train_dataloaders, val_dataloaders, test_dataloaders = None, None, None
    if args.dataset == 'MyDataset':  # E:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = MyDataset_trim(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_dataset = MyDataset_trim(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1, shuffle=True)
        test_dataset = MyDataset_trim(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloaders, val_dataloaders, test_dataloaders


def train(model, criterion1, criterion2, optimizer, train_dataloader, val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd = 0, 0, 0, 0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    val_loss_list = []
    set_seed(1207)

    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        # pbar = enumerate(train_dataloader)
        epoch_loss = 0
        ce_loss = 0
        dice_loss = 0
        step = 0
        for x, y, _, mask, CE_label in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            ce_labels = CE_label.to(device)

            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion1(output, labels)
                loss /= len(outputs)
            else:

                output = model(inputs)
                loss_1 = criterion1(torch.sigmoid(output), ce_labels)

                loss_2 = criterion2(torch.sigmoid(output[:, 1]), labels[:, 1])
                loss_2 += criterion2(torch.sigmoid(output[:, 2]), labels[:, 2])
                loss_2 += criterion2(torch.sigmoid(output[:, 3]), labels[:, 3])

                loss = 0.5*loss_1 + 0.5*loss_2

                ce_loss += loss_1.item()
                dice_loss += loss_2.item()

            if threshold != None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info(
                "%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou, aver_iou, aver_dice, aver_hd, val_loss_list = val(model, best_iou, val_dataloader, val_loss_list)


        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))

        if epoch > 10 and scheduler.get_last_lr()[0] > 1e-5:
            scheduler.step()
            print(scheduler.get_last_lr())

    return model


def val(model, best_iou, val_dataloaders, val_loss_list):
    model = model.eval()
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        hd_total = 0
        num = len(val_dataloaders)  # 验证集图片的总数

        miou_total_0, miou_total_1, miou_total_2 = 0, 0, 0
        dice_total_0, dice_total_1, dice_total_2 = 0, 0, 0

        print(num)
        batch_loss = 0
        for x, _, pic, mask, ce_label in val_dataloaders:
            x = x.to(device)
            _labels = _.to(device)

            y = model(x)
            y = torch.sigmoid(y)

            if args.deepsupervision:
                img_y = torch.squeeze(y).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            img_y = np.where(img_y < 0.5, 0, 1)
            hd = get_hd_ce(mask[0], img_y)
            hd_total += hd[1] + hd[2] + hd[3]

            iou, dice = get_iou_ce(mask[0], img_y)

            miou_total_0 += iou[0]
            dice_total_0 += dice[0]
            miou_total_1 += iou[1]
            dice_total_1 += dice[1]
            miou_total_2 += iou[2]
            dice_total_2 += dice[2]


            if i < num: i += 1  # 处理验证集下一张图
        val_loss_list.append(batch_loss)
        aver_iou = (miou_total_0 + miou_total_1 + miou_total_2) / (3 * num)
        aver_hd = hd_total / (3*num)
        aver_dice = (dice_total_0 + dice_total_1 + dice_total_2) / (3 * num)

        print('aver_iou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f' % (aver_iou, aver_hd, aver_dice))

        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou, best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'.../3VV_demo/saved_model/2th_stage_seg/' + str(args.arch) + '_' + str(
                           args.batch_size) + 'deeplabv3.pth')
        return best_iou, aver_iou, aver_dice, aver_hd, val_loss_list


def test(val_dataloaders, save_predict=False):
    logging.info('final test........')
    if save_predict == True:
        dir = r'.../3VV_demo/result/2th_stage_seg/' + str(args.arch) + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')

    model.load_state_dict(torch.load(r'.../3VV_demo/saved_model/2th_stage_seg/' + str(args.arch) + '_' + str(
             args.batch_size) + 'deeplabv3.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()
    # plt.ion() #开启动态模式
    with torch.no_grad():

        i = 0  # 验证集中第i张图
        hd_total = 0
        num_all = 0
        hd_total_P, hd_total_A, hd_total_V = 0, 0, 0

        miou_total_0, miou_total_1, miou_total_2 = 0, 0, 0
        dice_total_0, dice_total_1, dice_total_2 = 0, 0, 0

        num = len(val_dataloaders)  # 验证集图片的总数

        for pic, _, pic_path, mask_path, ce_label in val_dataloaders:
            num_all += 1

            pic = pic.to(device)
            predict = model(pic)
            predict = torch.sigmoid(predict)

            if args.deepsupervision:
                predict = torch.squeeze(predict).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  # 输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            predict = np.where(predict < 0.5, 0, 1)
            mask = np.array(_[0])

            iou, dice = get_iou_ce(mask_path[0], predict)
            hd = get_hd_ce(mask_path[0], predict)

            hd_total += hd[1] + hd[2] + hd[3]

            miou_total_0 += iou[0] # 获取当前预测图的miou，并加到总miou中
            dice_total_0 += dice[0]
            miou_total_1 += iou[1]
            dice_total_1 += dice[1]
            miou_total_2 += iou[2]
            dice_total_2 += dice[2]

            hd_total_P += hd[1]
            hd_total_A += hd[2]
            hd_total_V += hd[3]

            if save_predict == True:

                pred = recover_image_size(predict, mask_path[0])
                # pred = np.where(pred < 1, 0, 1)

                recover_txt_path = '.../3VV_demo/result/new_dataset/test/txt/'

                with open(recover_txt_path + pic_path[0].split('/')[-1].replace('.jpg', '.txt'), 'r') as f:
                    bbox = f.readlines()[0].split(' ')

                    image = cv2.imread('.../dataset/full_size_mask/' + mask_path[0].split('/')[-1])
                    image_height, image_width = image.shape[:2]

                    # 读取裁剪后的图像
                    # cropped_image = cv2.imread(mask_path[0])

                    # 创建一个和原始图像一样大小的黑色背景图像
                    background = np.zeros((image_height, image_width, 3), dtype=np.uint8)

                    # 计算裁剪后的图像在原始图像中的位置
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                    # 将裁剪后的图像放置到原始图像中的对应位置
                    background[y1:y2, x1:x2] = pred
                    # background[y1:y2, x1:x2, 1] = pred
                    # background[y1:y2, x1:x2, 2] = pred

                    cv2.imwrite(dir + mask_path[0].split('/')[-1], background)
                f.close()


            print('iou={},dice={}'.format(iou, dice))
            if i < num: i += 1  # 处理验证集下一张图

        # miou_total = miou_total_0 + miou_total_1 + miou_total_2
        # dice_total = dice_total_0 + dice_total_1 + dice_total_2
        #
        # print('Miou=%f,aver_hd=%f,M_dice=%f' % (miou_total / (3*num), hd_total / (3*num), dice_total / (3*num)))
        # print('P_iou=%f,A_iou=%f,V_iou=%f ,P_dice=%f,A_dice=%f,V_dice=%f' %(miou_total_0/num, miou_total_1/num, miou_total_2/num,
        #                                                                     dice_total_0/num, dice_total_1/num, dice_total_2/num))
        # print('P_hd = %f, A_hd = %f, V_hd = %f' %(hd_total_P/num, hd_total_A/num, hd_total_V/num))



if __name__ == "__main__":

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize])

    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size, args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
                 (args.arch, args.epoch, args.batch_size, args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = BinaryDiceLoss()

    optimizer = optim.Adam(model.parameters(), weight_decay=1e-8, lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    if 'train' in args.action:
        train(model, criterion1, criterion2, optimizer, train_dataloaders, val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)

    print("end")