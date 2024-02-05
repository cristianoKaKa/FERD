import os
import random
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets.HADDatasets import HADDataset, HADTestDataset
from models.test.resnet import wide_resnet101_2 as saved_encoder
from models.resnet import wide_resnet101_2, ConvH, Pixel_Classifier
from models.de_resnet import de_wide_resnet50_2, de_ConvH
from utils.utils import time_string, convert_secs2time, AverageMeter, print_log, save_checkpoint, write_eval_result, write_mean_result, seed_torch
from utils.RX import RX
from losses.losses import CosLoss
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser(description='hyperspectral anomaly detection')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--data_path', type=str, default='./data/HAD100Dataset/')
    parser.add_argument('--start_channel_id', type=int, default=0, help='the start id of spectral channel')
    parser.add_argument('--input_channel', type=int, default=50, help='the spectral channel number of input HSI')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60, help='the maximum of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of Adam')
    parser.add_argument('--seed', type=int, default=10, help='manual seed')
    parser.add_argument('--train_ratio', type=float, default=1, help='data ratio used for training')
    parser.add_argument('--sensor', type=str, default='aviris_ng',help='sensor used in training,  aviris_ng or aviris  test')
    parser.add_argument('--save_txt', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    
    # manual seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_torch(seed=args.seed)

    # build save path        
    args.save_dir = os.path.join('./result/', str(args.input_channel) + 'bands_'
                                 +str(args.sensor) + '_' + 'seed_' + str(args.seed))
    epoch_write_dir = os.path.join(args.save_dir, 'epoch')
    if not os.path.exists(epoch_write_dir):
        os.makedirs(epoch_write_dir)
    
    args.saved_models_dir = os.path.join('./result/', str(args.input_channel) + 'bands_'
                                 +str(args.sensor) + '_' + 'seed_' + str(args.seed) + '/saved_models')
    if not os.path.exists(args.saved_models_dir):
        os.makedirs(args.saved_models_dir)

    log = open(os.path.join(args.save_dir, 'training_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)


    # load model and dataset
    encoder, bn = wide_resnet101_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    convh = ConvH(input_channel=args.input_channel)
    classifier = Pixel_Classifier(input_channel=args.input_channel)
    deconvh = de_ConvH(input_channel=args.input_channel)
    saved_enc, _ = saved_encoder(pretrained=True)

    encoder.cuda(device=args.device_ids[0])
    bn.cuda(device=args.device_ids[0])
    decoder.cuda(device=args.device_ids[0])
    convh.cuda(device=args.device_ids[0])
    classifier.cuda(device=args.device_ids[0])
    deconvh.cuda(device=args.device_ids[0])
    saved_enc.cuda(device=args.device_ids[0])
    
    optimizer = torch.optim.Adam(list(classifier.parameters())+list(convh.parameters())+list(decoder.parameters())+list(bn.parameters())+list(deconvh.parameters()), lr=args.lr, betas=(0.5,0.999))

    # load dataset
    kwargs = {'num_workers':4, 'pin_memory': True}
    train_dataset = HADDataset(dataset_path=args.data_path,sensor= args.sensor, resize=args.img_size,
                                start_channel=args.start_channel_id, channel=args.input_channel, train_ratio = args.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size*len(args.device_ids), shuffle=True, **kwargs)
    test_dataset = HADTestDataset(dataset_path=args.data_path, resize=args.img_size, channel=args.input_channel)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1*len(args.device_ids), shuffle=False, **kwargs)

    # start training
    start_time = time.time()
    epoch_time = AverageMeter()

    best_auc = 0

    for epoch in range(1, args.epochs + 1):

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)

        losses = train(args, convh, classifier, encoder, bn, decoder, deconvh, epoch, train_loader, optimizer, log)
        print_log(('Train Epoch: {}  Loss: {:.8f} '.format(epoch,  losses.avg)), log)

        test_imgs, mixfs, scores, gt_imgs = test(args, classifier, convh, encoder, bn, decoder, test_loader)

        scores = np.asarray(scores)
        gt_imgs = np.asarray(gt_imgs)

        # calculate ROCAUC
        AU_ROC_per_img = np.zeros(len(test_imgs))
        for i in range(len(test_imgs)):
            AU_ROC_per_img[i] = roc_auc_score(gt_imgs[i, :].flatten() == 1,
                                                              scores[i, :].flatten())
        mean_AU_ROC = np.mean(AU_ROC_per_img)

        if best_auc < mean_AU_ROC:
            best_auc = mean_AU_ROC
            if args.save_txt:
                write_eval_result(os.path.join(args.save_dir, 'best.txt'), test_dataset.test_img,
                                  AU_ROC_per_img,list(range(len(test_dataset.test_img))))
                write_mean_result(os.path.join(args.save_dir, 'best_meanauc.txt'), best_auc)
                
            if args.save_model:
                convh_save_dir = os.path.join(args.saved_models_dir, 'convh.pt')
                enc_save_dir = os.path.join(args.saved_models_dir, 'enc.pt')
                pc_save_dir = os.path.join(args.saved_models_dir, 'pc.pt')
                save_checkpoint(convh_save_dir, convh)
                saved_enc.layer1 = encoder.layer1
                save_checkpoint(enc_save_dir, saved_enc)
                save_checkpoint(pc_save_dir, classifier)

        print_log('mean pixel ROCAUC: %.5f' % (mean_AU_ROC), log)

        if args.save_txt:
            write_eval_result(os.path.join(epoch_write_dir, 'epoch{:d}.txt'.format(epoch)),test_dataset.test_img,AU_ROC_per_img,
                              list(range(len(test_dataset.test_img))))


        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    log.close()


def train(args, convh, classifier, encoder, bn, decoder, deconvh, epoch, train_loader, optimizer, log):
    
    classifier.train()
    convh.train()
    encoder.eval()
    bn.train()
    decoder.train()
    deconvh.train()

   # model.train()
    losses = AverageMeter()
    
    for (data) in tqdm(train_loader):

        data = data.cuda(device=args.device_ids[0])
        labels = torch.zeros(data.size(0), data.size(2), data.size(3)).cuda(device=args.device_ids[0])

        f_abc_enc, fa_cut_enc = encoder(convh(data))
        input_dec = bn(f_abc_enc)
        output_dec, f_abc_dec, fc_cut_dec = decoder(input_dec, data)
        output_dec = deconvh(output_dec)
        pc, mixf = classifier(fa_cut_enc, fc_cut_dec)

        model_output = pc.view(-1)
        labels = labels.view(-1)
        BceLoss = nn.BCEWithLogitsLoss()
        MseLoss = nn.MSELoss(reduction='mean')

        loss = 0.1*BceLoss(model_output, labels) + 0.1*MseLoss(data, output_dec) + 1.0*CosLoss(f_abc_enc, f_abc_dec)
        losses.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

def test(args, classifier, convh, encoder, bn, decoder, test_loader):

    classifier.eval()
    convh.eval()
    encoder.eval()
    bn.eval()
    decoder.eval()

    scores = []
    test_imgs = []
    gt_imgs = []
    mixfs = []

    for (data, gt) in tqdm(test_loader):

        test_imgs.extend(data.cpu().numpy())
        gt_imgs.extend(gt.cpu().numpy())

        with torch.no_grad():
            data = data.cuda(device=args.device_ids[0])

            _, fa_cut_enc = encoder(convh(data))
            _, mixf = classifier(fa_cut_enc, fa_cut_enc)

            score = np.zeros([data.shape[0],data.shape[-1],data.shape[-1]])
            for i in range(data.shape[0]):
                score1 = RX(mixf+data)
                score[i, :] = score1

        if len(score.shape) == 2:
            score=np.expand_dims(score,axis=0)
        scores.extend(score)
        mixfs.extend(mixf.cpu().numpy())

    return test_imgs, mixfs, scores, gt_imgs


if __name__ == '__main__':

    main()
