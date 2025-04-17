import os
import argparse
import random
import math
import numpy as np
import time

import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
    
from CLIP.adaptercoop import CLIP_Inplanted
from CLIP.clip import create_model

from loss import FocalLoss, BinaryDiceLoss

from dataset.medical_zero import MedTestDataset, MedTrainDataset
from dataset.medical_few import MedDataset

from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

import warnings
warnings.filterwarnings("ignore")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='biomedclip_local',)
    parser.add_argument('--pretrain', type=str, default='CLIP/ckpt/open_clip_pytorch_model.bin')
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='/root/data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3,6,9,12], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)
    
    # fixed feature extractor
    biomedclip_model,tokenizer = create_model(model_name=args.model_name, 
                              force_image_size=args.img_size, 
                              device=device, 
                              pretrained=args.pretrain, 
                              require_pretrained=True)
    
    biomedclip_model.eval()

    # 模型添加适配器
    model = CLIP_Inplanted(clip_model=biomedclip_model, obj = args.obj,tokenizer=tokenizer,features=args.features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}_coop.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    model.prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    for name, param in model.named_parameters():
        param.requires_grad = True


    # load test dataset
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    seg_features = []
    det_features = []
    for image in support_loader:
        image = image[0].to(device)
        with torch.no_grad():
            _, _, seg_patch_tokens, det_patch_tokens, _ = model(image)
            seg_patch_tokens = [p.contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p.contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)
    seg_mem_features = [torch.cat([seg_features[j][i].view(-1,seg_features[j][i].shape[-2],seg_features[j][i].shape[-1]) for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i].view(-1,det_features[j][i].shape[-2],det_features[j][i].shape[-1]) for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]       
    # seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    # det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
    
    test(args, model, test_loader, seg_mem_features, det_mem_features)


def test(args, model, test_loader, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []
    logits_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []
    seg_logit_map_few= []
        

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, text_features, seg_patch_tokens, det_patch_tokens, logits = model(image)
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:

                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    batch_cos_sim = []
                    for b in range(p.shape[0]):
                        cos = cos_sim(seg_mem_features[idx][b], p[b])
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                        batch_cos_sim.append(anomaly_map_few_shot[0].cpu().numpy())
                        # print('batch_cos_sim.shape:', batch_cos_sim[0].shape)
                    anomaly_maps_few_shot.append(np.stack(batch_cos_sim, axis=0))
                    # print('anomaly_maps_few_shot.shape:', anomaly_maps_few_shot[0].shape)
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                logit_score_few = score_map_few.mean(axis=(1,2,3))
                seg_score_map_few.append(score_map_few)
                seg_logit_map_few.append(logit_score_few)


                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    # * 多batch 取消 .squenze(0)
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features.t())
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True)
                    # * 1: -> 1:2 维持中间维度
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1:2, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.extend(score_map_zero)
                
            else:
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    batch_cos_sim = []
                    for b in range(p.shape[0]):
                        cos = cos_sim(seg_mem_features[idx][b], p[b])
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                            size=args.img_size, mode='bilinear', align_corners=True)
                        batch_cos_sim.append(anomaly_map_few_shot[0].cpu().numpy())
                        # print('batch_cos_sim.shape:', batch_cos_sim[0].shape)
                    anomaly_maps_few_shot.append(np.stack(batch_cos_sim, axis=0))
                    # print('anomaly_maps_few_shot.shape:', anomaly_maps_few_shot[0].shape)
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean(axis=(1,2,3))
                det_image_scores_few.extend(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features.t())
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean(dim=1)
                det_image_scores_zero.extend(anomaly_score.cpu().numpy())

            gt_mask_list.extend(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            
            logits = torch.softmax(logits, dim=1)[:,1]
            logits_list.extend(logits.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    logits_list = np.array(logits_list)
    
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = [seg_score_map_zero[j][i] if len(seg_score_map_zero[j].shape) > 2 else seg_score_map_zero[j]
            for j in range(len(seg_score_map_zero))        # 先遍历元素索引 j
            for i in range(seg_score_map_zero[j].shape[0])  # 再遍历元素内部的样本索引 i
            ]
        seg_score_map_few = [seg_score_map_few[j][i] if len(seg_score_map_few[j].shape) > 2 else seg_score_map_few[j]
            for j in range(len(seg_score_map_few))        # 先遍历元素索引 j
            for i in range(seg_score_map_few[j].shape[0])  # 再遍历元素内部的样本索引 i
            ]

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)
        
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        
        seg_logit_map_few = np.concatenate(seg_logit_map_few)
        seg_logit_map_few = np.array(seg_logit_map_few)
        seg_logit_map_few = (seg_logit_map_few - seg_logit_map_few.min()) / (seg_logit_map_few.max() - seg_logit_map_few.min())
        logits_list = 0.5 * seg_logit_map_few + 0.5 * logits_list
        roc_auc_im = roc_auc_score(gt_list, logits_list)
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:
        # * 多batch展平
        det_image_scores_zero = np.concatenate(det_image_scores_zero)
        det_image_scores_few = np.concatenate(det_image_scores_few)
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det





if __name__ == '__main__':
    main()


