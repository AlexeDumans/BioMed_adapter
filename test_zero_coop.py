import os
import argparse
import random
import math
import numpy as np

import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
    
from CLIP.adaptercoop import CLIP_Inplanted
from CLIP.clip import create_model

from loss import FocalLoss, BinaryDiceLoss

from dataset.medical_zero import MedTestDataset, MedTrainDataset
from dataset.medical_few import MedDataset

from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

import warnings
warnings.filterwarnings("ignore")


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--model_name', type=str, default='biomedclip_local',)
    parser.add_argument('--pretrain', type=str, default='CLIP/ckpt/open_clip_pytorch_model.bin')
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='/root/data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save_path', type=str, default='./ckpt/zero-shot/')
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12], help="features used")
    parser.add_argument('--seed', type=int, default=111)
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
    model = CLIP_Inplanted(clip_model=biomedclip_model, obj=args.obj,tokenizer=tokenizer,features=args.features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}_coop.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    model.prompt_learner.load_state_dict(checkpoint["prompt_learner"])

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    score = test(args, model, test_loader)
        


def test(args, seg_model, test_loader):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    logits_list = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, text_features, ori_seg_patch_tokens, ori_det_patch_tokens, logits = seg_model(image)
            ori_seg_patch_tokens = [p[:, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[:, 1:, :] for p in ori_det_patch_tokens]
            
            # image
            anomaly_score = 0
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features.t())
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean(dim=1)
            image_scores.append(anomaly_score.cpu().numpy())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features.t())
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1:2, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)
            
            gt_mask_list.extend(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)
            
            logits = torch.softmax(logits, dim=1)[:,1]
            logits_list.extend(logits.cpu().detach().numpy())
        
        
    # 
    gt_list = np.array(gt_list)
    
    gt_mask_list = np.concatenate(gt_mask_list,axis=0)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    # 
    logits_list = np.array(logits_list)
    segment_scores = np.concatenate(segment_scores)
    image_scores = np.concatenate(image_scores)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    logits_roc_auc = roc_auc_score(gt_list, logits_list)
    print(f'{args.obj} logits AUC : {round(logits_roc_auc,4)}')
    
    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det

if __name__ == '__main__':
    main()


