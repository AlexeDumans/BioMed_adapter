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
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}

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
    parser.add_argument("--epoch", type=int, default=100, help="epochs")
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
    model = CLIP_Inplanted(clip_model=biomedclip_model, args.obj,tokenizer=tokenizer, features=args.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True

        # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    ctx_optimizer = torch.optim.Adam([model.prompt_learner.ctx], lr=args.learning_rate, betas=(0.5, 0.999))

    # load test dataset
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_mse = torch.nn.MSELoss()

    # text prompt
    # with torch.cuda.amp.autocast(), torch.no_grad():
    #     text_features = encode_text_with_prompt_ensemble(biomedclip_model, tokenizer, REAL_NAME[args.obj], device)


    best_result = 0

    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')

        loss_list = []
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                image_features,text_features, seg_patch_tokens, det_patch_tokens,logits = model(image)
                # seg_patch_tokens size { [batch_size,196,512] * 4} 
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

                # det loss
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features.t())    
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)
                
                loss_ce = F.cross_entropy(logits,label)
                
                # Now calculate the frozen pre-trained features
                fixed_embeddings =  model.prompt_learner.fixed_embeddings # precomputed pre-trained frozen textual features
                fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

                zero_shot_features = model.prompt_learner.ZS_image_encoder(image)
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)

                scores = []
                for i in range(fixed_embeddings.shape[1]):
                    temp_logits = model.logit_scale * image_features @ fixed_embeddings[:,i,:].cuda().t()
                    max_logits = torch.max(temp_logits, dim=1).values
                    sp = torch.mean(max_logits)
                    scores.append(sp.item())
                
                s_bar = torch.median(torch.tensor(scores))
                d_bar = torch.median(torch.abs(torch.tensor(scores)-s_bar))
                z = (torch.tensor(scores) - s_bar) / d_bar
                tau = 1.5
                mask = torch.abs((z - torch.mean(z))/torch.std(z)) <= tau
                scores = torch.masked_select(torch.tensor(scores),mask)
                scores = torch.tensor(scores).unsqueeze(1).unsqueeze(1).cuda()
                selected_embeddings = fixed_embeddings[:,mask].mean(dim=1)
                selected_embeddings = selected_embeddings / selected_embeddings.norm(dim=-1, keepdim=True)
            

                fixed_embeddings = fixed_embeddings.mean(dim=1)
                fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
                zero_shot_logits = model.logit_scale * zero_shot_features.cuda() @ selected_embeddings.cuda().t()
                
                loss_sccm = loss_mse(text_features, fixed_embeddings.cuda()) * 1
                
                loss_kdsp = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    F.log_softmax(zero_shot_logits, dim=1),
                    reduction='sum',
                    log_target=True
                ) / logits.numel()
                loss_kdsp = loss_kdsp * 1
                
                

                if CLASS_INDEX[args.obj] > 0:
                    # pixel level
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features.t())
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    loss = seg_loss * 0.5 + det_loss * 0.5 + loss_bce + loss_sccm + loss_kdsp
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    ctx_optimizer.zero_grad()
                    loss.backward()
                    ctx_optimizer.step()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    loss = det_loss * 0.5 + loss_bce + loss_sccm + loss_kdsp
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    ctx_optimizer.zero_grad()
                    loss.backward()
                    ctx_optimizer.step()
                    det_optimizer.step()

                loss_list.append(loss.item())

        print("Loss: ", np.mean(loss_list))


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
        # * 多batch处理
        seg_mem_features = [torch.cat([seg_features[j][i].view(-1,seg_features[j][i].shape[-2],seg_features[j][i].shape[-1]) for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i].view(-1,det_features[j][i].shape[-2],det_features[j][i].shape[-1]) for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        # seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        # det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        

        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        if result > best_result:
            best_result = result
            print("Best result\n")
            if args.save_model == 1:
                ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, 
                            ckp_path)
          

def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, _, seg_patch_tokens, det_patch_tokens, _ = model(image)
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:

                # few-shot, seg head
                # * 添加多batch处理
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
                seg_score_map_few.append(score_map_few)


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
                seg_score_map_zero.append(score_map_zero)
                


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
                det_image_scores_few.append(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features.t())
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean(dim=1)
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            # ?  在batch_size = 4 , 对于Liver数据集来说，最后会有一张image被单独留下， 可能是 .squeeze() 配合之前的’展开代码‘ 的问题
            gt_mask_list.append(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            

    gt_list = np.array(gt_list)
    
    # * 多batch展平
    gt_mask_list = np.concatenate(gt_mask_list,axis=0)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:
        # * 多batch展平
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


