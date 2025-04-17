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
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--model_name', type=str, default='biomedclip_local',)
    parser.add_argument('--pretrain', type=str, default='CLIP/ckpt/open_clip_pytorch_model.bin')
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='/root/data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=100, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3,6,9,12], help="features used")
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
    model = CLIP_Inplanted(clip_model=biomedclip_model, obj=args.obj, tokenizer=tokenizer, features=args.features_list).to(device)
    model.eval()
    
    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    ctx_optimizer = torch.optim.Adam([model.prompt_learner.ctx], lr=args.learning_rate, betas=(0.5, 0.999))

    # load dataset and loader
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}
    print(args.obj)
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_mse = torch.nn.MSELoss()

    save_score = 0.0

    for epoch in range(args.epoch):
        print('epoch', epoch, ':')

        loss_list = []
        idx = 0
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            if idx % (len(train_loader) // 5) == 0:
                score = test(args, model, test_loader)
                if score >= save_score:
                    save_score = score
                    ckp_path = f'./ckpt/zero-shot/{args.obj}_coop.pth'
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapters': model.det_adapters.state_dict(),
                                'prompt_learner': model.prompt_learner.state_dict()}, 
                                ckp_path)
                    print(f'best epoch found: epoch {epoch} batch {idx}')
                print('\n')
            idx += 1

            image = image.squeeze(0).to(device)
            seg_idx = seg_idx[0].item()

            with torch.cuda.amp.autocast():
                image_features, text_features, seg_patch_tokens, det_patch_tokens, logits = model(image)
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # image level
                det_loss = 0
                image_label = image_label.squeeze(0).to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features.t()).unsqueeze(0)    
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                # coop loss
                loss_ce = F.cross_entropy(logits, image_label.long())
                
                # Now calculate the frozen pre-trained features
                fixed_embeddings = model.prompt_learner.fixed_embeddings
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
                emb_mask = torch.abs((z - torch.mean(z))/torch.std(z)) <= tau
                scores = torch.masked_select(torch.tensor(scores),emb_mask)
                scores = torch.tensor(scores).unsqueeze(1).unsqueeze(1).cuda()
                selected_embeddings = fixed_embeddings[:,emb_mask].mean(dim=1)
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

                if seg_idx > 0:
                    # pixel level
                    seg_loss = 0
                    mask = mask.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features.t()).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    loss = seg_loss + det_loss + loss_ce + loss_sccm + loss_kdsp
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    ctx_optimizer.zero_grad()
                    loss.backward()
                    ctx_optimizer.step()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    loss = det_loss + loss_ce + loss_sccm + loss_kdsp
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    ctx_optimizer.zero_grad()
                    loss.backward()
                    ctx_optimizer.step()
                    det_optimizer.step()

                loss_list.append(loss.item())

        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # logs
        print("Loss: ", np.mean(loss_list))

def test(args, model, test_loader):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    logits_list = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, text_features, seg_patch_tokens, det_patch_tokens, logits = model(image)
            seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
            
            # image
            anomaly_score = 0
            patch_tokens = det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features.t())
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
            anomaly_score += anomaly_map.mean(dim = 1 )
            image_scores.append(anomaly_score.cpu().numpy())

            # pixel
            patch_tokens = seg_patch_tokens
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
            
            gt_mask_list.append(mask.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

            logits = torch.softmax(logits, dim=1)[:,1]
            logits_list.extend(logits.cpu().detach().numpy())
        
    #        
    gt_list = np.array(gt_list)

    gt_mask_list = np.concatenate(gt_mask_list,axis=0)
    gt_mask_list = np.asarray(gt_mask_list)
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