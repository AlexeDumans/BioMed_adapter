import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from prompt import BIOMEDCOOP_TEMPLATES
import copy

CLASSNAMES = {'Brain': [ "normal brain","glioma tumor"], 'Liver':["normal liver" , "liver tumor"], 'Retina_RESC': ['normal retina', 'retinal edema'], 'Chest': ['normal chest x-ray' , 'thoracic abnormlaity'], 'Retina_OCT2017':['normal retina', 'retinal pathology'], 'Histopathology': ['normal tissue', 'metastatic tumor'] }

# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class PromptLearner(nn.Module):
    def __init__(self, tokenizer,obj, biomedclip_model):
        super().__init__()
        n_cls = 2 # len(classnames)
        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = 224
        
        self.biomedclip_model = copy.deepcopy(biomedclip_model)
        self.tokenizer = tokenizer
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx==4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init).cuda()
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if False:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print("Context vectors shape: ", ctx_vectors.shape)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 多模板的异常分类
        prompt_normal = [ 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect']
        prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
        prompt_state = [prompt_normal, prompt_abnormal]
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                prompted_sentence.append(ctx_init + " " + s)
            # prompted_sentence = tokenize(prompted_sentence).to(device)
            prompted_sentence = self.tokenizer(prompted_sentence)
            prompted_sentence = torch.tensor(prompted_sentence).cuda()
            # prompted_sentence = prompted_sentence.mean(dim=0)
            text_features.append(prompted_sentence)
        tokenized_prompts = torch.stack(text_features, dim=0).cuda() # (n_cls, n_tkn)

        # tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        biomedclip_model_temp = self.biomedclip_model.float().eval().cuda()
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            embedding = embedding.mean(dim=1)# (n_cls, 1, dim)
            self.ZS_image_encoder = biomedclip_model_temp.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []

            for i in range(50):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDCOOP_TEMPLATES[classname][i]) for classname in CLASSNAMES[obj]])

                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = 0  # name_lens
        self.class_token_position = "end"

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts,tokenized_prompts,normalize: bool = False):
        x = prompts.cuda()
        attn_mask = (x != 0 ).long() # self.config.pad_token_id 0 
        
        seq_length = prompts.size()[1]
        position_ids = torch.arange(512).expand((1, -1)) # max_position_embeddings 512
        position_ids = position_ids[:,  0 : seq_length + 0].cuda() # past_key_values_length 0
        
        token_type_ids = torch.zeros(position_ids.size(), dtype=torch.long).cuda()
        buffered_token_type_ids = token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(x.size()[0], seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(x)
    
        
        # x = prompts + self.model.text.transformer.embeddings.position_embeddings.weight.type(self.dtype)
        position_embeddings = self.model.text.transformer.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.model.text.transformer.embeddings.token_type_embeddings(token_type_ids)
        
        x = x + token_type_embeddings
        x += position_embeddings.type(self.dtype)
        x = self.model.text.transformer.embeddings.LayerNorm(x)
        x = self.model.text.transformer.embeddings.dropout(x)
        
        x = self.model.text.transformer.encoder(x)
        # sequence_output = x[0]
        # pooled_output =  self.model.text.transformer.pooler(sequence_output)
        # x = (sequence_output,pooled_output) + x[1:]
        
        x = self.model.text.pooler(x,attn_mask)
        x = self.model.text.proj(x)
        # x = self.model.encode_text(prompts,True,token_prompt)

        return F.normalize(x,dim=-1) if normalize else x 

        
class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, obj,tokenizer,features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=512) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(768, bottleneck=512) for i in range(len(features))] )
        
        # coop
        self.prompt_learner = PromptLearner(tokenizer,obj, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
    
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.text.transformer.dtype

    def forward(self, x):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts,tokenized_prompts)
        
        
        # Image - -
        x = self.image_encoder.trunk.patch_embed(x)
        
        cls_token = self.image_encoder.trunk.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.image_encoder.trunk.pos_embed

        seg_patch_tokens = []
        det_patch_tokens = []

        for i, block in enumerate(self.image_encoder.trunk.blocks):
            x = block(x)
            if (i + 1) in self.features:
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)
                
                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out
                
                if i == 0:
                    seg_patch_tokens = [seg_adapt_med]
                    det_patch_tokens = [det_adapt_med]
                else:
                    seg_patch_tokens.append(seg_adapt_med)
                    det_patch_tokens.append(det_adapt_med)

        x = self.image_encoder.trunk.norm(x)
        
        image_features = x[:, 0]
        
        if hasattr(self.image_encoder, 'head'):
            image_features = self.image_encoder.head.proj(image_features)
            
        image_features_norm = image_features / image_features.norm(dim=-1,keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1,keepdim=True)
        logis = logit_scale * image_features_norm @ text_features_norm.t()

        return image_features,text_features, seg_patch_tokens, det_patch_tokens , logis




