import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import json

import numpy as np
import h5py
import numpy
import re

_tokenizer = _Tokenizer()




def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y


def BDCovpool(x):
    batchSize, dim, M = x.data.shape
    x = x.reshape(batchSize, dim, M)
    if torch.isnan(x).any().item():
         print("input is nan")
    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype) 
    x_pow2 = x.bmm(x.transpose(1, 2) * 1./ (2*M)) 
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    if torch.isnan(dcov).any().item():
             print("dcov is nan")
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.sqrt(dcov + 1e-5)
    d1 = dcov.bmm(I_M * 1./ dim)
    d2 = (I_M * 1./ dim ).bmm(dcov)
    d3 = (I_M * 1./ dim ).bmm(dcov).bmm(I_M * 1./ dim)
    out = dcov - d1 - d2 + d3
    return out


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ViFi_CLIP',
                      "vision_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT,
                      "vision_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_VISION,
                      "num_frames": cfg.DATA.NUM_FRAMES,
                      "language_ctx": cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, is_mask=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        return x



class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        if cfg.TEST.TEMPLATE == 'LLM':
            self.LLM_prompt = True
        else:
            self.LLM_prompt = False
        self.json_path=cfg.TEST.LLM_JSON
        self.json_path_tc_clip=cfg.TEST.LLM_JSON_TC_CLIP
        dtype = clip_model.dtype
        self.use_prompt_stage = cfg.TRAINER.ViFi_CLIP.PROMPT_MODEL
        ctx_init = cfg.TRAINER.ViFi_CLIP.CTX_INIT
        ZS_evaluation = cfg.TRAINER.ViFi_CLIP.ZS_EVAL
        if ZS_evaluation:
            text_aug = f"{{}}"
            tokenized_prompts = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for c in classnames])
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype).cuda()
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.use_prompt_stage:
            n_cls = len(classnames)
            # Make sure Language depth >= 1
            assert cfg.TRAINER.ViFi_CLIP.PROMPT_DEPTH_TEXT >= 1, "In VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
            n_ctx = cfg.TRAINER.ViFi_CLIP.N_CTX_TEXT
            ctx_dim = clip_model.ln_final.weight.shape[0]

            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = n_ctx
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            logger.info(f"V-L design")
            logger.info(f'Initial text context: "{prompt_prefix}"')
            logger.info(f"Number of context words (tokens) for Language prompting: {n_ctx}")
            logger.info(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.ViFi_CLIP.N_CTX_VISION}")
            self.ctx = nn.Parameter(ctx_vectors)

            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
            self.n_cls = n_cls
            self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        elif self.LLM_prompt:
            prompt_list = []
            with open(self.json_path, 'r', encoding='utf-8') as json_file:
               data = json.load(json_file)
            for name in classnames:
                prompts = data[name]  
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompt_list.append(tokenized_prompts)
                            
            tokenized_prompts = torch.stack(prompt_list, dim=0).squeeze()  #[400, 10 , 77]
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)
            self.tokenized_prompts = tokenized_prompts 
        else:
            # # No prompting
            # ctx_init = ctx_init.replace("_", " ")
            # prompt_prefix = ctx_init
            # prompts = [prompt_prefix + " " + name + "." for name in classnames]
            # tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            # self.register_buffer("complete_text_embeddings", embedding)
            # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
            ctx_init = ctx_init.replace("_", " ")
            self.prompt_prefix = ctx_init
            with open(self.json_path, 'r', encoding='utf-8') as json_file:
               data = json.load(json_file)
               data_keys = list(data.keys())
            print(data_keys)
            prompts = [self.prompt_prefix + " " + name + "." for name in data_keys]
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.register_buffer("complete_text_embeddings", embedding)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        if self.use_prompt_stage:
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
        else:
            prompts = self.complete_text_embeddings

        return prompts
    
def Get_weight_features(global_token, features_all, is_mask=False):
    global_token = global_token.view(global_token.size()[0], 1, global_token.size()[-1]).repeat(1,  features_all.size()[1], 1)
    weight = (global_token * features_all).sum(dim=-1) * (global_token.size()[-1] ** -0.5)  
    weight_softmax = F.softmax(weight, dim=-1) 
    weight_softmax = weight_softmax.view(*weight.size()[:], 1)
    weight_features = weight_softmax * features_all

    return weight_features


class Get_BDC_Representation(nn.Module):
    def __init__(self, ):
        super().__init__()

    def BDCovpool(self, x):
        batchSize, dim, M = x.data.shape
        x = x.reshape(batchSize, dim, M)
        if torch.isnan(x).any().item():
            print("input is nan")
        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype) 
        x_pow2 = x.bmm(x.transpose(1, 2) * 1./ (2*M)) 
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        if torch.isnan(dcov).any().item():
                print("dcov is nan")
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.sqrt(dcov + 1e-5)
        d1 = dcov.bmm(I_M * 1./ dim)
        d2 = (I_M * 1./ dim ).bmm(dcov)
        d3 = (I_M * 1./ dim ).bmm(dcov).bmm(I_M * 1./ dim)
        out = dcov - d1 - d2 + d3
        return out
    
    def Triuvec(self, x):
        batchSize, dim, dim = x.shape
        r = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero(as_tuple = False)
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
        y = r[:, index].squeeze()
        return y
    
    def Get_weight_features(self, global_token, features_all):
        global_token = global_token.view(global_token.size()[0], 1, global_token.size()[-1]).repeat(1,  features_all.size()[1], 1)
        weight = (global_token * features_all).sum(dim=-1) * (global_token.size()[-1] ** -0.5)  
        weight_softmax = F.softmax(weight, dim=-1) 
        weight_softmax = weight_softmax.view(*weight.size()[:], 1)
        weight_features = weight_softmax * features_all
        return weight_features

    def forward(self, x_global, x):
        x_weighted = self.Get_weight_features(x_global, x)
        x_bdc = self.BDCovpool(x_weighted.transpose(1, 2))
        x_bdc = self.Triuvec(x_bdc) 
        return x_bdc
    
class BDC_CLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vis_bdc_reduce = nn.Linear(cfg.MODEL.DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.text_bdc_reduce = nn.Linear(cfg.MODEL.DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.dropout = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        self.mask_text_token = False
        self.LayerNorm_vis_bdc = nn.LayerNorm(cfg.MODEL.DIM)
        self.LayerNorm_text_bdc = nn.LayerNorm(cfg.MODEL.DIM)
        self.cg = cfg
        self.fr = cfg.DATA.NUM_FRAMES
        self.logits_bdc_text_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if cfg.TRAIN.IS_PRETRAIN:
            # self.text_bdc_head = nn.Linear(int( cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM+1)  /2 ), cfg.DATA.NUM_CLASSES)
            self.vis_bdc_head = nn.Linear(int( cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM+1)  /2 ), cfg.DATA.NUM_CLASSES)
        else:
            # self.text_bdc_head_downstream = nn.Linear(int( cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM+1)  /2 ), cfg.DATA.NUM_CLASSES)
            self.vis_bdc_head_downstream = nn.Linear(int( cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM+1)  /2 ), cfg.DATA.NUM_CLASSES)
        
        self.vis_bdc_Linear_Q = nn.Linear(cfg.TRAIN.REDUCE_DIM , cfg.TRAIN.REDUCE_DIM , bias=False)
        self.vis_bdc_Linear_K = nn.Linear(cfg.TRAIN.REDUCE_DIM , cfg.TRAIN.REDUCE_DIM , bias=False)
        self.Get_BDC = Get_BDC_Representation()

        if cfg.TEST.TEMPLATE == 'LLM':
            self.LLM_prompt = True
        else:
            self.LLM_prompt = False

    def Text_BDC_Adapter(self, eos_positions, text_features):
        text_features_reduce = self.text_bdc_reduce(self.LayerNorm_text_bdc(text_features))
        text_eos_features_reduce = text_features_reduce[torch.arange(text_features_reduce.shape[0]), eos_positions]
        text_bdc = self.Get_BDC(text_eos_features_reduce, text_features_reduce)
        return text_bdc
    
    def Video_BDC_Adapter(self, image_features):
        bt = image_features.size()[0]
        b = bt // self.fr
        image_features_reduce = self.vis_bdc_reduce(self.LayerNorm_vis_bdc(image_features)) #[bt, p+1, 192], p: number of patch tokens
        class_token_reduce = image_features_reduce[:, 0, :]  #[bt, 192]
        vis_bdc = self.Get_BDC(class_token_reduce, image_features_reduce)
        class_token_reduce = class_token_reduce.view(b, self.fr, -1)
        q, k, v = self.vis_bdc_Linear_Q(class_token_reduce), self.vis_bdc_Linear_K(class_token_reduce), vis_bdc.view(b, self.fr, -1)
        attn = (q @ k.transpose(-2, -1)) * (q.size()[-1] ** -0.5) # [bs, 15, 15]
        attn = attn.softmax(dim=-1)
        x_attn = (attn @ v)
        video_bdc = x_attn + v

        return video_bdc

    def forward(self, image):

        logit_scale = self.logit_scale.exp()
        logit_scale_bdc = self.logits_bdc_text_scale.exp()
        if self.LLM_prompt:
            prompts = self.prompt_learner() 
            tokenized_prompts = self.tokenized_prompts
            if not self.training:
                if self.cg.TEST.ENSEMBLE:
                    text_eos_list = []
                    text_bdc = []
                    for i in range(prompts.size()[1]):
                        tokenized_prompts_ = tokenized_prompts[:, i].squeeze(1)
                        prompts_ = prompts[:, i].squeeze(1)
                        with torch.no_grad():
                            text_features = self.text_encoder(prompts_, tokenized_prompts_)
                            text_features_reduce = self.text_bdc_reduce(self.LayerNorm_text_bdc(text_features))
                            eos_positions = tokenized_prompts_.argmax(dim=-1).cuda()
                            text_eos_features_reduce = text_features_reduce[torch.arange(text_features_reduce.shape[0]), eos_positions]
                            weight_text_features = Get_weight_features(text_eos_features_reduce, text_features_reduce)
                            text_bdc_i = BDCovpool(weight_text_features.transpose(1, 2))
                            text_bdc_i = Triuvec(text_bdc_i) 
                            text_bdc_i /= text_bdc_i.norm(dim=-1, keepdim=True) 
                            
                            text_eos_features_ = text_features[torch.arange(text_features.shape[0]), tokenized_prompts_.argmax(dim=-1)]
                            text_eos_features_ /= text_eos_features_.norm(dim=-1, keepdim=True) 
                            
                            text_bdc.append(text_bdc_i)
                            text_eos_list.append(text_eos_features_)
                    text_eos_features = torch.stack(text_eos_list, dim=1)
                    # text_eos_features = text_eos_features.mean(1) 
                    text_eos_features = text_eos_features[:, 1:].mean(1) + text_eos_features[:, 0] 
                    text_bdc = torch.stack(text_bdc, dim=1)
                    # text_bdc = text_bdc.mean(1)
                    text_bdc = text_bdc[:, 1:].mean(1) + text_bdc[:, 0] 
                else:
                    tokenized_prompts_ = tokenized_prompts[:, 0]
                    prompts_ = prompts[:, 0]         
                    with torch.no_grad():
                        text_features = self.text_encoder(prompts_, tokenized_prompts_)
                        text_eos_features = text_features[torch.arange(text_features.shape[0]), tokenized_prompts_.argmax(dim=-1)]
                        text_features_reduce = self.text_bdc_reduce(self.LayerNorm_text_bdc(text_features))
                    text_eos_features_reduce = text_features_reduce[torch.arange(text_features_reduce.shape[0]), tokenized_prompts_.argmax(dim=-1)]
                    weight_text_features = Get_weight_features(text_eos_features_reduce, text_features_reduce)
                    text_bdc = BDCovpool(weight_text_features.transpose(1, 2))
                    text_bdc = Triuvec(text_bdc) 
            else:
                random_indices = torch.randint(0, prompts.size()[1], (prompts.size()[0],)) 
                tokenized_prompts_ = tokenized_prompts[torch.arange(prompts.size()[0]), random_indices]
                prompts_ = prompts[torch.arange(prompts.size()[0]), random_indices]
                text_features = self.text_encoder(prompts_, tokenized_prompts_)
                eos_positions = tokenized_prompts_.argmax(dim=-1).cuda()
                text_eos_features = text_features[torch.arange(text_features.shape[0]), eos_positions] # get eos token
                text_bdc = self.Text_BDC_Adapter(eos_positions, text_features)

        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)
        image_features = self.image_encoder(image.type(self.dtype))
        vis_bdc = self.Video_BDC_Adapter(image_features)
        vis_bdc = vis_bdc.mean(1) 

        if self.cg.TRAIN.IS_PRETRAIN:
            vis_bdc_logits = self.vis_bdc_head(self.dropout(vis_bdc)) 
        else:
            vis_bdc_logits = self.vis_bdc_head_downstream(self.dropout(vis_bdc)) 
 

        vis_bdc = vis_bdc / vis_bdc.norm(dim=-1, keepdim=True)
        text_bdc = text_bdc / text_bdc.norm(dim=-1, keepdim=True)
        text_bdc_logits = logit_scale_bdc * vis_bdc @ text_bdc.t()

        # text2vision
        class_token = image_features[:, 0, :].view(b, t, -1) 
        image_cls_features = class_token.mean(dim=1, keepdim=False)  # image features are now ready
        image_cls_features = image_cls_features / image_cls_features.norm(dim=-1, keepdim=True)
        text_eos_features = text_eos_features / text_eos_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_cls_features @ text_eos_features.t()

        return logits, vis_bdc_logits, text_bdc_logits



def returnCLIP(config, logger=None,
               class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    logger.info("Building BDC-CLIP CLIP")
    model = BDC_CLIP(config, class_names, clip_model, logger)

    if config.TRAINER.ViFi_CLIP.PROMPT_MODEL:
        logger.info("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
    else:
        # Now need to control freezing of CLIP for fine-tuning
        train_complete_clip = config.TRAINER.ViFi_CLIP.USE
        if train_complete_clip == "both":
            logger.info("Turning on gradients for COMPLETE ViFi-CLIP model")
            for name, param in model.named_parameters():
                param.requires_grad_(True)
        else:
            if train_complete_clip == "image":
                logger.info("Turning on gradients for image side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "image_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
            else:
                logger.info("Turning on gradients for TEXT side the ViFi-CLIP model")
                for name, param in model.named_parameters():
                    if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    # logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model
