import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist


def is_main_process():
    return dist.get_rank() == 0


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]

    
def build_optimizer(config, model):
     
    vis_params = []
    backbone_params = []
    text_params = []
    vis_head_params = []
    vis_attention_params = []
    for pname, p in model.named_parameters():
        if any(k in pname for k in ['vision_']): 
                if "head" in pname:
                    p.requires_grad = True
                    vis_head_params +=[p]
                elif "proj" in pname:
                    p.requires_grad = True
                    vis_attention_params +=[p]
                else:
                    p.requires_grad = True
                    vis_params +=[p]
        elif any(k in pname for k in ['text_']): 
            if 'text_encoder' not in pname:
                p.requires_grad = True
                text_params +=[p]
            else:
                p.requires_grad = True
                backbone_params +=[p]
        else:
            p.requires_grad = True
            backbone_params +=[p]
    
    params_group = [
        {'params': vis_params, 'lr': config.TRAIN.LR_ADAPTER },
        {'params': text_params, 'lr': config.TRAIN.LR_ADAPTER },
        {'params': vis_head_params, 'lr': config.TRAIN.LR_HEAD },
        {'params': vis_attention_params, 'lr': config.TRAIN.LR_ADAPTER},
        {'params': backbone_params, 'lr': config.TRAIN.LR }
    ]

    model = model.module if hasattr(model, 'module') else model

    optimizer = optim.AdamW(params_group,
                            weight_decay=config.TRAIN.WEIGHT_DECAY,
                            betas=(0.9, 0.98), eps=1e-6, )

    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.LR / 100,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler
