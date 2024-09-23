from .DETR_GAT import build as build_detr_gat
from .DETR_CLIP import build as build_detr_clip
from .DETR_Elmo import build as build_detr_elmo


def build_model(args):
    if args.model == 'detr_gat':
        return build_detr_gat(args)
    elif args.model == 'detr_clip':
        return build_detr_clip(args)
    elif args.model == 'detr_elmo':
        return build_detr_elmo(args)
    else:
        raise
