## Note: CLIP gives 768 dim embedding 
## Goal: To adapt this to UC + UA + UO setting 
# First run on UA, UC and then extend it to UO

from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F
from torch.random import set_rng_state

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math

from .backbone import build_backbone
from .matcher import build_matcher
from .DETR_base import build_detr_base

from datasets.static_hico import ACT_IDX_TO_ACT_NAME, HICO_INTERACTIONS, ACT_TO_ING, HOI_IDX_TO_ACT_IDX, \
    HOI_IDX_TO_OBJ_IDX, MAP_AO_TO_HOI, UA_HOI_IDX

import nncore
from nncore.nn import build_msg_pass_modules
from nncore.ops import cosine_similarity


# from .coop import CustomCLIP, load_clip_to_cpu

class HOIModel(nn.Module):
    """
    CLIP text features (obj + action) + Single stage Hungarian matching.
    """

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False,
                 args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim) ## trainable queries initialised randomly 
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1) ## FFN head for obj class pred
        # self.obj_class_embed = nn.Linear(hidden_dim, 768) ## FFN head for object class pred (changed from EoID to adopt to CLIP)
        # self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes) 
        self.verb_class_embed = nn.Linear(hidden_dim, 768) ## FFN head for verb class pred (changed from EoID to adopt to CLIP)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) ## FFN head for h_box (3 layer MLP: hidden_dim -> hidden_dim -> hidden_dim -> 4)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # FFN head for o_box (3 layer MLP: hidden_dim -> hidden_dim -> hidden_dim -> 4)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) ## That 1x1 convolution layer after passing input through CONV backbone to change num channels to match the hidden dim of transformer 
        self.backbone = backbone
        self.aux_loss = aux_loss # True 
        self.use_matching = args.use_matching # False 
        self.dec_layers_hopd = args.dec_layers_hopd ## 3 layers for HOPD decoder
        self.dec_layers_interaction = args.dec_layers_interaction # 3 layers for Int decoder
        if self.use_matching:
            self.matching_embed = nn.Linear(hidden_dim, 2)

        # self.CLIP_action_semantics = torch.from_numpy(np.load("./sent2vec_action_semantics.npy")).to(args.device) # (117, 768)
        self.CLIP_action_semantics = torch.from_numpy(np.load("./CLIP_action_semantics.npy")).to(args.device)
        # self.CLIP_object_semantics = torch.from_numpy(np.load("./CLIP_object_semantics_81.npy")).to(args.device) # (81, 768)
        # self.CLIP_interaction_semantics = torch.from_numpy(np.load("./CLIP_interaction_semantics.npy")).to(args.device) # (600, 768)
        
        self.score_factor_action = 10 # Find a proper value for it
        # self.score_factor_object = 10 # Find a proper value for it

    def forward(self, samples: NestedTensor):
        # print(samples.tensors.shape)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) # features is a list

        src, mask = features[-1].decompose()
        # print(mask)
        
        assert mask is not None
        hopd_out, interaction_decoder_out = self.transformer(self.input_proj(src), mask, self.query_embed.weight,
                                                             pos[-1])[:2]

        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid() # Why sigmoid?
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
        
        outputs_obj_class = self.obj_class_embed(hopd_out)
        # text_features_obj = self.CLIP_object_semantics.float()
        # text_features_obj = text_features_obj / text_features_obj.norm(dim=-1, keepdim=True) # 81x768
        # outputs_obj_class = outputs_obj_class @ text_features_obj.t() # taking the dot product
        # outputs_obj_class = outputs_obj_class * self.score_factor_object
        

        if self.use_matching:
            outputs_matching = self.matching_embed(hopd_out)

        outputs_verb_class = self.verb_class_embed(interaction_decoder_out)

        outputs_verb_class = outputs_verb_class / outputs_verb_class.norm(dim=-1, keepdim=True)
        text_features = self.CLIP_action_semantics
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # 117x768
        outputs_verb_class = outputs_verb_class @ text_features.t() # taking the dot product of the verb logits (from the transformer) with the text features of the actions from the GAT
        outputs_verb_class = outputs_verb_class * self.score_factor_action # multiplying with 10 to get the corresponding action scores

        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1]}
        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else: # This will be run
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord,
                      outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, \
                     'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in
                    zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1], \
                        outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1], \
                        outputs_matching[-min_dec_layers_num: -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in
                    zip(outputs_obj_class[-min_dec_layers_num: -1], outputs_verb_class[-min_dec_layers_num: -1], \
                        outputs_sub_coord[-min_dec_layers_num: -1], outputs_obj_coord[-min_dec_layers_num: -1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        # self.dym_bce_loss = DynamicBCELoss()
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher ## Two-stage matcher 
        self.weight_dict = weight_dict ## The lambdas for different losses (multi-task learning)
        self.eos_coef = eos_coef ## 0.1
        self.losses = losses ## losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight) ## store this in state_dict without being used by optimizer

        self.alpha = args.alpha

        self._unseen_idx = [HOI_IDX_TO_ACT_IDX[idx] for idx in UA_HOI_IDX]
        self.unseen_mask = torch.tensor([1. for _ in range(117)])
        self.unseen_mask[self._unseen_idx] = 0

        self.obj_reweight = args.obj_reweight # False
        self.verb_reweight = args.verb_reweight # False 
        self.use_static_weights = args.use_static_weights # False 

        Maxsize = args.queue_size

        # if self.obj_reweight:
        #     self.q_obj = Queue(maxsize=Maxsize)
        #     self.p_obj = args.p_obj
        #     self.obj_weights_init = self.cal_weights(self.obj_nums_init, p=self.p_obj)

        # if self.verb_reweight:
        #     self.q_verb = Queue(maxsize=Maxsize)
        #     self.p_verb = args.p_verb
        #     self.verb_weights_init = self.cal_weights(self.verb_nums_init, p=self.p_verb)

    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = [0] * (num_fgs + 1)
        num_all = sum(label_nums[:-1])

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all / label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight > 0])

        weight[-1] = np.power(num_all / label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # print(src_logits[target_classes!=self.num_obj_classes])

        if not self.obj_reweight:
            obj_weights = self.empty_weight
        elif self.use_static_weights:
            obj_weights = self.obj_weights_init
        else:
            obj_label_nums_in_batch = [0] * (self.num_obj_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    obj_label_nums_in_batch[label] += 1

            if self.q_obj.full(): self.q_obj.get()
            self.q_obj.put(np.array(obj_label_nums_in_batch))
            accumulated_obj_label_nums = np.sum(self.q_obj.queue, axis=0)
            obj_weights = self.cal_weights(accumulated_obj_label_nums, p=self.p_obj)

            aphal = min(math.pow(0.999, self.q_obj.qsize()), 0.9)
            obj_weights = aphal * self.obj_weights_init + (1 - aphal) * obj_weights

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)

        target_classes[idx] = target_classes_o

        if not self.verb_reweight:
            verb_weights = None
        elif self.use_static_weights:
            verb_weights = self.verb_weights_init
        else:
            verb_label_nums_in_batch = [0] * (self.num_verb_classes + 1)
            for target_class in target_classes:
                for label in target_class:
                    label_classes = torch.where(label > 0)[0]
                    if len(label_classes) == 0:
                        verb_label_nums_in_batch[-1] += 1
                    else:
                        for label_class in label_classes:
                            verb_label_nums_in_batch[label_class] += 1

            if self.q_verb.full(): self.q_verb.get()
            self.q_verb.put(np.array(verb_label_nums_in_batch))
            accumulated_verb_label_nums = np.sum(self.q_verb.queue, axis=0)
            verb_weights = self.cal_weights(accumulated_verb_label_nums, p=self.p_verb)

            aphal = min(math.pow(0.999, self.q_verb.qsize()), 0.9)
            verb_weights = aphal * self.verb_weights_init + (1 - aphal) * verb_weights

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, weights=verb_weights, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                        exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}

        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.eq(0).float()
        weight = self.unseen_mask.unsqueeze(0).unsqueeze(0).repeat(pred.shape[0], pred.shape[1], 1).to(pred.device)
        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * weight
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * weight

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'matching_labels': self.loss_matching_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices['indices'], num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob = obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[
                index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu'), 'obj_scores': os.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    detr_base = build_detr_base(args)

    model = HOIModel(
        backbone,
        detr_base,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = min(args.dec_layers_hopd, args.dec_layers_interaction)
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    if args.use_matching:
        losses.append('matching_labels')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)

    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, criterion, postprocessors
