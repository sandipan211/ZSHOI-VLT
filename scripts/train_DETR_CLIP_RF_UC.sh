# EDIT: Earlier nproc_per_node=4
# EDIT: Earlier epochs=90
# EDIT: Eariler batch size = 4
# EDIT: Earlier --pretrained params/detr-r50-pre-2stage-q64.pth

# python3 -m torch.distributed.launch \
#         --nproc_per_node=1 \
#         --use_env \
        python3 main.py \
        --pretrained params/detr-r50-pre-2stage-q64.pth \
        --output_dir output/CDN_CLIP_Baseline_RF_UC \
        --dataset_file hico_uc_st \
        --hoi_path ../data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 50 \
        --use_nms_filter \
        --batch_size 8 \
        --clip_backbone RN50x16 \
        --model cdn_clip \
        --inter_score \
        --vdetach \
        --uc_type rare_first \
        --lr 0.00001 \
        --start_epoch 40 \
        --finetune
