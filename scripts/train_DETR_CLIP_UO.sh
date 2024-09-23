# --pretrained params/detr-r50-pre-2stage-q64_new.pth \
# try with 1e-5 lr from 9th epoch itself

python3 main.py \
    --pretrained params/detr-r50-pre-2stage-q64_new.pth \
    --output_dir output/DETR_CLIP_UO \
    --dataset_file hico_uo_st \
    --hoi_path data/hico_20160224_det \
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
    --model detr_clip \
    --inter_score \
    --num_workers 2 \
    --vdetach \
    --lr 0.00001 