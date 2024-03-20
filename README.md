# LOCnet
Jointly Training Classification and Depth Prediction

command for training locnet

`torchrun --nproc_per_node 2 --master_port=25641 locnet_train.py --train_data_dir /gpfs/data/shared/imagenet/ILSVRC2012/train --val_data_dir /cifs/data/tserre_lrs/projects/prj_video_imagenet/mae/data/imagenet/val --depth_dir /cifs/data/tserre_lrs/projects/prj_model_vs_human/imagenet_depth --model hconvgru_resnet --batch_size 2 --log_interval 100 --epochs 200 --warmup-epochs 5 --smoothing 0.01`
