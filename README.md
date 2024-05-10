# LOCNet
Jointly Training Classification and Depth Prediction.



## Command for Training LOCNet

`torchrun --nproc_per_node 2 --master_port=25641 main.py --train_data_dir /gpfs/data/shared/imagenet/ILSVRC2012/train --val_data_dir /cifs/data/tserre_lrs/projects/prj_video_imagenet/mae/data/imagenet/val --depth_dir /cifs/data/tserre_lrs/projects/prj_model_vs_human/imagenet_depth --model locnet --batch_size 32 --log_interval 50 --epochs 100 --warmup-epochs 5 --smoothing 0.1 --log-wandb --save-ddir predicted_depth_maps/trial4`



## Command for Evaluating LOCNet on OOD Dataset

`python evaluation.py`