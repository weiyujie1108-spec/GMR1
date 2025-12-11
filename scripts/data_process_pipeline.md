# Data Process Pipeline

## BVH

python scripts/bvh_to_robot_dataset.py --src_folder data/ground --tgt_folder retargeted_data/ground/ --robot unitree_g1_fixed_wrist --num_cpus 20 --memory_threshold 30

python merge_pkl_files.py --input retargeted_data/ground --output merged_motions/ground.pkl

## SMPL-X



