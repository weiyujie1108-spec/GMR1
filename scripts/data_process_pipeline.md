# Data Process Pipeline

delete all annotation file in root folder

## BVH
step 1: convert bvh to robot dataset
* ams_23dof
python scripts/bvh_to_robot_dataset.py --src_folder data/ground --tgt_folder retargeted_data/ground/ --robot unitree_g1_fixed_wrist --num_cpus 20 --memory_threshold 30 --save_auto_check_format --save_format ams_23dof

* ams_29dof
python scripts/bvh_to_robot_dataset.py --src_folder data/ground --tgt_folder retargeted_data/ground/ --robot unitree_g1 --num_cpus 20 --memory_threshold 30 --save_auto_check_format --save_format ams_29dof

* gmr
python scripts/bvh_to_robot_dataset.py --src_folder data/ground --tgt_folder retargeted_data/ground/ --robot unitree_g1 --num_cpus 20 --memory_threshold 30 --save_auto_check_format --save_format gmr

step 2
* ams_23dof
python scripts/auto_check.py --motion_folder retargeted_data/ground  --robot_type unitree_g1_fixed_wrist
* ams_29dof / gmr
python scripts/auto_check.py --motion_folder retargeted_data/ground  --robot_type unitree_g1

step 3
python merge_pkl_files.py --input retargeted_data/ground --output merged_motions/ground.pkl


## SMPL-X
step 1: convert smplx to robot dataset
* ams_23dof
python scripts/smplx_to_robot_dataset.py --src_folder data/jitter_example --tgt_folder retargeted_data/jitter_example/ --robot unitree_g1_fixed_wrist --num_cpus 1 --memory_threshold 6 --save_auto_check_format --save_format ams_23dof

* ams_29dof
python scripts/smplx_to_robot_dataset.py --src_folder data/jitter_example --tgt_folder retargeted_data/jitter_example/ --robot unitree_g1 --num_cpus 1 --memory_threshold 6 --save_auto_check_format --save_format ams_29dof

* gmr
python scripts/smplx_to_robot_dataset.py --src_folder data/jitter_example --tgt_folder retargeted_data/jitter_example/ --robot unitree_g1 --num_cpus 1 --memory_threshold 6 --save_auto_check_format --save_format gmr

step 2
* ams_23dof
python scripts/auto_check.py --motion_folder retargeted_data/jitter_example  --robot_type unitree_g1_fixed_wrist

* ams_29dof / gmr
python scripts/auto_check.py --motion_folder retargeted_data/jitter_example  --robot_type unitree_g1

step 3
python merge_pkl_files.py --input retargeted_data/test_smplx --output merged_motions/test_smplx.pkl


## labeler
* ams_23dof
python scripts/robot_motion_labeler.py   --robot unitree_g1_fixed_wrist   --robot_motion_folder retargeted_data/jitter_example   --annotation_file auto_check_annotations.json

* ams_29dof / gmr
python scripts/robot_motion_labeler.py   --robot unitree_g1   --robot_motion_folder retargeted_data/ground   --annotation_file auto_check_annotations.json


