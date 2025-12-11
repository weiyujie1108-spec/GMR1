# Data Process Pipeline

## BVH

python scripts/bvh_to_robot_dataset.py --src_folder data/ground --tgt_folder retargeted_data/ground/ --robot unitree_g1_fixed_wrist --num_cpus 20 --memory_threshold 30 --save_auto_check_format

python scripts/auto_check.py --motion_folder retargeted_data/ground  --robot_type unitree_g1_fixed_wrist

python merge_pkl_files.py --input retargeted_data/ground --output merged_motions/ground.pkl

## SMPL-X

python scripts/smplx_to_robot_dataset.py --src_folder data/AMASS/CMU/01 --tgt_folder retargeted_data/test_smplx/ --robot unitree_g1_fixed_wrist --num_cpus 1 --memory_threshold 6 --save_auto_check_format

python scripts/auto_check.py --motion_folder retargeted_data/test_smplx  --robot_type unitree_g1_fixed_wrist

python merge_pkl_files.py --input retargeted_data/test_smplx --output merged_motions/test_smplx.pkl


## labeler
python scripts/robot_motion_labeler.py   --robot unitree_g1_fixed_wrist   --robot_motion_folder retargeted_data/test_smplx   --annotation_file auto_check_annotations.json