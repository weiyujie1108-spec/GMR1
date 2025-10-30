import glob
import os
import sys

import joblib


# all_data = joblib.load('resources/motions/g1/g1_23dof_791_fixed_waist_wrist_bias.pkl')
# all_data = joblib.load('resources/motions/g1/merged_motions_3634.pkl')
# all_data = joblib.load('resources/motions/g1/split_motions_60s_all.pkl')
# all_data = joblib.load('resources/motions/g1/merged_sampled_static_poses_0822.pkl')
# motion_file =  f"resources/motions/g1/lafan_g1_all_wo_jump_fall.pkl"
motion_file =  f"/home/yixuan/yixuan/g1-23dof/legged_gym/resources/motions/g1/demo_10.pkl"
# motion_file =  f"resources/motions/g1/merged_sampled_static_poses_0818.pkl"
# motion_file = f"/Users/yixuanpan/g1_test_code/Humanoid-private-g1/legged_gym/resources/motions/g1/merged_sampled_static_poses_0822.pkl"
all_data = joblib.load(motion_file)
# all_data = joblib.load('resources/motions/g1/amass_all_free_waist.pkl')
# all_data = joblib.load('resources/motions/g1/sample_single_foot_static_constant_2048_3_z_bias.pkl')
# all_data = joblib.load('resources/motions/g1/mai_demo.pkl')
# all_data = joblib.load('resources/motions/g1/dance1_subject2.pkl')
# all_data = joblib.load('resources/motions/g1/walk1_subject1.pkl')
# select_motions = ['0-BMLmovi_Subject_32_F_MoSh_Subject_32_F_2_poses', '0-CMU_138_138_24_poses', '0-KIT_572_wave_both01_poses', '0-KIT_572_violin_left02_poses']
# select_motions = ['0-BMLmovi_Subject_32_F_MoSh_Subject_32_F_2_poses', '0-KIT_572_wave_both01_poses', '0-KIT_572_violin_left02_poses']
# select_motions = ['0-KIT_572_wave_both01_poses','0-BMLmovi_Subject_32_F_MoSh_Subject_32_F_2_poses', '0-KIT_572_violin_left02_poses', '0-KIT_572_wipe_circular_right07_poses','0-KIT_572_stir_left05_poses', '0-CMU_138_138_24_poses', '0-KIT_572_wave_both01_poses']
# select_motions = ['0-KIT_572_wipe_circular_right07_poses']
# select_motions = [ 'g1_dance1_subject2_segment001','0-gvhmr_motion_set_1_148','0-gvhmr_motion_set_1_012', '0-KIT_572_violin_left02_poses', '0-KIT_572_wipe_circular_right07_poses','0-BMLmovi_Subject_32_F_MoSh_Subject_32_F_2_poses', '0-KIT_572_wave_both01_poses']
# select_motions = [ 'g1_dance1_subject2_segment001','0-gvhmr_motion_set_1_148','0-gvhmr_motion_set_1_012', '0-KIT_572_violin_left02_poses', '0-KIT_572_wipe_circular_right07_poses','0-BMLmovi_Subject_32_F_MoSh_Subject_32_F_2_poses', '0-KIT_572_wave_both01_poses']
# select_motions = [ '0-gvhmr_dance_1']
# select_motions = [ '0-KIT_359_walking_fast06_poses']
# select_motions = [ '0-KIT_425_walking_medium02_poses']
# select_motions = [ 'g1_dance1_subject2']
# select_motions = [ 'g1_walk1_subject1']
# select_motions = [ '0-CMU_138_138_24_poses']
# select_motions = [ '0-KIT_348_walking_slow02_poses']
# select_motions = [ '0-gvhmr_motion_set_1_064']
# select_motions = ['g1_dance1_subject2_segment001','0-gvhmr_motion_set_1_148','0-gvhmr_motion_set_1_012']
# select_motions = ['g1_dance1_subject2_segment001','0-gvhmr_motion_set_1_012']
# select_motions = [ '0-gvhmr_motion_set_1_148'] # lift right
# select_motions = [ '0-gvhmr_motion_set_1_298'] # lift left
# select_motions = [ '0-gvhmr_motion_set_1_012'] # bend lift
# select_motions = [ 'single_stand_pose_right_001', 'single_stand_pose_left_004', 'single_stand_pose_left_006','single_stand_pose_right_011']
# select_motions = [ 'single_stand_pose_right_001', 'single_stand_pose_left_004']
# select_motions = [ 'single_stand_pose_right_001', 'single_stand_pose_left_004', 'single_stand_pose_right_031', 'single_stand_pose_left_034', 'single_stand_pose_left_036', 'single_stand_pose_right_041',
#                   'single_stand_pose_right_057', 'single_stand_pose_left_064', 'single_stand_pose_right_079', 'single_stand_pose_left_120', 
#                   'single_stand_pose_right_125', 'single_stand_pose_right_089', 'single_stand_pose_left_092', 'single_stand_pose_left_108', 
#                   'single_stand_pose_right_127', 'single_stand_pose_left_010', ] # merged_sampled_static_poses_0818  , 'single_stand_pose_left_016'
# select_motions = [ '0-gvhmr_motion_set_1_194']
# 
# select_motions = ['g1_dance1_subject2_segment001', 'g1_dance1_subject2_segment002', 'g1_dance1_subject2_segment003', 
#                   '0-KIT_348_walking_slow02_poses', '0-KIT_425_walking_medium02_poses', '0-KIT_359_walking_fast06_poses', '0-KIT_7_RightTurn06_poses', '0-KIT_12_WalkingStraightBackwards01_poses', '0-gvhmr_motion_set_1_136', '0-gvhmr_motion_set_2_065', '0-gvhmr_motion_set_2_056',
#                   '0-gvhmr_motion_set_1_098', '0-gvhmr_motion_set_1_224', '0-gvhmr_motion_set_1_240', '0-gvhmr_motion_set_1_157', '0-gvhmr_motion_set_1_088', '0-gvhmr_motion_set_1_029', '0-gvhmr_motion_set_1_064',
#                   '0-gvhmr_motion_set_1_298', '0-gvhmr_motion_set_1_148', '0-gvhmr_motion_set_1_099', '0-gvhmr_motion_set_1_260', '0-gvhmr_motion_set_1_021', '0-gvhmr_motion_set_1_228', '0-gvhmr_motion_set_1_111',
#                   '0-gvhmr_motion_set_1_047', '0-gvhmr_motion_set_1_262', '0-gvhmr_motion_set_1_194', '0-gvhmr_motion_set_1_012', '0-gvhmr_motion_set_1_261']
# select_motions = ['g1_dance1_subject2_segment1', 'g1_dance1_subject2_segment2', 'g1_dance1_subject2_segment3', 
#                   '0-KIT_348_walking_slow02_poses', '0-KIT_425_walking_medium02_poses', '0-KIT_359_walking_fast06_poses', '0-KIT_7_RightTurn06_poses', '0-KIT_12_WalkingStraightBackwards01_poses', '0-gvhmr_motion_set_1_136', '0-gvhmr_motion_set_2_065', '0-gvhmr_motion_set_2_056',
#                   '0-gvhmr_motion_set_1_098', '0-gvhmr_motion_set_1_224', '0-gvhmr_motion_set_1_240', '0-gvhmr_motion_set_1_157', '0-gvhmr_motion_set_1_088', '0-gvhmr_motion_set_1_029', '0-gvhmr_motion_set_1_064',
#                   '0-gvhmr_motion_set_1_298', '0-gvhmr_motion_set_1_148', '0-gvhmr_motion_set_1_099', '0-gvhmr_motion_set_1_260', '0-gvhmr_motion_set_1_021', '0-gvhmr_motion_set_1_228', '0-gvhmr_motion_set_1_111',
#                   '0-gvhmr_motion_set_1_047', '0-gvhmr_motion_set_1_262', '0-gvhmr_motion_set_1_194', '0-gvhmr_motion_set_1_012', '0-gvhmr_motion_set_1_261']
# select_motions = ['single_stand_pose_left_000', 'single_stand_pose_right_001', 'single_stand_pose_left_002', 
#                   'single_stand_pose_right_005', 'single_stand_pose_right_011', 'single_stand_pose_left_018', 'single_stand_pose_right_019',]
# select_motions = ['single_stand_pose_left_000', 'single_stand_pose_right_007', 'single_stand_pose_left_008', 'single_stand_pose_left_018',] # 0822-ok
# select_motions = ['single_stand_pose_left_000', 'single_stand_pose_right_001', 'single_stand_pose_right_003', 'single_stand_pose_left_004', 'single_stand_pose_right_005', 
#                   'single_stand_pose_right_007', 'single_stand_pose_left_008', 'single_stand_pose_left_018',] # 0822
select_motions = ['single_stand_pose_left_062','single_stand_pose_left_040','single_stand_pose_left_008','single_stand_pose_left_018'] # 0822
# ok 'single_stand_pose_left_008'
# soso 'single_stand_pose_left_018'
# ,'single_stand_pose_right_049', 'single_stand_pose_right_031'
# select_motions = [ 'single_stand_pose_left_062', 'single_stand_pose_right_005', 'single_stand_pose_right_003', 'single_stand_pose_right_007']
# danger single_stand_pose_left_044 ,'single_stand_pose_right_059', 'single_stand_pose_right_001', 'single_stand_pose_right_077'

# select_motions = all_data.keys()
# demo_data_1 = joblib.load('resources/motions/g1/demo_0713.pkl')
# demo_data_1 = joblib.load('resources/motions/g1/demo_0713.pkl')
# demo_data_1 = joblib.load('resources/motions/g1/ye_wen_squat_hip_yaw_bias_larger_adjusted_hand_v6.pkl')
# demo_data_2 = joblib.load('resources/motions/g1/g1_run1_subject5_153_253.pkl')
# demo_data_2 = joblib.load('resources/motions/g1/g1_run1_subject5_153_337.pkl')
# demo_data_1 = joblib.load('resources/motions/g1/ye_wen_squat_hip_yaw_bias_large_v4.pkl')
demo_data_1 = joblib.load('retargeted_data/AMASS/ACCAD/Male2Running_c3d/C3_-_run_stageii.pkl')

# demo_data_3 = joblib.load('resources/motions/g1/0-gvhmr_yewen3.pkl')
demo_data_2 = joblib.load('/home/yixuan/yixuan/g1-23dof/legged_gym/resources/motions/g1/lafan_g1_all_wo_jump_fall.pkl')
demo_data_3 = joblib.load('/home/yixuan/yixuan/g1-23dof/legged_gym/resources/motions/g1/g1_23dof_amass_3k.pkl')
# demo_data_3 = joblib.load('/home/yixuan/yixuan/unified-deploy/legged_gym/resources/motions/g1/demo_10.pkl')
data_dump = {}
# for motion in select_motions:
# # for motion in all_data.keys():
#     print(motion)
#     data_dump[motion] = all_data[motion]
# for motion in demo_data_1.keys():
#     print(motion)
#     if motion in data_dump.keys():
#         dump_motion = motion + '_demo_data_1'
#     else:
#         dump_motion = motion
#     data_dump[dump_motion] = demo_data_1[motion]
# for motion in demo_data_2.keys():
#     print(motion)
#     if motion in data_dump.keys():
#         dump_motion = motion + '_demo_data_2'
#     else:
#         dump_motion = motion
#     data_dump[dump_motion] = demo_data_2[motion]
for motion in demo_data_3.keys():
    print(motion)
    if motion in data_dump.keys():
        dump_motion = motion + '_demo_data_3'
    else:
        dump_motion = motion
    data_dump[dump_motion] = demo_data_3[motion]
for motion in select_motions:
# for motion in all_data.keys():
    print(motion)
    data_dump[motion] = all_data[motion]

# joblib.dump(data_dump, 'resources/motions/g1/0-MPI_Limits_03099_lar1_poses.pkl')
joblib.dump(data_dump, 'resources/motions/g1/selected_motions_for_save_deployment.pkl')
# joblib.dump(data_dump, 'resources/motions/g1/0-KIT_572_wipe_circular_right07_poses.pkl')