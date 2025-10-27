from general_motion_retargeting.optitrack_vendor.NatNetClient import setup_optitrack
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import threading
import argparse
import time
from mocap_api import *
import numpy as np
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from scipy.spatial.transform import Rotation as R
import torch

def get_event_type_name(event_type_value):
    """
    Convert event type value to corresponding enum name
    
    Args:
        event_type_value: The numeric event type value
        
    Returns:
        str: The corresponding event type name
    """
    event_type_map = {
        MCPEventType.InvalidEvent: 'InvalidEvent',
        MCPEventType.AvatarUpdated: 'AvatarUpdated',
        MCPEventType.TrackerUpdated: 'TrackerUpdated',
        MCPEventType.AliceIMUUpdated: 'AliceIMUUpdated',
        MCPEventType.AliceRigidbodyUpdated: 'AliceRigidbodyUpdated',
        MCPEventType.AliceTrackerUpdated: 'AliceTrackerUpdated',
        MCPEventType.AliceMarkerUpdated: 'AliceMarkerUpdated',
    }
    return event_type_map.get(event_type_value, f'Unknown({event_type_value})')


class MocapStream:
    """
    Mocap Hybrid Data Server Demo class for demonstrating how to get Hybrid Data Server data through Mocap API
    """
    
    def __init__(self):
        """
        Initialize Mocap HDS Demo instance
        """
        self.app = None
        self.running = False
    
    def start(self, udp_port=7012):
        """
        Start Mocap application and handle event loop
        
        Args:
            udp_port: UDP port number, default is 7012
        """
        # Initialize Mocap application
        self.app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_bvh_rotation(MCPBvhRotation.XYZ)
        self.app.set_settings(settings)
        self.app.open()
        print(f"Mocap application initialized, UDP port: {udp_port}")
        
        self.running = True
        try:
            while self.running:
                evts = self.app.poll_next_event()
                for evt in evts:
                    if evt.event_type == MCPEventType.AvatarUpdated: # avatar bvh
                        self._handle_avatar_data(evt)
                    elif evt.event_type == MCPEventType.AliceTrackerUpdated: # tracker
                        self._handle_tracker_data()
                    elif evt.event_type == MCPEventType.AliceMarkerUpdated: # marker
                        self._handle_marker_data()
                    else:
                        print('Other events:', get_event_type_name(evt.event_type))
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.stop()

    def _handle_marker_data(self):
        """
        Handle marker data
        """
        alicehub = MCPAliceHub()
        recv, count = alicehub.get_marker_list()
        if count > 0:
            recv, count1 = alicehub.get_marker_list(count)
            timestamp = alicehub.get_marker_timestamp()
            for i in range(count):
                marker_handle = recv[i]
                marker = MCPMarker(marker_handle)
                marker_x, marker_y, marker_z = marker.get_marker_position()
                print(f'marker data : timestamp: {timestamp}, position: {marker_x}, {marker_y}, {marker_z}')
    
    def _handle_tracker_data(self):
        """
        Handle tracker data
        """
        alicehub = MCPAliceHub()
        recv, count = alicehub.get_PWR_list()
        if count > 0:
            recv, count1 = alicehub.get_PWR_list(count)
            timestamp = alicehub.get_PWR_timestamp()
            for i in range(count):
                PWRHandle = recv[i]
                MCPPWRH = MCPPWR(PWRHandle)
                id = MCPPWRH.get_PWR_id()
                status = MCPPWRH.get_PWR_status()
                position = MCPPWRH.get_PWR_position()
                quaternion = MCPPWRH.get_PWR_quaternion()
                print('tracker data : timestamp',timestamp,'id:', id, 'status:', status, 'position:', position, 'quaternion:', quaternion)

    def _handle_avatar_data(self, evt):
        """
        Handle avatar data
        """
        avatar = MCPAvatar(evt.event_data.avatar_handle)
        self.avatar = avatar
        # # Get and print timecode information
        # second, nanosecond = avatar.get_avatar_posture_ptp_time()
        # print(f" avatar posture ptp time : {second},{nanosecond}")
        # joints = avatar.get_joints()  # Get all joint data
        # for joint in joints:
        #     link_name = joint.get_name()  # Get joint name
        #     position = joint.get_local_position()  # Get joint position
        #     rotation = joint.get_local_rotation()  # Get joint rotation
        #     print(f"avatar data : joint: {link_name}, position: {position}, rotation: {rotation}")
    
    def get_lafan1_data_from_avatar_data(self):
        """
        Get lafan1 data from avatar data
        """
        # mocap: bones names and parents (hardcoded)
        bones = ['Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Neck1', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3']
        parents = np.array([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9, 10, 11,  9, 13, 14, 15,
                            16, 17, 18, 16, 20, 21, 22, 16, 24, 25, 26, 16, 28, 29, 30, 16, 32,
                            33, 34,  9, 36, 37, 38, 39, 40, 41, 39, 43, 44, 45, 39, 47, 48, 49,
                            39, 51, 52, 53, 39, 55, 56, 57])
        
        # mocap realtime data
        joints = self.avatar.get_joints()  # Get all joint data
        pos = np.zeros((len(joints), 3))
        quat = np.zeros((len(joints), 4))
        for i, joint in enumerate(joints):
            pos[i] = joint.get_local_position()  # Get joint position
            quat[i] = joint.get_local_rotation()  # Get joint rotation
    
        global_data = utils.quat_fk(quat, pos, parents)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        result = {}
        for i, bone in enumerate(bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][i])
            position = global_data[1][i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
        result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
        result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]
        return result
            
   
    def stop(self):
        """
        Close Mocap application
        """
        self.running = False
        if self.app:
            self.app.close()
            print("Mocap application closed")

def main(args):
    # Check if firewall is disabled on this machine
    print("Make sure to disable firewall on both machines:")
    print("On OptiTrack computer: Disable Windows Firewall")
    print("On this computer: sudo ufw disable")

    mocap_stream = MocapStream()

    # start a thread to client.run()
    thread = threading.Thread(target=mocap_stream.start)
    thread.start()

    if not mocap_stream:
        print("Failed to setup Mocap Stream")
        exit(1)

    print(f"Mocap Stream connected: {mocap_stream.running}")
    print("Starting motion retargeting...")

    retarget = GMR(
            src_human="bvh_nokov",
            tgt_robot=args.robot,
            actual_human_height=1.75,
        )
    viewer = RobotMotionViewer(robot_type=args.robot)
    # Initialize the forward kinematics
    device = "cuda:0"
    kinematics_model = KinematicsModel(retarget.xml_file, device=device)

    while True:
        lafan1_data = mocap_stream.get_lafan1_data_from_avatar_data()
        print(lafan1_data)
        qpos = retarget.retarget(lafan1_data)
        viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            rate_limit=False,
        )
        qpos_list = []
        qpos_list.append(qpos.copy())
        qpos_list = np.array(qpos_list)
        root_pos = qpos_list[:, :3]
        root_rot = qpos_list[:, 3:7]
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:]
        identity_root_pos = torch.zeros((1, 3), device=device)
        identity_root_rot = torch.zeros((1, 4), device=device)
        identity_root_rot[:, -1] = 1.0
        local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                identity_root_rot, 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="192.168.200.160")
    parser.add_argument("--client_ip", type=str, default="192.168.200.117")
    parser.add_argument("--use_multicast", type=bool, default=False)
    # parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot", type=str, default="unitree_g1_fixed_wrist")
    args = parser.parse_args()
    main(args)
    