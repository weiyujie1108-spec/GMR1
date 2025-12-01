### 运行指令: python robot_target_master.py
import asyncio
import websockets
import pickle
import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ultralytics import YOLO
from easydict import EasyDict as edict

# === 引入 HybrIK 预处理 ===
try:
    from hybrik.utils.presets import SimpleTransform3DSMPLCam
except ImportError:
    print("❌ Error: 找不到 hybrik 模块，请确认 PYTHONPATH")
    exit(1)

# === TensorRT 推理类 (保持不变) ===
class HybrikTRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.input_shape = (1, 3, 256, 256)
        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * 4)
        self.h_beta = cuda.pagelocked_empty((1, 10), dtype=np.float32)
        self.d_beta = cuda.mem_alloc(self.h_beta.nbytes)
        self.h_theta = cuda.pagelocked_empty((1, 24, 3, 3), dtype=np.float32)
        self.d_theta = cuda.mem_alloc(self.h_theta.nbytes)
        self.h_transl = cuda.pagelocked_empty((1, 3), dtype=np.float32)
        self.d_transl = cuda.mem_alloc(self.h_transl.nbytes)
        
        self.context.set_tensor_address("input", int(self.d_input))
        self.context.set_tensor_address("pred_beta", int(self.d_beta))
        self.context.set_tensor_address("pred_theta_mat", int(self.d_theta))
        self.context.set_tensor_address("transl", int(self.d_transl))

    def infer(self, img_tensor):
        self.context.set_input_shape("input", self.input_shape)
        cuda.memcpy_htod_async(self.d_input, np.ascontiguousarray(img_tensor), self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_beta, self.d_beta, self.stream)
        cuda.memcpy_dtoh_async(self.h_theta, self.d_theta, self.stream)
        cuda.memcpy_dtoh_async(self.h_transl, self.d_transl, self.stream)
        self.stream.synchronize()
        
        theta_vec = np.zeros((24, 3), dtype=np.float32)
        theta_mat = self.h_theta.reshape(24, 3, 3)
        for i in range(24): 
            theta_vec[i] = cv2.Rodrigues(theta_mat[i])[0].ravel()
            
        return {'pred_theta': theta_vec, 'transl': self.h_transl.copy(), 'pred_beta': self.h_beta.copy()}

# === 数据转换逻辑 ===
def pack_smplx_data(pred_theta, pred_beta, transl):
    beta_padded = np.concatenate([pred_beta.flatten(), np.zeros(6, dtype=np.float32)])
    root_orient = pred_theta[0:1] 
    pose_body = pred_theta[1:22].flatten()
    
    return {
        'betas': beta_padded[None, :],
        'root_orient': root_orient[None, :],
        'pose_body': pose_body[None, :],
        'trans': transl[None, :], # (1, 3)
        'gender': 'neutral'
    }

async def send_motion_data(websocket):
    print(f"Client connected: {websocket.remote_address}")
    
    # === [关键修改 1] 初始化 RealSense (开启 Depth + Color + Align) ===
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 开启深度流
    
    # 创建对齐对象 (深度对齐到颜色)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    profile = pipeline.start(cfg)
    
    # 获取相机内参 (用于像素 -> 3D坐标转换)
    depth_profile = profile.get_stream(rs.stream.depth) # 注意：对齐后使用Color的内参或对齐后的Depth内参
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    
    print(f"RealSense Intrinsics: {intrinsics}")
    
    # 加载模型
    print("Loading Models...")
    hybrik_engine = HybrikTRTInference("hybrik_trt10.engine")
    detector = YOLO("yolov8s.engine", task='detect') 
    
    # 预处理
    bbox_3d_shape = (2.0, 2.0, 2.0)
    dummpy_set = edict({'joint_pairs_17': None, 'joint_pairs_24': None, 'joint_pairs_29': None, 'bbox_3d_shape': bbox_3d_shape})
    transformer = SimpleTransform3DSMPLCam(dummpy_set, scale_factor=0.3, color_factor=0.2, occlusion=True, input_size=(256, 256), output_size=(64, 64), depth_dim=64, bbox_3d_shape=bbox_3d_shape, rot=30, sigma=2, train=False, add_dpg=False, loss_type='L1')

    print(">>> Streaming Started!")
    try:
        while True:
            # === [关键修改 2] 获取对齐后的帧 ===
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames) # 对齐
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            vis_img = frame.copy()
            
            # 1. 检测 (YOLO)
            results = detector(frame, classes=[0], conf=0.5, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                # 获取最大框
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = np.argmax(areas)
                bbox = boxes[idx]
                
                # 画框
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # === [关键修改 3] 用 RealSense 测量 3D 坐标 ===
                # 计算中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # 限制边界
                cx = min(max(cx, 0), 639)
                cy = min(max(cy, 0), 479)
                
                # 读取深度 (单位: 米)
                # 注意：为了更稳，可以取中心区域 3x3 或 5x5 的中值，这里先取单点
                dist = depth_frame.get_distance(cx, cy)
                
                # 如果距离有效 (>0)
                if 0.1 < dist < 10.0:
                    # 反投影：像素(u,v) + 深度(z) -> 空间坐标(x,y,z)
                    # RealSense 坐标系: X右, Y下, Z前 (单位: 米)
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                    real_transl = np.array(point_3d, dtype=np.float32)
                    
                    # 可视化距离
                    cv2.putText(vis_img, f"Depth: {dist:.2f}m", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    # 如果深度无效（比如太近或太远），降级使用 HybrIK 的估算值（或上一帧）
                    # 这里暂时设为 None，后面判断
                    real_transl = None
                    cv2.putText(vis_img, "Depth Invalid", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # 2. HybrIK 推理
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_input, _, _ = transformer.test_transform(img_rgb, bbox)
                res = hybrik_engine.infer(pose_input.unsqueeze(0).numpy())
                
                # === [关键修改 4] 替换 transl ===
                final_transl = res['transl'][0] # 默认用 HybrIK 的
                if real_transl is not None:
                    final_transl = real_transl # 如果 RealSense 有效，就覆盖它！
                    
                # 3. 打包发送
                smplx_data = pack_smplx_data(res['pred_theta'], res['pred_beta'], final_transl)
                await websocket.send(pickle.dumps(smplx_data))
            
            else:
                cv2.putText(vis_img, "No Person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                await asyncio.sleep(0.001)

            cv2.imshow("RealSense + HybrIK Feed", vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.001)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

async def main():
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    async with websockets.serve(send_motion_data, "0.0.0.0", 8765) as server:
        print("=" * 60)
        print("WebSocket Server running on ws://0.0.0.0:8765")
        print(f"Local IP address: {local_ip}")
        print(f"Other hosts can connect using: ws://{local_ip}:8765")
        print("=" * 60)
        await asyncio.Future() 

if __name__ == "__main__":
    asyncio.run(main())