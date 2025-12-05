import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch
import smplx
import os

MODEL_PATH = "./smplx/SMPLX_NEUTRAL.npz" 
MOTION_PATH = "/Users/hyunwookim/Documents/motion/DanceDB/Dance_clip_8.npy"

class SMPLAnimator:
    def __init__(self, model_path, motion_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        motion_np = np.load(motion_path)
        
        if isinstance(motion_np, np.lib.npyio.NpzFile):
            motion_np = motion_np[motion_np.files[0]]
            
        motion_tensor = torch.tensor(motion_np).float().to(self.device)
        
        self.root_orient = motion_tensor[:, :3]        # 0 - 3
        self.pose_body   = motion_tensor[:, 3:66]      # 3 - 66 (63 dims)
        self.pose_hand   = motion_tensor[:, 66:156]    # 66 - 156 (90 dims)
        self.pose_jaw    = motion_tensor[:, 156:159]   # 156 - 159 (3 dims) - Adjusted index based on 66+90=156
        # self.face_expr = motion_tensor[:, 159:209]   # 159 - 209 (50 dims)
        # self.face_shape= motion_tensor[:, 209:309]   # 209 - 309 (100 dims)
        self.trans       = motion_tensor[:, 309:312]   # 309 - 312 (3 dims)
        self.betas       = motion_tensor[:, 312:]      # 312 - end (10 dims)
        self.left_hand_pose  = self.pose_hand[:, :45]
        self.right_hand_pose = self.pose_hand[:, 45:]
        self.num_frames = motion_tensor.shape[0]
        self.fps = 60.0 

        print(f"Loaded Motion-X flat data: {self.num_frames} frames.")
        self.model = smplx.create(
            model_path=model_path, 
            model_type='smplx',
            gender='neutral', 
            use_pca=False,
            num_betas=10,
            ext='npz'
        ).to(self.device)

        self.faces = self.model.faces[::10] 

    def update(self, t):
        # 1. CALCULATE FRAME INDEX
        frame_idx = int(t * self.fps) % self.num_frames
        
        # 2. RUN THE MODEL
        output = self.model(
            betas=self.betas[frame_idx:frame_idx+1],
            global_orient=self.root_orient[frame_idx:frame_idx+1],
            body_pose=self.pose_body[frame_idx:frame_idx+1],
            left_hand_pose=self.left_hand_pose[frame_idx:frame_idx+1],
            right_hand_pose=self.right_hand_pose[frame_idx:frame_idx+1],
            jaw_pose=self.pose_jaw[frame_idx:frame_idx+1],
            transl=self.trans[frame_idx:frame_idx+1],
            return_verts=True
        )
        
        # 3. EXTRACT VERTICES
        v = output.vertices[0].detach().cpu().numpy()
        
        # 4. COORDINATE FIX (Y-Up to Z-Up for Matplotlib)
        v_corrected = v.copy()
        v_corrected[:, 1] = v[:, 2] # Swap Y and Z
        v_corrected[:, 2] = v[:, 1]
        v_corrected[:, 1] *= -1     # Invert Y to prevent mirroring
        
        return v_corrected

def run_gui():
    try:
        human = SMPLAnimator(MODEL_PATH, MOTION_PATH)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=10, azim=45)
    
    trisurf = [None]

    def animate(frame):
        t = frame * (1/30.0) 
        new_verts = human.update(t)
        
        if trisurf[0]: 
            trisurf[0].remove()
            
        trisurf[0] = ax.plot_trisurf(
            new_verts[:, 0], new_verts[:, 1], new_verts[:, 2],
            triangles=human.faces,
            cmap='magma',       
            shade=True,
            linewidth=0,
            antialiased=False
        )
        return trisurf[0],

    ani = FuncAnimation(fig, animate, frames=200, interval=33, blit=False)
    plt.show()

if __name__ == "__main__":
    run_gui()