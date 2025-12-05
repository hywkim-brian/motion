import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch
import smplx
import os
import trimesh
from scipy.sparse import csr_matrix

MODEL_PATH = "./smplx/SMPLX_NEUTRAL.npz" 
MOTION_PATH = "./DanceDB/Dance_clip_8.npy"

class MeshDecimator:
    def __init__(self, high_verts, high_faces, decimation_ratio=0.8):
        
        self.high_mesh = trimesh.Trimesh(vertices=high_verts, faces=high_faces)
        self.mode = 'remesh' 
        
        self.low_mesh = self.high_mesh.simplify_quadric_decimation(0.8)
        self.low_faces = self.low_mesh.faces
        print(f"Final Vertex Count: {len(self.low_mesh.vertices)}")
        closest, distance, triangle_id = self.high_mesh.nearest.on_surface(self.low_mesh.vertices)
        barycentric = trimesh.triangles.points_to_barycentric(
            triangles=self.high_mesh.vertices[self.high_mesh.faces[triangle_id]],
            points=closest
        )
        n_low = len(self.low_mesh.vertices)
        n_high = len(high_verts)
        rows = np.repeat(np.arange(n_low), 3)
        cols = self.high_mesh.faces[triangle_id].flatten()
        data = barycentric.flatten()
        self.transfer_matrix = csr_matrix((data, (rows, cols)), shape=(n_low, n_high))

    def transfer(self, high_poly_verts):
        if self.mode == 'slice':
            return high_poly_verts
        else:
            return self.transfer_matrix.dot(high_poly_verts)

class SMPLAnimator:
    def __init__(self, model_path, motion_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")

        motion_np = np.load(motion_path)
        if isinstance(motion_np, np.lib.npyio.NpzFile):
            motion_np = motion_np[motion_np.files[0]]
        motion_tensor = torch.tensor(motion_np).float().to(self.device)
        
        self.root_orient = motion_tensor[:, :3]
        self.pose_body   = motion_tensor[:, 3:66]
        self.pose_hand   = motion_tensor[:, 66:156]
        self.pose_jaw    = motion_tensor[:, 156:159]
        self.trans       = motion_tensor[:, 309:312]
        self.betas       = motion_tensor[:, 312:]

        self.left_hand_pose  = self.pose_hand[:, :45]
        self.right_hand_pose = self.pose_hand[:, 45:]
        
        self.num_frames = motion_tensor.shape[0]
        self.fps = 30.0

        self.model = smplx.create(
            model_path=model_path, 
            model_type='smplx', gender='neutral', 
            use_pca=False, num_betas=10, ext='npz'
        ).to(self.device)

        output = self.model(betas=self.betas[0:1], return_verts=True)
        template_verts = output.vertices[0].detach().cpu().numpy()
        template_faces = self.model.faces

        self.decimator = MeshDecimator(template_verts, template_faces)
        self.faces = self.decimator.low_faces 

    def update(self, t):
        frame_idx = int(t * self.fps) % self.num_frames
        with torch.no_grad():
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
            high_verts = output.vertices[0].cpu().numpy()
        low_verts = self.decimator.transfer(high_verts)
        
        v_corrected = low_verts.copy()
        v_corrected[:, 1] = low_verts[:, 2]
        v_corrected[:, 2] = low_verts[:, 1]
        v_corrected[:, 1] *= -1 
        
        return v_corrected

def run_gui():
    human = SMPLAnimator(MODEL_PATH, MOTION_PATH)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.view_init(elev=10, azim=45)
    
    trisurf = [None]

    def animate(frame):
        t = frame * (1/30.0) 
        verts = human.update(t)
        
        if trisurf[0]: 
            trisurf[0].remove()
            
        trisurf[0] = ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=human.faces,
            cmap='magma',       
            shade=True,
            edgecolor='none',
            linewidth=0,
            antialiased=False
        )
        return trisurf[0],

    ani = FuncAnimation(fig, animate, frames=200, interval=33, blit=False)
    plt.show()

if __name__ == "__main__":
    run_gui()