import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

FILE_NAME = "./meshes/blub.obj"  
SWIM_SPEED = 8.0        
WAVE_FREQ = 1.5         
TAIL_SWING = 0.3        

class SimpleSwimmer:
    def __init__(self, filepath):
        self.vertices, self.faces = self.load_obj(filepath)
        self.normalize_mesh()
        self.min_x = self.vertices[:, 0].min()
        self.max_x = self.vertices[:, 0].max()
        self.len_x = self.max_x - self.min_x

    def load_obj(self, filename):
        verts = []
        faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    verts.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    idxs = [int(p.split('/')[0]) - 1 for p in parts]
                    if len(idxs) == 3:
                        faces.append(idxs)
                    elif len(idxs) == 4: 
                        faces.append([idxs[0], idxs[1], idxs[2]])
                        faces.append([idxs[0], idxs[2], idxs[3]])
        return np.array(verts), np.array(faces)

    def normalize_mesh(self):
        center = self.vertices.mean(axis=0)
        self.vertices -= center
        scale = np.max(np.abs(self.vertices))
        self.vertices /= scale
        self.base_vertices = self.vertices.copy()

    def update(self, t):
        v = self.base_vertices.copy()
        # --- THE HARD-CODED MATH ---
        # 1. Calculate normalized position along spine (0.0 at nose, 1.0 at tail)
        # We assume the user aligned the fish along X.
        norm_x = (v[:, 0] - self.min_x) / self.len_x  
        #flip-> alignment to blub mesh
        norm_x = 1.0 - norm_x
        # 2. Stiffness Mask
        # Square the value so the head (0.0) is very stiff, and tail (1.0) is loose.
        # If your fish moves its head too much, increase power to 3 or 4.
        mask = norm_x ** (2.)
        # 3. The Wave Function
        # We offset the Z axis (Sideways) based on X position and Time.
        wave = np.sin(WAVE_FREQ * v[:, 0] - SWIM_SPEED * t)
        v[:, 0] += TAIL_SWING * mask * wave 
        return v

def run_gui():
    fish = SimpleSwimmer(FILE_NAME)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=30, azim=-60)
    trisurf = [None]

    def animate(frame):
        t = frame * 0.05
        new_verts = fish.update(t)
        if trisurf[0]: trisurf[0].remove()
        trisurf[0] = ax.plot_trisurf(
            new_verts[:, 0], new_verts[:, 1], new_verts[:, 2],
            triangles=fish.faces,
            cmap='jet',       
            shade=True,
            linewidth=0,
            antialiased=False
        )
        return trisurf[0],

    ani = FuncAnimation(fig, animate, frames=200, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    run_gui()
    #if you want just the function that takes t, do 
    # fish = SimpleSwimmer(FILE_NAME)
    # new_verts = fish.update(t)


