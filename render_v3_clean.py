import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def compute_shaded_colors(mesh, light_dir=np.array([0.4, 0.4, 0.8]), base_color=np.array([0.88, 0.91, 0.95])):
    """Compute diffuse shaded colors for mesh faces based on normal vector."""
    normals = mesh.face_normals
    light_dir = light_dir / np.linalg.norm(light_dir)
    # Cosine of angle between normal and light
    diff = np.clip(np.dot(normals, light_dir), 0.0, 1.0)
    # Ambient + Diffuse
    intensity = 0.35 + 0.65 * diff
    colors = np.outer(intensity, base_color)
    return np.clip(colors, 0.0, 1.0)

def render_v3_views():
    mesh = trimesh.load("stl_output/Jouet_Sheriff_Full_1to45.stl")
    cradle = trimesh.load("stl_output/Jouet_Sheriff_Display_Cradle_1to45.stl")
    rudder = trimesh.load("stl_output/Jouet_Sheriff_Rudder_1to45.stl")
    rudder.apply_translation([-10.0 / 45.0, 0.0, 0.0])

    fig = plt.figure(figsize=(18, 13), dpi=150)
    fig.patch.set_facecolor('#14181f')

    light1 = np.array([0.3, 0.5, 0.8])
    boat_colors = compute_shaded_colors(mesh, light1, np.array([0.90, 0.93, 0.96]))
    cradle_colors = compute_shaded_colors(cradle, light1, np.array([0.28, 0.45, 0.62]))
    rudder_colors = compute_shaded_colors(rudder, light1, np.array([0.75, 0.45, 0.20]))

    # VIEW 1: Top Deck Plan
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_facecolor('#14181f')
    poly1 = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=boat_colors, edgecolor='none', alpha=1.0)
    ax1.add_collection3d(poly1)
    ax1.set_xlim([0, 140])
    ax1.set_ylim([-32, 32])
    ax1.set_zlim([-25, 35])
    ax1.view_init(elev=88, azim=-90)
    ax1.axis('off')
    ax1.set_title("1. Вид сверху: рубка во всю ширину, закругления кокпита, узкие кормовые скамейки", 
                  color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)

    # VIEW 2: Forward 3/4 View (Bow & Coachroof)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('#14181f')
    poly2 = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=boat_colors, edgecolor='none', alpha=1.0)
    ax2.add_collection3d(poly2)
    poly2_c = Poly3DCollection(cradle.vertices[cradle.faces], facecolors=cradle_colors, edgecolor='none', alpha=0.9)
    ax2.add_collection3d(poly2_c)
    ax2.set_xlim([0, 140])
    ax2.set_ylim([-32, 32])
    ax2.set_zlim([-25, 35])
    ax2.view_init(elev=22, azim=-40)
    ax2.axis('off')
    ax2.set_title("2. Носовая 3/4: ровный склон рубки, окно заходит на крышу, контуры боковых окон", 
                  color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)

    # VIEW 3: Aft Cockpit & Stern View
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_facecolor('#14181f')
    poly3 = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=boat_colors, edgecolor='none', alpha=1.0)
    ax3.add_collection3d(poly3)
    poly3_r = Poly3DCollection(rudder.vertices[rudder.faces], facecolors=rudder_colors, edgecolor='none', alpha=1.0)
    ax3.add_collection3d(poly3_r)
    ax3.set_xlim([-15, 135])
    ax3.set_ylim([-32, 32])
    ax3.set_zlim([-25, 35])
    ax3.view_init(elev=26, azim=-142)
    ax3.axis('off')
    ax3.set_title("3. Кормовая 3/4: сплошной транец, румпель НАД транцем, закругленные комингсы", 
                  color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)

    # VIEW 4: Pure Side Profile
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_facecolor('#14181f')
    poly4 = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=boat_colors, edgecolor='none', alpha=1.0)
    ax4.add_collection3d(poly4)
    poly4_r = Poly3DCollection(rudder.vertices[rudder.faces], facecolors=rudder_colors, edgecolor='none', alpha=1.0)
    ax4.add_collection3d(poly4_r)
    ax4.set_xlim([-15, 140])
    ax4.set_ylim([-32, 32])
    ax4.set_zlim([-25, 35])
    ax4.view_init(elev=0, azim=-90)
    ax4.axis('off')
    ax4.set_title("4. Профиль: точные обводы Филиппа Арле, контур иллюминатора, фальшкиль", 
                  color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout()
    out_path = "stl_output/Jouet_Sheriff_3D_Renders.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    print(f"Rendered clean shaded V3 views to: {out_path}")

if __name__ == "__main__":
    render_v3_views()
