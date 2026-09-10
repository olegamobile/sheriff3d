import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BG = '#14181f'


def compute_shaded_colors(mesh, light_dir=np.array([0.4, 0.4, 0.8]), base_color=np.array([0.88, 0.91, 0.95])):
    """Diffuse shading per face from its normal; no edge lines, so planar faces stay clean."""
    normals = mesh.face_normals
    light_dir = light_dir / np.linalg.norm(light_dir)
    diff = np.clip(np.dot(normals, light_dir), 0.0, 1.0)
    intensity = 0.35 + 0.65 * diff
    colors = np.outer(intensity, base_color)
    return np.clip(colors, 0.0, 1.0)


def add_mesh(ax, mesh, colors, alpha=1.0):
    ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colors, edgecolor='none', alpha=alpha))


def setup_axes(ax, xl, yl, zl, elev, azim, title, zoom=1.0):
    ax.set_facecolor(BG)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_zlim(zl)
    ax.set_box_aspect((xl[1] - xl[0], yl[1] - yl[0], zl[1] - zl[0]), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')
    ax.set_title(title, color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)


def render_views():
    full = trimesh.load("stl_output/Jouet_Sheriff_Full_1to45.stl")
    wl = trimesh.load("stl_output/Jouet_Sheriff_Waterline_1to45.stl")
    cradle = trimesh.load("stl_output/Jouet_Sheriff_Display_Cradle_1to45.stl")
    rudder = trimesh.load("stl_output/Jouet_Sheriff_Rudder_1to45.stl")
    rudder.apply_translation([-10.0 / 45.0, 0.0, 0.0])

    fig = plt.figure(figsize=(18, 13), dpi=150)
    fig.patch.set_facecolor(BG)

    light = np.array([0.3, 0.5, 0.8])
    full_colors = compute_shaded_colors(full, light, np.array([0.90, 0.93, 0.96]))
    wl_colors = compute_shaded_colors(wl, light, np.array([0.90, 0.93, 0.96]))
    cradle_colors = compute_shaded_colors(cradle, light, np.array([0.28, 0.45, 0.62]))
    rudder_colors = compute_shaded_colors(rudder, light, np.array([0.75, 0.45, 0.20]))

    # 1. Deck plan
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    add_mesh(ax1, wl, wl_colors)
    setup_axes(ax1, (0, 140), (-32, 32), (-5, 35), 88, -90,
               "1. План палубы: бак с уткой, рубка с люком и окнами, кокпит с банками,\nкормовой банкой, рундуком, стойками с упором для ног и балкой оттяжки", zoom=1.25)

    # 2. Port bow quarter, like the real photo from the pontoon
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    add_mesh(ax2, wl, wl_colors)
    setup_axes(ax2, (10, 140), (-32, 32), (-5, 35), 28, -60,
               "2. С левого борта от носа: длинный плоский бак, плоский скат рубки 25°\nс акриловым люком, трапециевидное окно, мачта на переднем краю крыши", zoom=1.25)

    # 3. Port quarter, looking into the cockpit
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    add_mesh(ax3, wl, wl_colors)
    add_mesh(ax3, rudder, rudder_colors)
    setup_axes(ax3, (-15, 115), (-32, 32), (-5, 35), 36, -125,
               "3. С левой раковины: полоса планширя, банки, пол кокпита, кормовая банка,\nнаклонённая переборка с дверью, румпель над транцем", zoom=1.25)

    # 4. Side profile with keel and cradle
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    add_mesh(ax4, full, full_colors)
    add_mesh(ax4, rudder, rudder_colors)
    add_mesh(ax4, cradle, cradle_colors, alpha=0.9)
    setup_axes(ax4, (-15, 140), (-32, 32), (-25, 35), 0, -90,
               "4. Профиль: обводы Филиппа Арле, низкая рубка, чугунный фальшкиль, кильблок", zoom=1.25)

    plt.tight_layout()
    out_path = "stl_output/Jouet_Sheriff_3D_Renders.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    print(f"Rendered clean shaded V4 views to: {out_path}")


if __name__ == "__main__":
    render_views()
