import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def render_model_previews():
    mesh = trimesh.load("stl_output/Jouet_Sheriff_Full_1to45.stl")
    cradle = trimesh.load("stl_output/Jouet_Sheriff_Display_Cradle_1to45.stl")
    rudder = trimesh.load("stl_output/Jouet_Sheriff_Rudder_1to45.stl")
    rudder.apply_translation([-10.0 / 45.0, 0.0, 0.0])

    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#1a1e24')

    # View 1: 3D Isometric View (on Cradle)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_facecolor('#1a1e24')
    
    # Subsample faces for fast crisp rendering
    faces_boat = mesh.faces
    verts_boat = mesh.vertices
    poly_boat = Poly3DCollection(verts_boat[faces_boat], alpha=0.9, edgecolor='#2b323c', linewidths=0.2)
    poly_boat.set_facecolor('#dce4ec')
    ax1.add_collection3d(poly_boat)

    faces_c = cradle.faces
    verts_c = cradle.vertices
    poly_c = Poly3DCollection(verts_c[faces_c], alpha=0.85, edgecolor='#1b2028', linewidths=0.2)
    poly_c.set_facecolor('#486581')
    ax1.add_collection3d(poly_c)

    faces_r = rudder.faces
    verts_r = rudder.vertices
    poly_r = Poly3DCollection(verts_r[faces_r], alpha=0.95, edgecolor='#1b2028', linewidths=0.2)
    poly_r.set_facecolor('#ba6820')
    ax1.add_collection3d(poly_r)

    ax1.set_xlim([0, 140])
    ax1.set_ylim([-35, 35])
    ax1.set_zlim([-25, 35])
    ax1.view_init(elev=22, azim=-50)
    ax1.set_title("Jouët Sheriff 600 - 3D Isometric View (with Cradle & Rudder)", color='white', fontsize=12, pad=10)
    ax1.axis('off')

    # View 2: Side Profile (Elevation)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_facecolor('#1a1e24')
    poly2 = Poly3DCollection(verts_boat[faces_boat], alpha=0.95, edgecolor='#334e68', linewidths=0.1)
    poly2.set_facecolor('#dce4ec')
    ax2.add_collection3d(poly2)
    poly2_r = Poly3DCollection(verts_r[faces_r], alpha=0.95, edgecolor='#1b2028', linewidths=0.1)
    poly2_r.set_facecolor('#ba6820')
    ax2.add_collection3d(poly2_r)
    ax2.set_xlim([-10, 140])
    ax2.set_ylim([-35, 35])
    ax2.set_zlim([-25, 35])
    ax2.view_init(elev=0, azim=-90) # side view
    ax2.set_title("Side Profile (Sheer, Fin Keel & Bulb, Coachroof, Rudder)", color='white', fontsize=12, pad=10)
    ax2.axis('off')

    # View 3: Top Deck View
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_facecolor('#1a1e24')
    poly3 = Poly3DCollection(verts_boat[faces_boat], alpha=0.95, edgecolor='#334e68', linewidths=0.1)
    poly3.set_facecolor('#dce4ec')
    ax3.add_collection3d(poly3)
    ax3.set_xlim([0, 140])
    ax3.set_ylim([-35, 35])
    ax3.set_zlim([-25, 35])
    ax3.view_init(elev=90, azim=-90) # top view
    ax3.set_title("Top Deck Plan (Foredeck, Hatch, Coachroof, Cockpit Benches)", color='white', fontsize=12, pad=10)
    ax3.axis('off')

    # View 4: Stern 3/4 View (Transom & Cockpit)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_facecolor('#1a1e24')
    poly4 = Poly3DCollection(verts_boat[faces_boat], alpha=0.9, edgecolor='#2b323c', linewidths=0.2)
    poly4.set_facecolor('#dce4ec')
    ax4.add_collection3d(poly4)
    poly4_r = Poly3DCollection(verts_r[faces_r], alpha=0.95, edgecolor='#1b2028', linewidths=0.2)
    poly4_r.set_facecolor('#ba6820')
    ax4.add_collection3d(poly4_r)
    ax4.set_xlim([-15, 135])
    ax4.set_ylim([-35, 35])
    ax4.set_zlim([-25, 35])
    ax4.view_init(elev=25, azim=-145) # aft quarter view
    ax4.set_title("Aft 3/4 Stern View (Transom Rake, Tiller, Cockpit Coaming)", color='white', fontsize=12, pad=10)
    ax4.axis('off')

    out_img = "stl_output/Jouet_Sheriff_3D_Renders.png"
    plt.tight_layout()
    plt.savefig(out_img, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    print(f"Rendered multi-angle 3D previews to {out_img}")

if __name__ == "__main__":
    render_model_previews()
