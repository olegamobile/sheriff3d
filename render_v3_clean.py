import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = np.array([0x14, 0x18, 0x1f]) / 255.0
LIGHT = np.array([0.3, 0.5, 0.8])


def view_matrix(elev_deg, azim_deg):
    """Rotation taking world coords to camera coords (camera looks along -Z, X right, Y up),
    matching matplotlib's elev/azim convention."""
    el, az = np.radians(elev_deg), np.radians(azim_deg)
    cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # towards viewer
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, cam)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    return np.vstack([right, up, cam])


def zbuffer_render(items, elev, azim, width, height, bounds=None, zoom=1.0):
    """Orthographic z-buffer rasterizer with per-face diffuse shading.
    items: list of (mesh, base_rgb). bounds: (xmin,xmax,ymin,ymax) in camera plane."""
    R = view_matrix(elev, azim)
    all_pts = np.concatenate([m.vertices for m, _ in items])
    cam_pts = all_pts @ R.T
    if bounds is None:
        cx, cy = cam_pts[:, 0].mean(), cam_pts[:, 1].mean()
        half = max(np.ptp(cam_pts[:, 0]) / width, np.ptp(cam_pts[:, 1]) / height) * 0.5 / zoom
        bounds = (cx - half * width, cx + half * width, cy - half * height, cy + half * height)
    xmin, xmax, ymin, ymax = bounds
    sx, sy = width / (xmax - xmin), height / (ymax - ymin)

    img = np.tile(BG, (height, width, 1))
    zbuf = np.full((height, width), -np.inf)
    light = LIGHT / np.linalg.norm(LIGHT)
    light_cam = R @ light
    light_cam = light_cam / np.linalg.norm(light_cam)

    for mesh, base in items:
        v = mesh.vertices @ R.T
        px = (v[:, 0] - xmin) * sx
        py = (ymax - v[:, 1]) * sy
        pz = v[:, 2]
        n = mesh.face_normals @ R.T
        shade = 0.35 + 0.65 * np.clip(n @ light_cam, 0.0, 1.0)
        colors = np.clip(np.outer(shade, base), 0.0, 1.0)
        for fi, (a, b, c) in enumerate(mesh.faces):
            if n[fi, 2] <= 0:            # back-facing
                continue
            xs = np.array([px[a], px[b], px[c]])
            ys = np.array([py[a], py[b], py[c]])
            zs = np.array([pz[a], pz[b], pz[c]])
            x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), width - 1))
            y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), height - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
            if abs(d) < 1e-12:
                continue
            w1 = ((gx - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (gy - ys[0])) / d
            w2 = ((xs[1] - xs[0]) * (gy - ys[0]) - (gx - xs[0]) * (ys[1] - ys[0])) / d
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            depth = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = inside & (depth > sub)
            sub[upd] = depth[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = colors[fi]
    return img


def render_views():
    full = trimesh.load("stl_output/Jouet_Sheriff_Full_1to45.stl")
    wl = trimesh.load("stl_output/Jouet_Sheriff_Waterline_1to45.stl")
    cradle = trimesh.load("stl_output/Jouet_Sheriff_Display_Cradle_1to45.stl")
    rudder = trimesh.load("stl_output/Jouet_Sheriff_Rudder_1to45.stl")
    rudder.apply_translation([-10.0 / 45.0, 0.0, 0.0])

    boat_c = np.array([0.90, 0.93, 0.96])
    cradle_c = np.array([0.28, 0.45, 0.62])
    rudder_c = np.array([0.75, 0.45, 0.20])
    W, H = 1300, 900

    views = [
        ([(wl, boat_c)], 88, -90,
         "1. План палубы: утопленный бак с уткой, скруглённая рубка с люком, окнами и проходами,\n"
         "банки до транца с полукруглыми плечами, доска-упор для ног и балка оттяжки"),
        ([(wl, boat_c)], 28, -60,
         "2. С левого борта от носа: узкий планширь и опущенный бак, скруглённый нос рубки\n"
         "(дуга сверху и на палубе) со скатом 32°, акриловый люк перегибается через кромку крыши"),
        ([(wl, boat_c), (rudder, rudder_c)], 36, -125,
         "3. С левой раковины: планширь, банки до транца с полукруглыми карманами у крыльев рубки,\n"
         "плоская переборка под 30° до порожка с большим входом, канавки-проходы на крыше, румпель"),
        ([(full, boat_c), (rudder, rudder_c), (cradle, cradle_c)], 0, -90,
         "4. Профиль: обводы Филиппа Арле, низкая рубка, чугунный фальшкиль, кильблок"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax, (items, elev, azim, title) in zip(axes.flat, views):
        img = zbuffer_render(items, elev, azim, W, H, zoom=0.8)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, color='#f1f5f9', fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = "stl_output/Jouet_Sheriff_3D_Renders.png"
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    print(f"Rendered z-buffer shaded V4 views to: {out_path}")


if __name__ == "__main__":
    render_views()
