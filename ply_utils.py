from plyfile import PlyElement, PlyData
import numpy as np
import os
import cv2

# ply = PlyData.read("/Users/dragos/Licenta/data/car1_resized/test.ply")
# vertex = ply['vertex'].data
# rgb = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=-1)
# # rgb = np.clip(rgb, 0, 1)
# # rgb = rgb.astype(np.float64)
# # print(rgb.shape, rgb.min(), rgb.max())
# print(rgb)

def project_points(points, R, t, fx, fy, cx, cy):
    cam_coords = (R @ points.T).T + t.reshape((1, 3))
    x, y, z = cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2]

    # Avoid division by zero
    z[z == 0] = 1e-6

    u = (fx * (x / z) + cx).astype(int)
    v = (fy * (y / z) + cy).astype(int)

    return u, v, z

def color_points_by_masks(cameras, masks_path, points, colors):
    for cam in cameras:
        fx, fy = cam["fx"], cam["fy"]
        width, height = cam["width"], cam["height"]
        cx, cy = width / 2, height / 2

        R = np.array(cam["rotation"])
        cam_pos = np.array(cam["position"])
        t = -R @ cam_pos.reshape((3, 1))

        img_name = cam["img_name"]
        img_base = os.path.splitext(img_name)[0]  # strips the .jpg
        mask_path = os.path.join(masks_path, img_base + ".png")  # assumes masks/mask_0001.jpg etc.

        if not os.path.exists(mask_path):
            print(f"[!] Mask not found: {mask_path}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[!] Failed to load mask: {mask_path}")
            continue
        mask = mask > 0  # binary
        # print(mask)

        # === Project points ===
        u, v, z = project_points(points, R, t, fx, fy, cx, cy)

        in_bounds = (
            (u >= 0) & (u < width) &
            (v >= 0) & (v < height) &
            (z > 0)
        )
        hit_ids = np.where(in_bounds)[0]
        u, v, z = u[hit_ids], v[hit_ids], z[hit_ids]
        Z = z
        count = 0
        for idx, x, y, z in zip(hit_ids, u, v, z):
            if mask[y, x] and z < 0.62:  # if inside mask
                count += 1
                # if z <0.6:
                #     print(True)
                colors[idx] = [1.0, 0.0, 0.0]  # red
        
        print(f"[+] Found {count} points in mask {img_name}")
        break

    return colors, Z

def save_ply(vertex, rgb, path):
    new_vertex_data = []

    for i in range(len(vertex)):
        v = vertex[i]

        # Replace only f_dc_* with updated RGB
        new_vertex = (
            v['x'], v['y'], v['z'],
            v['nx'], v['ny'], v['nz'],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],  # updated colors
            *[v[f"f_rest_{j}"] for j in range(45)],  # f_rest_0 to f_rest_44
            v['opacity'],
            v['scale_0'], v['scale_1'], v['scale_2'],
            v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'],
        )

        new_vertex_data.append(new_vertex)

    # Define the data structure
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # Add all f_rest_* fields
    vertex_dtype += [(f"f_rest_{i}", 'f4') for i in range(45)]

    # Remaining fields
    vertex_dtype += [
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]

    # Create numpy array
    vertex_array = np.array(new_vertex_data, dtype=vertex_dtype)

    # Create PlyElement and PlyData
    ply_el = PlyElement.describe(vertex_array, 'vertex')
    ply_data = PlyData([ply_el], text=False)

    # Save to disk
    ply_data.write(path)


def project_points_cl(points, camera):
    # Convert points to camera space
    R = camera['rotation']
    t = camera['position']
    
    # Transform points from world to camera coordinates
    points_cam = np.dot(points - t, R)
    
    # Project to image plane
    fx, fy = camera['fx'], camera['fy']
    px = fx * points_cam[:, 0] / points_cam[:, 2] + camera['width'] / 2
    py = fy * points_cam[:, 1] / points_cam[:, 2] + camera['height'] / 2
    
    # Check if points are in front of camera
    valid = points_cam[:, 2] > 0
    z = points_cam[:, 2]
    
    # Check if points project inside image bounds
    valid = valid & (px >= 0) & (px < camera['width']) & (py >= 0) & (py < camera['height']) & (z < 0.4) #the condition for depth
    
    return px, py, valid, z

def color_points_by_masks_cl(points, colors, cameras, mask_dir, new_color=[1.0, 0.0, 0.0]):
    new_colors = np.copy(colors)
    colored_points = np.zeros(len(points), dtype=bool)
    
    for camera in cameras:
        # Get 2D projections
        px, py, valid, z = project_points_cl(points, camera)
        # in_mask = None
        # Load corresponding mask
        mask_path = f"{mask_dir}/{camera['img_name'].replace('.jpg', '.png')}"
        print(mask_path)
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[!] Failed to load mask: {mask_path}")
                continue
            mask = mask > 0
            
            # Find points that project within mask
            px_valid = np.round(px[valid]).astype(int)
            py_valid = np.round(py[valid]).astype(int)
            
            # Check which points fall inside mask
            in_mask = mask[py_valid, px_valid]
            
            # Get indices of points that project into mask
            mask_indices = np.where(valid)[0][in_mask]

            print(len(mask_indices))
            # valid[in_mask]
            
            # Color points not already colored
            new_indices = mask_indices[~colored_points[mask_indices]]
            new_colors[new_indices] = new_color
            colored_points[new_indices] = True
            
        except Exception as e:
            print(f"Error processing mask for {camera['img_name']}: {e}")
    
    return new_colors, z[valid][in_mask]