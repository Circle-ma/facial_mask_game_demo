import numpy as np
import  torch
import torch.nn.functional as F
import os

from .Sim3DR import rasterize

def prepare_for_render(focal, center, device, camera_distance, init_lit, recenter, znear, zfar, bfm_file):

    model = np.load(bfm_file)
    # mean face shape. [3*N,1]
    mean_shape = model['meanshape'].astype(np.float32)
    # identity basis. [3*N,80]
    id_base = model['idBase'].astype(np.float32)
    # expression basis. [3*N,64]
    exp_base = model['exBase'].astype(np.float32)
    # face indices for each vertex that lies in. starts from 0. [N,8]
    point_buf = model['point_buf'].astype(np.int64) - 1
    # vertex indices for each face. starts from 0. [F,3]
    face_buf = model['tri'].astype(np.int64) - 1

    if recenter:
        mean_shape = mean_shape.reshape([-1, 3])
        mean_shape = mean_shape - np.mean(mean_shape, axis=0, keepdims=True)
        mean_shape = mean_shape.reshape([-1, 1])
    SH_a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
    SH_c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]

    def perspective_projection(focal, center):
        # return p.T (N, 3) @ (3, 3) 
        return np.array([
            focal, 0, center,
            0, focal, center,
            0, 0, 1
        ]).reshape([3, 3]).astype(np.float32).transpose()
   
    persc_proj = perspective_projection(focal, center)
    init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)

    mean_shape = torch.tensor(mean_shape).to(device)
    id_base = torch.tensor(id_base).to(device)
    exp_base = torch.tensor(exp_base).to(device)
    point_buf = torch.tensor(point_buf).to(device)
    face_buf = torch.tensor(face_buf).to(device)
    SH_a = torch.tensor(SH_a).to(device)
    SH_c = torch.tensor(SH_c).to(device)
    persc_proj = torch.tensor(persc_proj).to(device)
    init_lit = torch.tensor(init_lit).to(device)

    fov = 2 * np.arctan(center / focal) * 180 / np.pi

    def compute_shape(id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)

        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', exp_base, exp_coeff)
        face_shape = id_part + exp_part + mean_shape.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])
    
    def compute_norm(face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, face_buf[:, 0]]
        v2 = face_shape[:, face_buf[:, 1]]
        v3 = face_shape[:, face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(device)], dim=1)
        
        vertex_norm = torch.sum(face_norm[:, point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm


    def compute_color(face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)

        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = SH_a, SH_c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
             a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(device),
            -a[1] * c[1] * face_norm[..., 1:2],
             a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
             a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2  - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    
    def compute_rotation(angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(device)
        zeros = torch.zeros([batch_size, 1]).to(device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)


    def to_camera(face_shape):
        face_shape[..., -1] = camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj


    def transform(face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)


    def compute_perspective_proj(v):
        assert len(v.shape) == 2
        assert type(v) == np.ndarray
        v[:, :2] = focal/center * v[:, :2] / v[:,2].reshape((-1, 1))
        v = (v + np.array((1, 1, 0)))
        v[:, 2] = -v[:, 2]
        return v


    def render(coef_id, coef_exp, coef_angle, coef_trans, coef_tex, coef_gamma, facial_mask):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 257)
        """
        face_shape = compute_shape(coef_id, coef_exp)
        rotation = compute_rotation(coef_angle)


        face_shape_transformed = transform(face_shape, rotation, coef_trans)
        face_vertex = to_camera(face_shape_transformed)
        
        face_proj = to_image(face_vertex)

        face_norm = compute_norm(face_shape)
        face_norm_roted = face_norm @ rotation

        face_texture = torch.from_numpy(np.load(facial_mask)[np.newaxis, ...]).to(device=face_vertex.device) / 255.
        face_color = compute_color(face_texture, face_norm_roted, coef_gamma)

        f_color = np.clip(face_color[0].numpy(), 0, 1)
        
        v = compute_perspective_proj(face_vertex[0].numpy())
        v[:, :2] = v[:, :2] / 2 * 224 - 0.5

        pred_face, pred_mask, pred_depth = rasterize(v.astype(np.float32), face_buf.numpy().astype(np.int32).copy(order='C'), f_color, height=224, width=224)
        pred_face = pred_face.transpose((2, 0, 1))[np.newaxis, ...] / 255.
        pred_mask = pred_mask[np.newaxis, np.newaxis, ...]

        return pred_mask, pred_face

    return render



