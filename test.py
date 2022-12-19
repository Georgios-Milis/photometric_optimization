import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import trimesh, pyrender

import sys
sys.path.append('./models/')
from models.FLAME import FLAME, FLAMETex
import util
from renderer import Renderer


def see_obj(path):
    """Visualize given obj."""
    obj_trimesh = trimesh.load(path)
    mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    # Add camera
    # bz = images.shape[0]
    # cam = torch.zeros(bz, 3)
    # cam[:, 0] = 5
    # camera = pyrender.OrthographicCamera(1, 1)
    # camera_pose = np.eye(4)
    # camera_pose[2, 3] = 0.1
    # scene.add(camera, pose=camera_pose)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def see_mask():
    """See the skin mask."""
    image_name = "00003"
    image_mask_folder = './FFHQ_seg/'
    image_mask_path = os.path.sep.join([image_mask_folder, image_name + '.npy'])
    image_mask = np.load(image_mask_path, allow_pickle=True)
    plt.figure()
    plt.imshow(image_mask)
    plt.savefig(f'./test_results/{image_name}_mask.png', bbox_inches='tight')


def see_texture(image_name):
    # If we load 'tex' we get the 50 texture parameters...
    # So we load 'albedos' to see the face
    # The result is the same as the one that util.save_obj() writes
    texture = np.load(f'./test_results/{image_name}.npy', allow_pickle=True)[()]['albedos']
    texture = texture[0].transpose(1, 2, 0)
    plt.figure()
    plt.imshow(texture)
    plt.savefig(f'./test_results/{image_name}_texture.png', bbox_inches='tight')


def render_texture(config, image_name):
    shape = nn.Parameter(torch.zeros(0, config.shape_params).float().to(device))
    exp = nn.Parameter(torch.zeros(0, config.expression_params).float().to(device))
    pose = nn.Parameter(torch.zeros(0, config.pose_params).float().to(device))
    cam = torch.zeros(0, config.camera_params); cam[:, 0] = 5.
    cam = nn.Parameter(cam.float().to(device))
    lights = nn.Parameter(torch.zeros(0, 9, 3).float().to(device))

    # FLAME layers
    flame = FLAME(config).to(device)
    flametex = FLAMETex(config).to(device)

    # Vertices
    vertices, _, _ = flame(shape_params=shape, expression_params=exp, pose_params=pose)
    trans_vertices = util.batch_orth_proj(vertices, cam)
    trans_vertices[..., 1:] = - trans_vertices[..., 1:]

    # Albedo
    tex_params = np.load(f'./test_results/{image_name}.npy', allow_pickle=True)[()]['tex']
    tex_params = torch.from_numpy(tex_params).to(device)
    albedo = flametex(tex_params) / 255

    mesh_file = './data/head_template_mesh.obj'
    render = Renderer(256, obj_filename=mesh_file).to(device)
    ops = render(vertices, trans_vertices, albedo, lights)

    rendered_image = ops['images']
    rendered_image = rendered_image[0].detach().float().cpu()
    plt.figure()
    plt.imshow(rendered_image)
    plt.show()



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    config = {
        'flame_model_path': '../FLAME/model/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'tex_space_path': '../FLAME/model/FLAME_texture.npz',  # acquire it from FLAME project page
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,
        'use_face_contour': True,
        'cropped_size': 256,
        'batch_size': 1,
        'image_size': 224,
        'e_lr': 0.005,
        'e_wd': 0.0001,
        'savefolder': './test_results/'
    }
    config = util.dict2obj(config)

    # see_obj('./data/head_template_mesh.obj')
    # see_obj('./test_results/00003.obj')

    render_texture(config, "00003")
