from typing import List
import torch
import trimesh
import numpy as np
import os
import string
import parmap
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Textures,
    look_at_view_transform,
    RasterizationSettings,
    FoVPerspectiveCameras,
    PointLights,
    HardGouraudShader,
    MeshRenderer,
    MeshRasterizer,
)


def sample_points(mesh: trimesh.Trimesh, num_points=10000):
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def convert_stl_to_obj(stlfilename):
    stlfile = open(stlfilename, "rb")
    stl = trimesh.exchange.stl.load_stl_binary(stlfile)
    stlfile.close()

    vertices = stl["vertices"]
    faces = stl["faces"] + 1
    face_normals = stl["face_normals"]

    objfilename = stlfilename[:-4] + ".obj"
    objfile = open(objfilename, "w")
    objfile.write("# File type: ASCII OBJ\n")
    objfile.write("# Generated from " + os.path.basename(stlfilename) + "\n")

    for v in vertices:
        objfile.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for f in faces:
        objfile.write(f"f {f[0]} {f[1]} {f[2]}\n")
    objfile.close()


def render_image(
    verts: List[torch.Tensor],
    faces: List[torch.Tensor],
    num_verts_per_mesh: List,
    elev: int,
    azim: int,
    device: str,
):
    N = len(verts)
    verts_rgb = torch.ones(
        (N, np.max(num_verts_per_mesh), 3), requires_grad=False, device=device
    )

    for i in range(N):
        verts_rgb[i, num_verts_per_mesh[i] :, :] = -1

    textures = Textures(verts_rgb=verts_rgb)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)

    R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=224,
        blur_radius=0,
        faces_per_pixel=1,
        max_faces_per_bin=40000,
        #     bin_size=0,
        perspective_correct=False,
    )
    lights = PointLights(device=device, location=[[0, 5, 0]])

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = HardGouraudShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer, shader)

    return renderer(meshes)


if __name__ == "__main__":
    set1_dir = "../data/1SET_STL"
    set1_objs = [x for x in os.listdir(set1_dir) if x[-3:] == "obj"]
