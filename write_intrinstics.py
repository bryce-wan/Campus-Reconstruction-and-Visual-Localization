import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.read_write_model import read_images_binary, qvec2rotmat
from hloc.utils.viz import plot_images, plot_keypoints, save_plot
from pathlib import Path
outputs = Path('outputs/theater/aa')
model = pycolmap.Reconstruction(outputs / 'sfm')

references = []
ids = []

for i in model.images.items():
    ids.append(i[0])
with open('intrinstics.txt', 'w') as file:
    for i in ids:
        name = model.images[i].name
        width = model.cameras[model.images[i].camera_id].width
        height = model.cameras[model.images[i].camera_id].height
        f, cx, cy, k = model.cameras[model.images[i].camera_id].params
        file.writelines(
            f"{name} SIMPLE_RADIAL {width} {height} {f} {cx} {cy} {k}\n")
