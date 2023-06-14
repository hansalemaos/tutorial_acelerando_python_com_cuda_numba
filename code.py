import numba.cuda as cuda
import numpy as np


@cuda.jit
def find_rgb_value(image, colors, matches1, matches2, lastindex):
    x, y = cuda.grid(2)

    if x < image.shape[0] and y < image.shape[1]:
        for color in colors:
            if (
                image[x, y, 0] == color[0]
                and image[x, y, 1] == color[1]
                and image[x, y, 2] == color[2]
            ):
                pos = cuda.atomic.add(lastindex, 0, 1)
                matches1[pos] += x
                matches2[pos] += y


def get_kernel(image, threadsperblock=(8, 8, 3)):
    blockspergrid_x = (image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_z = (image.shape[2] + threadsperblock[2] - 1) // threadsperblock[2]

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    return lambda *a, **kw: find_rgb_value[blockspergrid, threadsperblock](*a, **kw)


def search_colors(
    pic, colors=(255, 255, 255), threadsperblock=(8, 8, 3), dtypetouse=np.int64
):
    image1 = cuda.to_device(pic)
    matchesx = np.zeros(pic.shape[0] * pic.shape[1], dtype=dtypetouse)
    matchesy = np.zeros(pic.shape[0] * pic.shape[1], dtype=dtypetouse)
    lastindex = np.array([0], dtype=dtypetouse)
    find_rgb_valuewithkernel = get_kernel(pic, threadsperblock)
    color = np.array(colors, dtype=dtypetouse)
    color1 = cuda.to_device(color)
    matches1 = cuda.to_device(matchesx)
    matches2 = cuda.to_device(matchesy)
    lastindex2 = cuda.to_device(lastindex)
    find_rgb_valuewithkernel(image1, color1, matches1, matches2, lastindex2)
    lai = lastindex2.copy_to_host()[0]
    ax = matches1[:lai].copy_to_host()
    ay = matches2[:lai].copy_to_host()
    results = np.dstack([ax, ay])
    return results

from locate_pixelcolor_numbacuda import search_colors
from a_cv_imwrite_imread_plus import open_image_in_cv
import numpy as np
# 4525 x 6623 x 3 picture https://www.pexels.com/pt-br/foto/foto-da-raposa-sentada-no-chao-2295744/

path = r"C:\Users\hansc\Downloads\pexels-alex-andrews-2295744.jpg"
colors=[(66,  71,  69),(62,  67,  65),(144, 155, 153),(52,  57,  55),(127, 138, 136),(53,  58,  56),(51,  56,  54),(32,  27,  18),(24,  17,   8),]
picnp = open_image_in_cv(path,channels_in_output=3)
coords=search_colors(pic=picnp,colors=colors,threadsperblock=(8, 8,3),dtypetouse = np.int32)
print(coords)
colorswhite=[(255,  255,  255)]
coords0=search_colors(pic=picnp,colors=colorswhite,threadsperblock=(8, 8,3),dtypetouse = np.int32)

