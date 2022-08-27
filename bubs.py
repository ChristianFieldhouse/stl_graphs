import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm

w = 500
num_blobs = 12
num_blobx = np.sqrt(num_blobs)

blobs = np.dstack([
    np.random.random((w, w))
    for _ in range(num_blobs)
])

def amplify(f, thresh=1, a=1):
    """boosts channels that are above the mean"""
    average = np.mean(f, axis=-1)
    for i in range(f.shape[-1]):
        factor = (f[:, :, i]/average)**a
        f[:, :, i] = f[:, :, i] * factor
        f[:, :, i] /= np.sqrt(np.mean(np.abs(f[:, :, i]**2)))
        f[:, :10, i] = f[:, -10:, i]
    return f

def smooth(f, l=w/num_blobx/10, offs=1/10):
    #todew:
    h, w, d = f.shape
    newones = [
        gaussian_filter(f[:, :, i], sigma=l)
        for i in range(d)
    ]
    return np.dstack(newones)

def save_channels(f, name):
    Image.fromarray(
        255 * (fields[:, :, :num_img_cols] == np.dstack([
            1.0 * np.max(fields, axis=-1) for _ in range(num_img_cols)
        ])).astype("uint8")
    ).save(f"{name}.png")

line_thickness = 4
line_f = 10
blob_thickness = w/num_blobx
def save_boundaries(f, name):
    s = np.sort(f, axis=-1)
    boundary = s[:, :, -2] / s[:, :, -1] > (1 - line_f * line_thickness/blob_thickness)
    Image.fromarray(
        boundary.astype("uint8") * 255
    ).save(f"{name}.png")

def save_zeros(f, name):
    f = f - np.mean(f)
    boundary = (np.abs(f[:, :, 0].real) <= np.sqrt(np.mean(np.abs(f)**2))*0.1)
    Image.fromarray(
        boundary.astype("uint8") * 255
    ).save(f"{name}.png")

num_img_cols = 3
for iter in tqdm(range(500)):
    blobs = smooth(amplify(blobs))
    save_boundaries(blobs, iter)


