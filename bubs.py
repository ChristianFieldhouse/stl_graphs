import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from network_to_stl import Network

w = 500
num_blobs = 51
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
        f[:, :10, i] = (f[:, -10:, i] + f[:, :10, i])/2
        f[:, -10:, i] = f[:, :10, i]
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
    junction = s[:, :, -3] / s[:, :, -1] > (1 - line_f * line_thickness/blob_thickness)
    
    im = boundary.astype("uint8") * 100 + junction.astype("uint8") * 155
    
    junctions = []
    contours, hierarchy = cv2.findContours(junction.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        x, y = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)
        channels = set(list(f[y, x]).index(val) for val in s[y, x, -3:])
        junctions.append((x, y, channels)) 
    print(junctions)
    def to_cylinder(j, circ=w):
        radius = circ/(2 * np.pi)
        angle = j[0] / radius
        return np.array((
            radius * np.cos(angle),
            radius * np.sin(angle),
            -j[1],
        )) / 5
    Image.fromarray(im).save(f"out/pngs/{name}.png")
    if len(junctions) > 3:
        culled = [j for j in junctions if j[:2] != (0, 0)]
        transformed, channels = [to_cylinder(j) for j in culled], [j[2] for j in culled]
        n = Network()
        for t in transformed:
            n.add_junction_at(t)
        for t, cs in zip(transformed, channels):
            junc = n.get_junction_at(t)
            for other_t, other_cs in zip(transformed, channels):
                if len(cs.intersection(other_cs)) == 2: 
                    num_neighbours = len(junc.neighbours)
                    if num_neighbours < 3:
                        n.join_junctions_at(t, other_t, symmetric=False)
            junc = n.get_junction_at(t)
            num_neighbours = len(junc.neighbours)
            #print(num_neighbours)
            closest = sorted(transformed, key=lambda x : np.sum((x - t)**2))
            if num_neighbours == 0:
                junc.neighbours.append(np.array((0, 0, 0)))
            for i in range(3 - num_neighbours):
                junc.neighbours.append(junc.center + np.sum(junc.center - junc.neighbours, axis=0))
                
        n.save(f"out/stls/{name}")
        return n
    

def save_zeros(f, name):
    f = f - np.mean(f)
    boundary = (np.abs(f[:, :, 0].real) <= np.sqrt(np.mean(np.abs(f)**2))*0.1)
    Image.fromarray(
        boundary.astype("uint8") * 255
    ).save(f"{name}.png")

num_img_cols = 3
for iter in tqdm(range(1500)):
    blobs = smooth(amplify(blobs))
    if iter % 10 == 0:
        save_boundaries(blobs, iter)


