import itertools

import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from network_to_stl import Network, save_stl
from stl import mesh

from mpl_toolkits import mplot3d
from matplotlib import pyplot

m = mesh.Mesh.from_file("inputs/torus.stl")

def testmesh():
    data = np.zeros(2, dtype=mesh.Mesh.dtype)
    data["vectors"][0] = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    data["vectors"][1] = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ])
    return mesh.Mesh(data.copy())
#m = testmesh()

m.centers = (m.v0 + m.v1 + m.v2)/3
def vert(m, idx):
    """return m.v{idx}"""
    return getattr(m, f"v{idx}")
def normalised(vec):
    return vec / np.sqrt(np.sum(vec**2))
def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))

class Edge():
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1
        self.dir = v1 - v0
        self.dirn = normalised(self.dir)
    
    def intersects(self, plane):
        if np.sum(self.dirn * plane.normal) == 0:
            return None
        a = np.dot(plane.point - self.v0, plane.normal) / np.dot(self.dir, plane.normal)
        return None if a < 0 else None if a > 1 else self.v0 + a*self.dir
    
    def length(self):
        return np.sqrt(np.sum(self.dir**2))


class Line():
    """Todo: why Line and Edge??"""
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction

    def __repr__(self):
        return f"Line({self.point} -> {self.direction})"
    
    def intersects(self, plane):
        if np.sum(self.direction * plane.normal) == 0:
            return None
        a = np.dot(plane.point - self.point, plane.normal) / np.dot(self.direction, plane.normal)
        return self.point + a*self.direction
    
    def dist_to_point(self, point):
        n = normalised(self.direction)
        diff = point - (self.point + np.sum(n*(point - self.point)) * n)
        return np.sqrt(np.sum(diff**2))

class Plane():
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal
    
    def __repr__(self):
        return f"Plane({self.point} -> {self.normal})"
    
    def __mul__(self, other):
        new_dir = np.cross(self.normal, other.normal)
        intersect_dir = np.cross(new_dir, self.normal)
        a = np.dot(other.point - self.point, other.normal) / np.dot(intersect_dir, other.normal)
        new_point = self.point + a * intersect_dir
        return Line(new_point, new_dir)

m.edges = [
    [Edge(vert(m, i)[k], vert(m, j)[k]) for i, j in ((0, 1), (1, 2), (2, 0))]
    for k in range(len(m.v0))
]

num_blobs = 120
num_blobx = np.sqrt(num_blobs)

points = [np.array(p) for p in [
    (0, 0, 1),
    (0, 0, -1),
    (1, 0, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (0, -1, 0),
]]
points = [random.choice(m.centers) + np.random.random(3)*0.01 for _ in range(num_blobs)]

def force(point, other):
    d = other - point + (np.random.random() - 0.5) / 10000
    def norm(diff, exponent=4):
        x, y, z = np.abs(diff)
        n = 1/(np.sum(np.abs(diff)**exponent) + 0.00001)
        return n
    return - d * norm(d)

def calculate_forces(points):
    forces = np.zeros_like(points)
    for i, p in enumerate(points): 
        for j, q in enumerate(points):
            if i == j:
                continue
            else:
                forces[i] += force(points[i], points[j])
    return forces

def repel(points, max_dist=0.05):
    forces = calculate_forces(points)
    forces = forces * max_dist / np.max(np.abs(forces))
    points += forces
    return points

def project_single(point, m=m):
    """Project 3d points onto 2d mesh, according to mesh normals."""
    center_distances = np.sqrt(np.sum((m.centers - point)**2, axis=-1))
    closest_center_distance = np.min(center_distances)
    closest_center = m.centers[np.where(center_distances == closest_center_distance)][0]
    absolute_closest = {
        "point": closest_center,
        "distance": closest_center_distance,
    }
    ## TODO: massively optimise; cull loads of triangles initially! And fail early...
    for i, n in enumerate(m.normals):
        if dist(m.centers[i], point) - np.max([e.length() for e in m.edges[i]]) > absolute_closest["distance"]:
            continue
        normalised_n = n / np.sqrt(np.sum(n**2))
        projected = point - np.dot(point - m.v0[i], normalised_n) * normalised_n
        closest = None
        edges_outside = []
        for k, edge in enumerate(m.edges[i]):
            normal = np.cross(edge.v0 - projected, edge.v1 - projected)
            if np.dot(normal, normalised_n) < 0:
                edges_outside.append(k)
        #print(edges_outside)
        if len(edges_outside) > 1:
            closest = vert(m, {(0, 1): 1, (1, 2): 2, (0, 2): 0}[tuple(edges_outside[:2])])[i]
        elif len(edges_outside) == 1:
            edge = m.edges[i][edges_outside[0]]
            on_line_from_c0 = np.sum((projected - edge.v0)*edge.dirn)
            if on_line_from_c0 > 1:
                closest = edge.v1
            elif on_line_from_c0 < 0:
                closest = edge.v0
            else:
                closest = edge.v0 + on_line_from_c0 * edge.dir
        else:
            closest = projected
        if np.sqrt(np.sum((point-closest)**2)) < absolute_closest["distance"]:
            absolute_closest["distance"] = np.sqrt(np.sum((point-closest)**2))
            absolute_closest["point"] = closest
            #print("closer:", absolute_closest)
    return absolute_closest["point"]

def project(points, m=m):
    return [project_single(point, m) for point in points]        

def bisection(point0, point1, m=m):
    midpoint = (point0 + point1)/2
    normal = point1 - point0
    normal_n = normal / np.sqrt(np.sum(normal**2))
    plane = Plane(midpoint, normal)
    """for line v0 + a*v1, a in [0, 1] want a : (v0 + a*v1 - x).n = 0
    so... a = (x.n - v0.n) / v1.n has to be in [0, 1]
    
    Can just enumerate all lines and then construct the bisection
    """
    edge_intersects = []
    for k in range(len(m.v0)):
        for edge in m.edges[k]:
            ip = edge.intersects(plane)
            if ip is not None:
                edge_intersects.append(ip)
    
    return edge_intersects

def intersections(line, m=m):
    ins = []
    for k in range(len(m.v0)):
        normal = np.cross(m.v1[k] - m.v0[k], m.v2[k] - m.v0[k])
        plane = Plane(m.v0[k], normal)
        intersection = line.intersects(plane)
        insides = 0 if intersection is None else len([
            edge for edge in m.edges[k] if
            np.dot(np.cross(intersection - edge.v0, intersection - edge.v1), normal) >= 0
        ])
        #print(insides, line.point, line.direction)
        if insides == 3:
            ins.append(intersection)
    #print(ins)
    return sorted(ins, key=lambda x: np.sum((x - line.point)))

def voronoi_cell(point, points):
    #print("cell")
    points = [p for p in points if not np.all(p==point)]
    planes = [Plane((point + p)/2, point-p) for p in points]
    points_and_planes = list(zip(points, planes))
    points_and_planes = sorted(points_and_planes, key=lambda x:np.sum((x[0]-point)**2))
    genplanes = [points_and_planes[0]]
    corner_rays = []
    
    closest_v0_idx = np.argmin(np.sum((m.v0 - point)**2, axis=-1))
    normal = m.normals[closest_v0_idx]
    #normal = np.cross(points_and_planes[0][0]-point, points_and_planes[1][0]-point)
    #print(f"{normal=}")
    
    def intersect_distance_anticlockwise(q0, q1, closest_intersect_sofar=1e10):
        p0, plane_q0 = q0
        p1, plane_q1 = q1
        if dist(p1, point) > 2*closest_intersect_sofar:
            return False, "point too far!"
        n = np.cross(p0 - point, p1-point)
        wrong_direction = np.dot(normal, n) <= 0
        if wrong_direction:
            return False, "wrong direction"
        common = plane_q0 * plane_q1
        intersects = sorted(intersections(common), key=lambda x: np.sum((x-point)**2))
        d = common.dist_to_point(point)
        if d > closest_intersect_sofar:
            return False, "too far"
        truedist = dist(intersects[0], point) if intersects else 1e10
        if truedist < closest_intersect_sofar:
            return True, truedist
        return False, "not quite"
    while True:
        #print(f"{genplanes=}")
        other_points_and_planes = [p for p in points_and_planes if all(np.any(p[0]!=g[0]) for g in genplanes[1:])]
        
        next_best = None
        best_score = 1e10
        for i in range(len(other_points_and_planes)):
            is_better, outcome = intersect_distance_anticlockwise(genplanes[-1], other_points_and_planes[i], best_score)
            #print(other_points_and_planes[i], is_better, outcome)
            if is_better:
                best_score = outcome
                next_best = other_points_and_planes[i]
            elif outcome == "point too far!":
                break
        if next_best is None:
            print([intersect_distance_anticlockwise(genplanes[-1], op, best_score) for op in other_points_and_planes])
        if np.all(next_best[0] == genplanes[0][0]):
            break
        genplanes.append(next_best)
        #corner_rays.append(genplanes[-1] * genplanes[-2])
    corner_rays = [p*q for (_, p), (_, q) in zip(genplanes, genplanes[-1:] + genplanes[:-1])]
    #print(list(zip(genplanes, genplanes[-1:] + genplanes[:-1])))
    #print(corner_rays)
    corners = [sorted(intersections(ray), key=lambda x: np.sum((x-point)**2))[0] for ray in corner_rays]
    nice_loop = []
    def lerp(p0, p1, a):
        return (1-a)*p0 + a*p1
    for i in range(len(genplanes)-1):
        for k in range(10):
            nice_loop.append(lerp(genplanes[i][0], genplanes[i+1][0], k/10))
    bisections = [bisection(point, q) for q, _ in genplanes]
    def cull_bisection(p0, p1, points):
        n = normalised(p1-p0)
        def first_half(p):
            return (np.dot(normalised(p-p0), n) > 0.5) and np.sum((p-p0)**2) < np.sum((p-p1)**2)
        def second_half(p):
            return (np.dot(normalised(p-p1), -n) > 0.5) and np.sum((p-p0)**2) > np.sum((p-p1)**2)
        def in_segment(p):
            return first_half(p) or second_half(p)
        return [p0] + sorted([p for p in points if in_segment(p)], key=lambda x:np.dot(x-p0, n)) + [p1]
    segments = [cull_bisection(c0, c1, b) for c0, c1, b in zip(corners, corners[1:] + corners[:1], bisections)]
    all_pts = []
    for s in segments:
        all_pts += s
    return corners, segments

def myplot(m, points, history):
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)
    
    collection = mplot3d.art3d.Poly3DCollection(m.vectors, alpha=0.2)
    face_color = [0.5, 0.5, 1] # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
    collection.set_facecolor(face_color)
    axes.add_collection3d(collection)
    axes.scatter(*[np.array(points)[:, i] for i in range(3)])
    for h in history:
        axes.scatter(*[np.array(h)[:, i] for i in range(3)])
    scale = m.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    pyplot.show()


def voronoi_network(points=points, r=1, vcs=None, eps=0.001):
    if vcs is None:
        vcs = []
        for p in tqdm(points):
            vcs.append(voronoi_cell(p, points))
    # compile list of corners:
    network = Network(r=r)
    # add unique corners as junctions:
    for corners, paths in vcs:
        for corner in corners:
            if all([dist(corner, j.center) > eps for j in network.junctions]):
                network.add_junction_at(corner)
    # join around circles
    for vc in vcs:
        corners, paths = vc
        num_pts = len(corners)
        for i in range(num_pts):
            next = (i + 1) % num_pts
            network.join_junctions_closest_to(corners[i], corners[next], symmetric=False, path=paths[i])
    return network

def min_distances(points):
    min_dists = []
    for i in range(len(points)):
        pa = np.array([p for p in points if np.any(p != points[i])])
        min_dists.append(np.sqrt(np.min(np.sum((points[i] - pa)**2, axis=-1))))
    return min_dists

history = []
def iterate(points, iters=3):
    for iter in (pbar := tqdm(range(iters))):
        #history.append(points.copy())
        points = project(repel(points))
        md = min_distances(points)
        pbar.set_description(f"{np.std(md) / np.mean(md)}")
    return points


array = np.array
float32 = np.float32
points = [array([-1.395503 , -0.2370153, -0.2184199], dtype=float32), array([ 0.39451176,  1.34575364, -0.22525268]), array([-0.11952332,  1.4106747 , -0.2184199 ], dtype=float32), array([-1.2503527 ,  0.20864701, -0.3860673 ], dtype=float32), array([ 0.51250786, -0.3963088 ,  0.31166765], dtype=float32), array([0.5765636 , 0.8851949 , 0.46657318], dtype=float32), array([0.20705704, 0.80674577, 0.43945763], dtype=float32), array([ 0.99858516,  0.7793507 , -0.3868024 ], dtype=float32), array([ 0.26128762,  1.06083403, -0.45252117]), array([-0.02137489, -1.42819157,  0.11682725]), array([-0.52153928,  1.26775777, -0.26402854]), array([-0.45880344,  0.2858922 ,  0.11521445]), array([ 0.87718313, -0.27674661, -0.45038905]), array([-0.46447744,  0.97313713,  0.45639099]), array([-0.11689833, -1.4107451 , -0.21907935], dtype=float32), array([1.43271704, 0.12498651, 0.06664454]), array([ 1.3892094 , -0.4769169 , -0.00489766], dtype=float32), array([ 0.1341782 , -0.63215013, -0.31286567]), array([ 0.3605293 , -1.4236994 ,  0.00552097], dtype=float32), array([ 1.336947  , -0.4589752 ,  0.22138599], dtype=float32), array([ 1.1591866 ,  0.90223104, -0.00437007], dtype=float32), array([-1.12564459,  0.60885563,  0.37233894]), array([-1.0442251 , -0.17424987, -0.46581027], dtype=float32), array([0.8339214 , 0.6488718 , 0.46657318], dtype=float32), array([ 0.50031478, -1.26240554, -0.26396389]), array([-1.2928267,  0.6996424,  0.       ], dtype=float32), array([ 1.1600368, -0.9028924,  0.       ], dtype=float32), array([-0.925273  , -0.50808203,  0.46657318], dtype=float32), array([1.336427  , 0.45879596, 0.22200686], dtype=float32), array([ 1.00284171,  0.44575073, -0.44662331]), array([-0.13347933,  0.67134741,  0.34260712]), array([0.64804083, 0.00095388, 0.31166765], dtype=float32), array([-0.57065856,  0.30680886, -0.31166765], dtype=float32), array([0.34921017, 1.3721447 , 0.2184199 ], dtype=float32), array([0.98086631, 0.01555007, 0.45768617]), array([-0.5666561 , -0.61021   , -0.43945763], dtype=float32), array([ 1.43128238, -0.16443807,  0.04575559]), array([-0.5617857 ,  0.6148149 ,  0.43945763], dtype=float32), array([ 0.13295749, -0.52706325,  0.11247836], dtype=float32), array([-0.52325718, -0.84540124,  0.46062682]), array([-1.2309058 , -0.6661325 ,  0.23712467], dtype=float32), array([-0.86449441, -0.93902613, -0.37620658]), array([-0.1184206,  1.4109539,  0.2184199], dtype=float32), array([-0.42254392,  0.96824566, -0.46653278]), array([ 0.30907142, -1.2283831 ,  0.3868024 ], dtype=float32), array([-0.58662264, -0.50826312,  0.40666476]), array([-0.2113952 , -0.73241544, -0.39741961]), array([0.24415806, 1.11034021, 0.43409522]), array([-0.5496824 , -0.9684448 , -0.43998292], dtype=float32), array([ 0.50630354, -0.5530942 , -0.38897768]), array([ 0.35169905,  0.4051021 , -0.10930688]), array([0.9984866 , 0.34478197, 0.46657318], dtype=float32), array([-0.25693408,  0.59447664, -0.31166765], dtype=float32), array([ 0.9993997 , -0.34309486,  0.46657318], dtype=float32), array([-1.43529339,  0.04605601, -0.06029129]), array([0.77302766, 1.1832069 , 0.22159962], dtype=float32), array([0.8040138, 1.2306348, 0.       ], dtype=float32), array([ 0.78729925,  0.11229867, -0.4206158 ]), array([-0.73515175, -1.18557423,  0.2209795 ]), array([-0.08725813, -1.0530431 , -0.46657318], dtype=float32), array([-0.96042469,  0.24113213,  0.45973379]), array([-1.4377401 ,  0.27752367,  0.        ], dtype=float32), array([ 1.4153504 ,  0.        , -0.21933848], dtype=float32), array([-0.86064121,  0.58106952,  0.46581319]), array([-0.5902845, -1.3457141, -0.0020952], dtype=float32), array([-0.88190684,  0.52902799, -0.46394415]), array([ 0.6564527 ,  0.5129189 , -0.43945763], dtype=float32), array([ 0.5144202 , -0.1752086 , -0.11247836], dtype=float32), array([-1.2513721 ,  0.20881712,  0.38490066], dtype=float32), array([-1.2922395 , -0.69932467,  0.0027086 ], dtype=float32), array([-0.08729582,  1.0535041 , -0.46639776], dtype=float32), array([ 0.87190024, -1.15322416,  0.04514131]), array([-0.12209608,  1.08598316,  0.45100718]), array([ 0.83561695, -0.6457387 , -0.46657318], dtype=float32), array([-0.97001838, -0.12364659,  0.45768252]), array([-1.42351335, -0.26065172,  0.07788082]), array([-0.8640405, -0.9245347,  0.3868024], dtype=float32), array([-1.2501523 , -0.20861295,  0.38629666], dtype=float32), array([-0.9956039, -1.0815141,  0.       ], dtype=float32), array([ 1.3367101 ,  0.45889312, -0.22166905], dtype=float32), array([-0.5359651 , -0.09029113, -0.11247836], dtype=float32), array([ 1.10679014, -0.79223237,  0.27151415]), array([1.1155988 , 0.8683053 , 0.22121464], dtype=float32), array([-1.1179692 ,  0.6050143 , -0.38207388], dtype=float32), array([-0.8584474,  0.9325225,  0.3862388], dtype=float32), array([-0.77167346, -0.244889  , -0.43064534]), array([-0.94589875, -0.56783843, -0.44620277]), array([-0.56803644,  1.2949936 ,  0.22075197], dtype=float32), array([-0.10462783, -1.262663  ,  0.3868024 ], dtype=float32), array([ 0.77308476,  1.1832943 , -0.2214818 ], dtype=float32), array([-0.57593632,  0.66752086, -0.44584205]), array([ 0.77456844, -1.1855652 , -0.2184199 ], dtype=float32), array([-0.51191427, -1.16921317, -0.37605364]), array([-0.63918877, -0.10718694,  0.31166765], dtype=float32), array([1.2669904, 0.       , 0.3868024], dtype=float32), array([-0.9690869 ,  0.18556046, -0.45852109]), array([-0.21902882, -0.49744606,  0.11247836], dtype=float32), array([1.3903514, 0.4773082, 0.       ], dtype=float32), array([ 0.5771974 , -0.8849169 , -0.46657318], dtype=float32), array([-0.4934446, -1.1642011,  0.3868024], dtype=float32), array([-5.9040749e-01,  1.3459945e+00, -8.5296982e-04], dtype=float32), array([-0.9956039,  1.0815141,  0.       ], dtype=float32), array([-0.16159281, -0.87374017,  0.44742257]), array([ 0.16440773, -0.91331253,  0.45198927]), array([0.5106701 , 0.3989158 , 0.31166765], dtype=float32), array([-0.8586824 ,  0.93277776, -0.38584718], dtype=float32), array([ 0.15934184, -1.39072277, -0.20903587]), array([ 1.1983413 , -0.41139168, -0.3868024 ], dtype=float32), array([ 0.4837559 , -0.72352907,  0.4441338 ]), array([-0.32480416, -1.41067434,  0.01114059]), array([ 0.57930887,  0.8833281 , -0.46657318], dtype=float32), array([-1.2440451 , -0.67324317, -0.22026086], dtype=float32), array([ 1.09556134, -0.81583039, -0.26996241]), array([ 0.58926177, -1.25308167,  0.23284713]), array([ 0.75850756, -0.99801991,  0.38746373]), array([ 1.11017772,  0.05119485, -0.44303372]), array([ 0.1471993 ,  1.43626722, -0.02646434]), array([ 0.15050022,  0.73457994, -0.38524388]), array([ 0.8333167 , -0.64949733,  0.46657318], dtype=float32), array([ 0.25639275, -1.0245676 , -0.46657318], dtype=float32)]
#points = iterate(points, iters=100)
vcs = None
#import pickle
#with open("vcs.bin", "rb") as f:
#    vcs = pickle.load(f)
vcs = [voronoi_cell(p, points) for p in tqdm(points)]
with open("vcs.bin", "wb") as f:
    pickle.dump(vcs, f)
network = voronoi_network(r=0.02, vcs=vcs, eps=0.0001)
network.make_neighbours_symmetric()
network.save("out/stls/torus")
#network.r=0.05

#n = Network(r=0.05)
#corners = np.array(list(itertools.product((1, -1), (1, -1), (1, -1))))
#for corner in corners:
#    n.add_junction_at(corner)
#for corner in corners:
#    corner = np.array(corner)
#    n.join_junctions_at(corner, corner * np.array([-1, 1, 1]), symmetric=False)
#    n.join_junctions_at(corner, corner * np.array([1, -1, 1]), symmetric=False)
#    n.join_junctions_at(corner, corner * np.array([1, 1, -1]), symmetric=False)
#n.save("cubey")

#save_stl(n.jj_tube(n.junctions[0], n.junctions[4]), "tubey")
myplot(m, points, [voronoi_cell(points[0], points)])

