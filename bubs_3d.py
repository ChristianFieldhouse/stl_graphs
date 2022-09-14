import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
from network_to_stl import Network
from stl import mesh

from mpl_toolkits import mplot3d
from matplotlib import pyplot

m = mesh.Mesh.from_file("inputs/icosphere3.stl")

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

num_blobs = 62
num_blobx = np.sqrt(num_blobs)

points = [np.array(p) for p in [
    (0.5, 0.5, 0),
    (0.1, 0, 0),
    (1, 0.1, 0),
    (0, 0.9, 0),
    (0.9, 1, 0),
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

def repel(points, average_dist=0.05):
    forces = calculate_forces(points)
    forces = forces * average_dist / np.mean(np.abs(forces))
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
        print(insides, line.point, line.direction)
        if insides == 3:
            ins.append(intersection)
    print(ins)
    return sorted(ins, key=lambda x: np.sum((x - line.point)))

def voronoi_cell(point, points):
    points = [p for p in points if not np.all(p==point)]
    planes = [Plane((point + p)/2, point-p) for p in points]
    points_and_planes = list(zip(points, planes))
    points_and_planes = sorted(points_and_planes, key=lambda x:np.sum((x[0]-point)**2))
    genplanes = [points_and_planes[0]]
    normal = np.cross(points_and_planes[0][0]-point, points_and_planes[1][0]-point)
    #print(f"{normal=}")
    def intersect_distance_anticlockwise(q0, q1):
        p0, plane_q0 = q0
        p1, plane_q1 = q1
        n = np.cross(p0 - point, p1-point)
        wrong_direction = np.dot(normal, n) <= 0
        #print(p0, p1, n)
        if wrong_direction:
            return 1e10
        common = plane_q0 * plane_q1
        d = common.dist_to_point(point)
        return d
    while True:
        other_points_and_planes = [p for p in points_and_planes if np.any(p[0]!=genplanes[-1][0])]
        other_points_and_planes = sorted(
            other_points_and_planes,
            key=lambda x: intersect_distance_anticlockwise(genplanes[-1], x)
        )
        #print("g", genplanes)
        #print("o", other_points_and_planes)
        #print(len(other_points_and_planes), len(genplanes))
        if np.all(other_points_and_planes[0][0] == genplanes[0][0]):
            break
        genplanes.append(other_points_and_planes[0])

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
    return all_pts # + nice_loop + corners

history = []
num_img_cols = 3
for iter in tqdm(range(30)):
    history.append(points.copy())
    points = project(repel(points))


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

vc = voronoi_cell(points[0], points)
myplot(m, points, [vc])

