
import numpy as np
import itertools

def triangle_ascii(triangle):
    v1, v2, v3 = triangle
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / sum(normal**2)
    return (
        f"facet normal {normal[0]} {normal[1]} {normal[2]} \n\touter loop\n" +
        f"\t\tvertex {v1[0]} {v1[1]} {v1[2]}\n" +
        f"\t\tvertex {v2[0]} {v2[1]} {v2[2]}\n" +
        f"\t\tvertex {v3[0]} {v3[1]} {v3[2]}\n" +
        f"\tendloop\n" + f"endfacet\n"
    )

def vertex_ring(p, n, r, k=10, v3=(0, 0, 1)):
    b1 = np.cross(v3, n)
    b1 = b1 / np.sqrt(np.sum(b1**2))
    b2 = np.cross(b1, n)
    b2 = b2 / np.sqrt(np.sum(b2**2))
    return [
        p + r * (b1 * np.sin(theta) + b2 * np.cos(theta))
        for theta in np.arange(0, 2*np.pi, 2*np.pi/k)
    ]
 

def tube(path, n0=None, n1=None, closed=False, r=1, k=10):

    normals = [
        path[i+1] - path[i-1]
        for i in range(len(path) - 1)
    ] + [path[0] - path[-2]]
    
    if n0 is not None:
        normals[0] = n0
    if n1 is not None:
        normals[-1] = n1
    
    vertex_rings = [
        vertex_ring(p, n, r, k)
        for p, n in zip(path, normals)
    ]
    
    triangles = []

    vertex_rings = vertex_rings + [vertex_rings[0] if closed else vertex_rings[-1]]
    for i in range(len(path)):
        r1, r2 = vertex_rings[i], vertex_rings[i+1]
        rotate = np.argmin(np.sum((r2 - r1[0])**2, axis=1))
        r2 = np.roll(r2, -rotate, axis=0)
        r1, r2 = r1 + [r1[0]], list(r2) + [r2[0]]
        for j in range(k):
            triangles.append((r1[j], r1[j+1], r2[j]))
            triangles.append((r1[j + 1], r2[j+1], r2[j]))
    return triangles

def circle_path(r, k):
    x = np.array((1, 0, 0))
    y = np.array((0, 1, 1))
    return [
        r * (x * np.sin(theta) + y * np.cos(theta))
        for theta in np.arange(0, 2*np.pi, 2*np.pi/k)
    ]

def save_sdl(triangles, name="new"):
    with open(name + ".stl", "w") as f:
        f.write("solid " + name + "\n")
        for triangle in triangles:
            f.write(triangle_ascii(triangle))
        f.write("endsolid")

class Junction():
    def __init__(self, center, neighbours=None, radius=1, k=10):
        self.center = center
        self.neighbours = neighbours
        self.radius = radius
        self.k = k
    
    def v(self):
        """local coordinate system"""
        vx = self.neighbours[0] - self.center
        vx = vx / np.sqrt(np.sum(vx**2))
        vz = np.cross(vx, self.neighbours[1] - self.center)
        vz = vz / np.sqrt(np.sum(vz**2))
        vy = np.cross(vx, vz)
        vy = vy / np.sqrt(np.sum(vy**2))
        return vx, vy, vz
    
    def arrows(self):
        """get the directions the tubes come out, and the centers of the rings"""
        vx, vy, vz = self.v()
        num_neighbours = len(self.neighbours)
        
        angle_between_rings = 2 * np.pi / num_neighbours
        n_gon_radius = self.radius / np.sin(angle_between_rings/2)
        n_gon_centers_radius = self.radius / np.tan(angle_between_rings/2)
        
        directions = [
            (np.cos(theta) * vx + np.sin(theta) * vy)
            for theta in [n * 2 * np.pi / num_neighbours for n in range(num_neighbours)]
        ]

        return [
            (direction, self.center + n_gon_centers_radius * direction)
            for direction in directions
        ]
    
    def triangles(self):
        assert len(self.neighbours) == 3, "Junction not implemented for neighbours != 3"
        _, _, vz = self.v()
        rings = [
            vertex_ring(
                position,
                direction,
                r=self.radius,
                v3=vz,
                k=self.k,
            )
            for direction, position in self.arrows()
        ]
        triangles = []
        for i, ring in enumerate(rings):
            next_ring = rings[(len(rings) + i - 1) % len(rings)]
            for j, vert in enumerate(ring[:len(ring)//2]):
                next_vert = ring[(j + 1) % len(ring)]
                next_ring_verts = [
                    next_ring[(len(ring) - j) % len(ring)],
                    next_ring[(len(ring) - j - 1) % len(ring)],
                ]
                triangles.append((vert, next_vert, next_ring_verts[0]))
                triangles.append((next_vert, next_ring_verts[0], next_ring_verts[1]))
        
        
        # todo : generalise away from 3!!
        m = self.k // 2
        triangles.append((rings[0][0], rings[1][0], rings[2][0]))
        triangles.append((rings[0][m], rings[1][m], rings[2][m]))
        return triangles
      

class Network():
    def __init__(self, nodes):
        self.junctions = []
        self.tubes = []
        self.add_nodes(nodes)
    
    def all_triangles(self):
        triangles = []
        for junction in self.junctions:
            triangles += junction.triangles()
            for neighbour in junction.neighbours:
                triangles += self.jj_tube(junction, self.get_junction_at(neighbour))
        return triangles
    
    def get_junction_at(self, center):
        matches = [junc for junc in self.junctions if np.all(center == junc.center)]
        if not matches:
            return None
        assert len(matches) == 1, len(matches)
        return matches[0]
    
    def add_nodes(self, nodes):
        nodes = np.array(nodes)
        for i, node in enumerate(nodes):
            self.junctions.append(Junction(node))
        
        for i, node in enumerate(nodes):
            closest = sorted(nodes, key=lambda x: np.sum((x - node)**2))
            junc = self.get_junction_at(node)
            junc.neighbours = closest[1:4]

    def jj_tube(self, j0, j1):
        j01_idx = np.argmax([np.dot(a[0], j1.center - j0.center) for a in j0.arrows()])
        j01_dir, j01_start = j0.arrows()[j01_idx]
        
        j10_idx = np.argmax([np.dot(a[0], j0.center - j1.center) for a in j1.arrows()])
        j10_dir, j10_start = j1.arrows()[j10_idx]
        
        path = np.linspace(j01_start, j10_start, 2)
        return tube(path, n0=j01_dir, n1=-j10_dir)

    def save(self, name="unnamed_network"):
        save_sdl(self.all_triangles(), name)

def hack(point):
    r = 50
    angle = point[0] / r
    return np.array((
        r * np.cos(angle), r * np.sin(angle), point[1]
    ))

if __name__ == "__main__":
    #save_sdl(tube(circle_path(5, 10), k=20))
    x_step = np.array((np.sqrt(3), 0, 0)) * 10
    y_step = np.array((0, 1, 0)) * 10
    cross_step = (y_step + x_step)/2
    points = [
        hack((x - y//4)*x_step + (y//2) * y_step + cross_step*((y + 1)//2))
        for x, y in itertools.product(range(18), range(20))
    ]
    save_sdl(Network(points).all_triangles(), name="junction")
