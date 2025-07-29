import numpy as np
import numba as nb

@nb.jit(nopython=True)
def compute_vn_kernel(vertices, faces):
    lerp = lambda old_val, new_val, new_weight: new_weight * new_val + (1 - new_weight) * old_val
    normals = np.zeros_like(vertices)
    counts = np.zeros_like(vertices)
    for f in faces:
        p1 = vertices[f[0]]
        p2 = vertices[f[1]]
        p3 = vertices[f[2]]
        normal = np.cross(p2 - p1, p3 - p1)
        normal = normal / np.linalg.norm(normal)
        normals[f[0]] = lerp(normals[f[0]], normal, 1/(counts[f[0]] + 1))
        normals[f[1]] = lerp(normals[f[1]], normal, 1/(counts[f[1]] + 1))
        normals[f[2]] = lerp(normals[f[2]], normal, 1/(counts[f[2]] + 1))
        counts[f[0]] += 1
        counts[f[1]] += 1
        counts[f[2]] += 1
    return normals

def compute_vn(mesh):
    """ compute vertex normals
        NOTE: for trimesh object, we recommend using this function to compute vertex normals
    
    Args:
        mesh (trimesh.Trimesh): the input mesh
    Returns:
        vertex_normals (np.ndarray): normalized vertex normals of shape==(nv, 3)
    """
    return compute_vn_kernel(mesh.vertices, mesh.faces)
