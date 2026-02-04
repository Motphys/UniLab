import numpy as np

## provide quat math utility from motrixsim.
def quat_rotate_inverse(quats, v):
    """
    Rotate a fixed vector v by a list of quaternions using a vectorized approach.
    Computes q^-1 * v * q (Inverse rotation).

    Parameters:
        quats (np.ndarray): Array of quaternions of shape (N, 4). Each quaternion is in [w, x, y, z] format (MuJoCo convention).
        v (np.ndarray): Fixed vector of shape (3,) to be rotated.

    Returns:
        np.ndarray: Array of rotated vectors of shape (N, 3).
    """
    # Normalize the quaternions to ensure they are unit quaternions
    # q^-1 * v * q
    w = quats[:, 0]
    im = -quats[:, 1:]  # Conjugate for inverse rotation
    cross_im_v = np.cross(im, v)
    return v + 2 * (w[:, np.newaxis] * cross_im_v + np.cross(im, cross_im_v))

def quat_mul(q1, q2):
    """
    Multiply two quaternions.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=1)

def axis_angle_to_quat(axis, angle):
    """
    Convert axis-angle to quaternion.
    """
    half_angle = angle / 2
    c = np.cos(half_angle)
    s = np.sin(half_angle)
    return np.stack([c, axis[:, 0]*s, axis[:, 1]*s, axis[:, 2]*s], axis=1)

def quat_rotate(quats, v):
    """
    Rotate a fixed vector v by a list of quaternions.
    Computes q * v * q^-1 (Standard rotation).
    """
    w = quats[:, 0]
    im = -quats[:, 1:] # Note: Definition of im part sign usually determines rotation direction
    # Standard formula for q * v * q' where q = [w, x, y, z] matches 
    # v' = v + 2*w*(q_im x v) + 2*(q_im x (q_im x v))
    # Note on sign: logic above in quat_rotate_inverse used im = -quats, implying quat convention
    # Here we use standard:
    xyz = quats[:, 1:]
    t = 2 * np.cross(xyz, v)
    return v + w[:, np.newaxis] * t + np.cross(xyz, t)
