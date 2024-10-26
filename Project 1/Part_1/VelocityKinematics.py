import sympy as sp

q = sp.symbols('q1:8')  # Joint variables 
q_dot = sp.symbols('q_dot1:8')  # Joint velocities

#  DH parameters
x_disp = [0, 0, 0.0825, -0.0825, 0, 0.088, 0]
z_disp = [0.333, 0, 0.316, 0, 0.384, 0, 0.21]
alpha = [-sp.pi/2, sp.pi/2, sp.pi/2, -sp.pi/2, sp.pi/2, sp.pi/2, 0]

# transformation matrix function
def dh_transform(a, d, alpha, theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])


T = sp.eye(4)  # Identity matrix for base
T_matrices = []  

for i in range(7):
    T_i = dh_transform(x_disp[i], z_disp[i], alpha[i], q[i])
    T = T * T_i  # Multiply the transformation matrices
    T_matrices.append(T)

# end-effector position and orientation
o_n = T[:3, 3] 
R_n = T[:3, :3]  

J_v = sp.zeros(3, 7)
J_w = sp.zeros(3, 7)

# Jacobian columns
z_prev = sp.Matrix([0, 0, 1])  # Initial z-axis 
o_prev = sp.Matrix([0, 0, 0])  # Initial origin 

for i in range(7):
    T_i = T_matrices[i]  # T matrix 
    R_i = T_i[:3, :3]  # Rotation matrix
    o_i = T_i[:3, 3]  # Position 

    z_i = R_i * sp.Matrix([0, 0, 1])  # Z-axis of the current joint

    # Linear velocity 
    J_v[:, i] = z_prev.cross(o_n - o_prev)

    # Angular velocity 
    J_w[:, i] = z_prev

    z_prev = z_i
    o_prev = o_i

# linear and angular Jacobians
J = sp.Matrix.vstack(J_v, J_w)

# Jacobian in zero configuration
zero_config = {q[i]: 0 for i in range(7)}  
J_zero_config = J.subs(zero_config)

print("Jacobian at zero configuration:")
sp.pprint(J_zero_config)