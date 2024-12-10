import numpy as np

def tanh(s):
    return np.tanh(s)

def tanh_deriv(s):
    return 1.0 - np.tanh(s)**2

def identity(s):
    return s

def identity_deriv(s):
    return np.ones_like(s)

def sign_func(s):
    return np.where(s >= 0, 1.0, -1.0)

def forward_pass(x, W1, W2, output_activation='tanh'):
    z_h = W1.T @ x  # (2x3)*(3x1) = (2x1)
    a_h = tanh(z_h) # hidden activation (2x1)

    a_h_with_bias = np.vstack(([1.0], a_h)) # (3x1)

    z_o = W2.T @ a_h_with_bias  # (1x3)*(3x1) = (1x1)

    if output_activation == 'tanh':
        a_o = tanh(z_o)
    elif output_activation == 'identity':
        a_o = identity(z_o)
    elif output_activation == 'sign':
        a_o = sign_func(z_o)
    else:
        raise ValueError("Unknown output activation")
    
    return z_h, a_h, z_o, a_o

def compute_error(a_o, y):
    return ((a_o - y)**2) / 4.0

def backprop(x, y, W1, W2, output_activation='tanh'):
    z_h, a_h, z_o, a_o = forward_pass(x, W1, W2, output_activation)

    if output_activation == 'tanh':
        output_deriv = tanh_deriv(z_o)
    elif output_activation == 'identity':
        output_deriv = identity_deriv(z_o)
    else:
        raise ValueError("For this part, use 'tanh' or 'identity'")

    delta_o = (a_o - y) * output_deriv / 2.0

    W2_no_bias = W2[1:,:]
    delta_h = (W2_no_bias * delta_o) * tanh_deriv(z_h) # (2x1)

    a_h_with_bias = np.vstack(([1.0], a_h)) # (3x1)
    G2 = a_h_with_bias * delta_o  # element-wise, results in (3x1)

    G1 = np.zeros_like(W1)
    for j in range(W1.shape[1]):
        G1[:, j] = x.flatten() * delta_h[j]

    return G1, G2, a_o

def numerical_gradient_check(x, y, W1, W2, output_activation='tanh', eps=1e-4):
    def f(W1_, W2_):
        _, _, _, a_o = forward_pass(x, W1_, W2_, output_activation)
        return compute_error(a_o, y)
    
    G1_num = np.zeros_like(W1)
    G2_num = np.zeros_like(W2)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_pos = W1.copy()
            W1_neg = W1.copy()
            W1_pos[i,j] += eps
            W1_neg[i,j] -= eps
            E_pos = f(W1_pos, W2)
            E_neg = f(W1_neg, W2)
            G1_num[i,j] = (E_pos - E_neg) / (2*eps)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_pos = W2.copy()
            W2_neg = W2.copy()
            W2_pos[i,j] += eps
            W2_neg[i,j] -= eps
            E_pos = f(W1, W2_pos)
            E_neg = f(W1, W2_neg)
            G2_num[i,j] = (E_pos - E_neg) / (2*eps)

    return G1_num, G2_num

if __name__ == "__main__":
    x = np.array([[1.0],  # bias
                  [2.0],
                  [1.0]])  # input
    y = -1.0
    
    W1 = np.ones((3,2)) * 0.15
    W2 = np.ones((3,1)) * 0.15

    print("===== Using Tanh Output Node =====")
    G1, G2, a_o = backprop(x, y, W1, W2, output_activation='tanh')
    print("Analytic Gradients:")
    print("G1:\n", G1)
    print("G2:\n", G2)

    G1_num, G2_num = numerical_gradient_check(x, y, W1, W2, output_activation='tanh', eps=1e-4)
    print("Numerical Gradients:")
    print("G1_num:\n", G1_num)
    print("G2_num:\n", G2_num)

    print("\n===== Using Identity Output Node =====")
    G1_id, G2_id, a_o_id = backprop(x, y, W1, W2, output_activation='identity')
    print("Analytic Gradients:")
    print("G1:\n", G1_id)
    print("G2:\n", G2_id)

    G1_num_id, G2_num_id = numerical_gradient_check(x, y, W1, W2, output_activation='identity', eps=1e-4)
    print("Numerical Gradients:")
    print("G1_num:\n", G1_num_id)
    print("G2_num:\n", G2_num_id)
