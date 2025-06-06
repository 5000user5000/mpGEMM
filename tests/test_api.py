import numpy as np
import mpgemm

def test_add_bias():
    C = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    bias = np.array([10,20,30], dtype=np.float32)
    R = mpgemm.add_bias(C.flatten().tolist(), 2, 3, bias.tolist())
    R = np.array(R).reshape(2,3)
    assert np.allclose(R, C + bias)

def test_relu():
    M = np.array([[-1,0],[2,-3]], dtype=np.float32)
    R = mpgemm.apply_activation(M.flatten().tolist(), 2, 2, mpgemm.Activation.ReLU)
    R = np.array(R).reshape(2,2)
    assert np.all(R >= 0)

def test_measure_error():
    ref  = [1.0, 2.0, 3.0]
    test = [1.1, 1.9, 2.5]
    stats = mpgemm.measure_error(ref, test)
    assert abs(stats["mse"] - 0.09) < 1e-6
    assert abs(stats["max_error"] - 0.5) < 1e-6