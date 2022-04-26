import torch
import numpy as np
from torch.autograd import grad
from torch import nn


def hessian_vector_product(ys, params, vs, params2=None):
    grads1 = grad(ys, params, create_graph=True)
    if params2 is not None:
        params = params2
    grads2 = grad(grads1, params, grad_outputs=vs)
    return grads2


def hessians(ys, params):
    jacobians = grad(ys, params, create_graph=True)
    

    outputs = []
    for j, param in zip(jacobians, params):
        hess = []
        j_flat = j.flatten()
        for i in range(len(j_flat)):
            grad_outputs = torch.zeros_like(j_flat)
            grad_outputs[i] = 1
            grad2 = grad(j_flat, param, grad_outputs=grad_outputs, retain_graph=True)[0]
            hess.append(grad2)
        outputs.append(torch.stack(hess).reshape(j.shape + param.shape))
    return outputs