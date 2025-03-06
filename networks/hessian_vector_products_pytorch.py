import torch


# forward over forward
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    def compute_grad(primals):
        primals.requires_grad_(True)
        out = f(primals)
        grad = torch.autograd.grad(out, primals, create_graph=True, grad_outputs=tangents)[0]
        return grad
    
    primals_out = f(primals)
    tangents_out = torch.autograd.grad(compute_grad(primals), primals, grad_outputs=tangents)[0]
    
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# reverse over reverse
def hvp_revrev(f, primals, tangents, return_primals=False):
    primals.requires_grad_(True)
    out = f(primals)
    grad = torch.autograd.grad(out, primals, create_graph=True, grad_outputs=tangents)[0]
    tangents_out = torch.autograd.grad(grad, primals, grad_outputs=tangents)[0]
    
    if return_primals:
        return out, tangents_out
    else:
        return tangents_out


# forward over reverse
def hvp_fwdrev(f, primals, tangents, return_primals=False):
    return hvp_revrev(f, primals, tangents, return_primals)


def hvp_revfwd(f, primals, tangents, return_primals=False):
    return hvp_revrev(f, primals, tangents, return_primals)