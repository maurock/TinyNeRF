import torch

def to_homogeneous(x):
    """
    Add a 1 as the last coordinate of a tensor
    """
    return torch.cat((x, torch.ones(size=(x.shape[0], 1), device=x.device)), axis=1)

def log(log_file, string):
    """
    Log a string to a .txt file
    """
    print(string)
    with open(log_file, 'a') as f:
        f.write(string + '\n')

def batchify(rays_o, rays_d):
    """
    Split rays into batches to avoid memory issues.
    """
    rays_d_batch = torch.split(rays_d, 10000, dim=0)

    return rays_d_batch