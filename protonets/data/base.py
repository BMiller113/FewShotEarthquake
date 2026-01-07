import torch


def convert_dict(k, v):
    return {k: v}


def _cudaify(obj):
    """
    Recursively move tensors (and tensor-like objects with .cuda) to GPU.
    Handles nested dicts/lists/tuples so episodic samples like:
      {"xs_img": {"pre": T, "post": T}, ...}
    are moved correctly.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda()
    if isinstance(obj, dict):
        return {kk: _cudaify(vv) for kk, vv in obj.items()}
    if isinstance(obj, list):
        return [_cudaify(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_cudaify(x) for x in obj)
    return obj


class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        # Recursively cudaify every value in the sample dict
        for k, v in data.items():
            data[k] = _cudaify(v)
        return data


class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]
