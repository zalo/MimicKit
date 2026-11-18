import torch

class StatsTracker():
    def __init__(self, n, device):
        self._count = torch.zeros([1], device=device, dtype=torch.long)

        self._mean = torch.zeros([n], device=device, dtype=torch.float32)
        self._mean_sq = torch.zeros([n], device=device, dtype=torch.float32)
        self._std = torch.zeros([n], device=device, dtype=torch.float32)
        return

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_count(self):
        return self._count.item()

    def reset(self):
        self._count.zero_()
        self._mean.zero_()
        self._mean_sq.zero_()
        self._std.zero_()
        return

    def update(self, xs):
        assert(xs.shape[1] == self._mean.shape[0])

        new_count = xs.shape[0]
        new_mean = torch.mean(xs, dim=0)
        new_mean_sq = torch.mean(torch.square(xs), dim=0)
        
        new_total = self._count + new_count
        w_old = self._count / new_total
        w_new = new_count / new_total

        self._mean[:] = w_old * self._mean + w_new * new_mean
        self._mean_sq[:] = w_old * self._mean_sq + w_new * new_mean_sq
        self._count[:] = new_total

        self._std[:] = self._calc_std(self._mean, self._mean_sq)
        return
    
    def _calc_std(self, mean, mean_sq):
        var = mean_sq - torch.square(mean)
        std = torch.sqrt(var)
        return std