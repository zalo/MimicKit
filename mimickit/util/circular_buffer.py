import torch

class CircularBuffer():
    def __init__(self, batch_size, buffer_len, shape, dtype, device):
        self._buffer = torch.zeros([batch_size, buffer_len] + list(shape), dtype=dtype, device=device)
        # the head is a tensor that is updated in-place instead of a python int, since torch.compile
        # would otherwise specialize (and recompile) on every value of the head
        self._head = torch.zeros([1], dtype=torch.long, device=device)
        self._buffer_indices = torch.arange(buffer_len, dtype=torch.long, device=device)
        self._batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        return

    def get_batch_size(self):
        return self._buffer.shape[0]

    def get_buffer_len(self):
        return self._buffer.shape[1]

    def push(self, data):
        self._buffer.index_copy_(1, self._head, data.unsqueeze(1))
        n = self.get_buffer_len()
        self._head.add_(1).remainder_(n)
        return

    def fill(self, batch_idx, data):
        ordered_indices = self._calc_ordered_indices()
        self._buffer[batch_idx.unsqueeze(-1), ordered_indices] = data
        return

    def fill_masked(self, batch_mask, data):
        ordered_indices = self._calc_ordered_indices()
        batch_mask = batch_mask.view([-1] + [1] * (self._buffer.dim() - 1))
        curr_data = self._buffer[:, ordered_indices]
        self._buffer[:, ordered_indices] = torch.where(batch_mask, data, curr_data)
        return

    def get(self, idx):
        n = self.get_buffer_len()
        buffer_idx = torch.remainder(self._head + idx, n)

        if (torch.is_tensor(idx)):
            data = self._buffer[self._batch_indices, buffer_idx]
        else:
            data = self._buffer[:, buffer_idx[0], ...]

        return data

    def get_all(self):
        ordered_indices = self._calc_ordered_indices()
        data = self._buffer[:, ordered_indices, ...]
        return data

    def reset(self):
        self._head.zero_()
        return

    def _calc_ordered_indices(self):
        n = self.get_buffer_len()
        buffer_idx = torch.remainder(self._head + self._buffer_indices, n)
        return buffer_idx
