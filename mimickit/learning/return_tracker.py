import torch

import envs.base_env as base_env

class ReturnTracker():
    def __init__(self, num_envs, device):
        self._episodes = torch.zeros([1], device=device, dtype=torch.long)
        self._mean_return = torch.zeros([1], device=device, dtype=torch.float32)
        self._mean_ep_len = torch.zeros([1], device=device, dtype=torch.float32)

        self._return_buf = torch.zeros([num_envs], device=device, dtype=torch.float32)
        self._ep_len_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        self._eps_per_env_buf = torch.zeros([num_envs], device=device, dtype=torch.long)
        return

    def get_mean_return(self):
        return self._mean_return

    def get_mean_ep_len(self):
        return self._mean_ep_len

    def get_episodes(self):
        return self._episodes.item()

    def get_eps_per_env(self):
        return self._eps_per_env_buf

    def reset(self):
        self._episodes.zero_()
        self._eps_per_env_buf.zero_()

        self._mean_return.zero_()
        self._mean_ep_len.zero_()

        self._return_buf.zero_()
        self._ep_len_buf.zero_()
        return

    @torch.compile
    def update(self, reward, done):
        assert(reward.shape == self._return_buf.shape)
        assert(done.shape == self._return_buf.shape)

        self._return_buf += reward
        self._ep_len_buf += 1

        # masked updates avoid a GPU-CPU sync on every step
        reset_mask = done != base_env.DoneFlags.NULL.value
        num_resets = torch.sum(reset_mask)
        denom = torch.clamp_min(num_resets, 1)

        new_mean_return = torch.sum(torch.where(reset_mask, self._return_buf, 0.0)) / denom
        new_mean_ep_len = torch.sum(torch.where(reset_mask, self._ep_len_buf, 0)) / denom

        new_count = torch.clamp_min(self._episodes + num_resets, 1)
        w_new = num_resets / new_count
        w_old = self._episodes / new_count

        self._mean_return = w_new * new_mean_return + w_old * self._mean_return
        self._mean_ep_len = w_new * new_mean_ep_len + w_old * self._mean_ep_len
        self._episodes += num_resets

        self._return_buf.masked_fill_(reset_mask, 0.0)
        self._ep_len_buf.masked_fill_(reset_mask, 0)
        self._eps_per_env_buf += reset_mask
        return