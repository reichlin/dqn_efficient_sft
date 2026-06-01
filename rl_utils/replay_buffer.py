import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, max_action_set_size, device="cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.max_action_set_size = max_action_set_size
        self.obs_shape = obs_shape

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)

        # Here actions are token IDs, not indices into the candidate set.
        self.actions = torch.empty((capacity,), dtype=torch.long, device=self.device)
        self.rewards = torch.empty((capacity,), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity,), dtype=torch.float32, device=self.device)

        self.action_sets = torch.full((capacity, max_action_set_size), -1, dtype=torch.long, device=self.device)
        self.action_set_masks = torch.zeros((capacity, max_action_set_size), dtype=torch.bool, device=self.device)
        self.action_set_lengths = torch.zeros((capacity,), dtype=torch.long, device=self.device)

        self.next_action_sets = torch.full((capacity, max_action_set_size), -1, dtype=torch.long, device=self.device)
        self.next_action_set_masks = torch.zeros((capacity, max_action_set_size), dtype=torch.bool, device=self.device)
        self.next_action_set_lengths = torch.zeros((capacity,), dtype=torch.long, device=self.device)

        self.ptr = 0
        self.size = 0

    def _as_tensor(self, x, dtype):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def _write_action_sets(self, rows, action_sets, storage, masks, lengths):
        for local_i, row in enumerate(rows.tolist()):
            storage[row].fill_(-1)
            masks[row].fill_(False)
            lengths[row] = 0

            aset = action_sets[local_i]
            aset = self._as_tensor(aset, torch.long).flatten()

            n = min(aset.numel(), self.max_action_set_size)
            if n > 0:
                storage[row, :n] = aset[:n]
                masks[row, :n] = True
                lengths[row] = n

    def add_batch(self, obs, action_sets, actions, rewards, next_obs, next_action_sets, dones):
        obs = self._as_tensor(obs, torch.float32)
        next_obs = self._as_tensor(next_obs, torch.float32)
        actions = self._as_tensor(actions, torch.long).flatten()
        rewards = self._as_tensor(rewards, torch.float32).flatten()
        dones = self._as_tensor(dones, torch.float32).flatten()

        batch_size = obs.shape[0]
        assert batch_size <= self.capacity, "Batch larger than replay capacity."

        rows = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity

        self.obs[rows] = obs
        self.next_obs[rows] = next_obs
        self.actions[rows] = actions
        self.rewards[rows] = rewards
        self.dones[rows] = dones

        self._write_action_sets(
            rows,
            action_sets,
            self.action_sets,
            self.action_set_masks,
            self.action_set_lengths,
        )
        self._write_action_sets(
            rows,
            next_action_sets,
            self.next_action_sets,
            self.next_action_set_masks,
            self.next_action_set_lengths,
        )

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, to_device=None):
        assert self.size > 0, "Cannot sample from an empty replay buffer."

        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],

            "action_sets": self.action_sets[idx],
            "action_set_masks": self.action_set_masks[idx],
            "action_set_lengths": self.action_set_lengths[idx],

            "next_action_sets": self.next_action_sets[idx],
            "next_action_set_masks": self.next_action_set_masks[idx],
            "next_action_set_lengths": self.next_action_set_lengths[idx],
        }

        if to_device is not None:
            to_device = torch.device(to_device)
            batch = {k: v.to(to_device, non_blocking=True) for k, v in batch.items()}

        return batch

    def __len__(self):
        return self.size