import torch
import torch.nn.functional as F


class TokenQNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, vocab_size, action_emb_dim=256, mlp_dim=1024):
        super().__init__()
        self.action_emb = torch.nn.Embedding(vocab_size, action_emb_dim)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + action_emb_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_dim, 1),
        )

    def forward(self, obs, action_ids, action_mask=None):
        """
        obs:        [B, H]
        action_ids: [B, K]
        action_mask:[B, K] bool, optional

        returns:    [B, K]
        """
        safe_action_ids = action_ids.clamp_min(0)
        action_emb = self.action_emb(safe_action_ids)

        B, K = action_ids.shape
        obs_expanded = obs.unsqueeze(1).expand(B, K, obs.shape[-1])

        x = torch.cat([obs_expanded, action_emb], dim=-1)
        q = self.net(x).squeeze(-1)

        q = torch.sigmoid(q)

        if action_mask is not None:
            q = q.masked_fill(~action_mask, -torch.inf)

        return q


def dqn_train_step(buffer, q_net, target_q_net, optimizer, constants, device, global_train_step):

    warmup_steps = constants['WARMUP_STEPS']
    train_batch_size = constants['TRAIN_BATCH_SIZE']
    gamma = constants['GAMMA']
    grad_clip = constants['GRAD_CLIP']
    target_update_every = constants['TARGET_UPDATE_FRQ']

    if len(buffer) < warmup_steps:
        return None, global_train_step

    q_net.train()

    batch = buffer.sample(train_batch_size, to_device=device)

    obs = batch["obs"]                         # [B, H]
    actions = batch["actions"].unsqueeze(1)    # [B, 1]
    rewards = batch["rewards"]                 # [B]
    next_obs = batch["next_obs"]               # [B, H]
    dones = batch["dones"]                     # [B]

    next_action_sets = batch["next_action_sets"]             # [B, K]
    next_action_set_masks = batch["next_action_set_masks"]   # [B, K]
    next_action_set_lengths = batch["next_action_set_lengths"]

    # Q(s_t, a_t)
    q_pred = q_net(obs, actions).squeeze(1)  # [B]

    with torch.no_grad():
        # max_a' Q_target(s_{t+1}, a')
        next_q_all = target_q_net(
            next_obs,
            next_action_sets,
            next_action_set_masks,
        )  # [B, K]

        next_q_max = next_q_all.max(dim=1).values

        # Terminal states have no next action set.
        next_q_max = torch.where(
            next_action_set_lengths > 0,
            next_q_max,
            torch.zeros_like(next_q_max),
        )

        target = rewards + gamma * (1.0 - dones) * next_q_max

    loss = F.smooth_l1_loss(q_pred, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
    optimizer.step()

    if global_train_step % target_update_every == 0:
        target_q_net.load_state_dict(q_net.state_dict())

    return {
        "loss": float(loss.item()),
        "q_mean": float(q_pred.mean().item()),
        "target_mean": float(target.mean().item()),
        "reward_mean": float(rewards.mean().item()),
    }, global_train_step+1