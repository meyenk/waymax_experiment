"""
Stage 1 model. Three encoder branches (agents, map points, traffic lights),
each pooled with a learned softmax relevance weighting (NOT a plain mean --
distance/relevance is learned per item, invalid/padded items get weight ~0),
concatenated with ego conditioning (yaw, speed), decoded into the ego's
future trajectory.

All positional/velocity inputs must already be in the ego-centric frame at
time t (see transforms.py) before reaching this model.
"""

import torch
import torch.nn as nn


class RelevancePool(nn.Module):
    """Learned softmax-weighted pooling over a set of item embeddings.
    Padded/invalid items get -inf logits -> exactly zero weight."""

    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, embeddings, mask):
        # embeddings: (B, N, D), mask: (B, N) bool, True = valid
        logits = self.score(embeddings).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights)  # all-invalid row -> zeros, not NaN
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class AgentBranch(nn.Module):
    """Per-agent GRU over history + static class embedding, then relevance pool.
    cont_feat_dim=6: [x, y, cos_yaw, sin_yaw, vx_rel, vy_rel] (ego frame)."""

    def __init__(self, cont_feat_dim=6, num_classes=4, class_embed_dim=8, hidden_dim=64):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.gru = nn.GRU(cont_feat_dim, hidden_dim, batch_first=True)
        self.pool = RelevancePool(hidden_dim + class_embed_dim)
        self.out_dim = hidden_dim + class_embed_dim

    def forward(self, hist_feats, class_ids, mask):
        # hist_feats: (B, N, hist_len, cont_feat_dim), class_ids/mask: (B, N)
        B, N, T, F = hist_feats.shape
        _, h_n = self.gru(hist_feats.view(B * N, T, F))
        h_n = h_n.squeeze(0).view(B, N, -1)
        combined = torch.cat([h_n, self.class_embed(class_ids)], dim=-1)
        return self.pool(combined, mask)


class MapBranch(nn.Module):
    """Static map points (lane markings, stop signs, crosswalks, ...) -> MLP -> pool.
    No time dimension -- these don't move."""

    def __init__(self, cont_feat_dim=2, num_types=20, type_embed_dim=8, hidden_dim=64):
        super().__init__()
        self.type_embed = nn.Embedding(num_types, type_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cont_feat_dim + type_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool = RelevancePool(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, point_xy, type_ids, mask):
        # point_xy: (B, N, 2), type_ids/mask: (B, N)
        combined = torch.cat([point_xy, self.type_embed(type_ids)], dim=-1)
        return self.pool(self.mlp(combined), mask)


class TrafficLightBranch(nn.Module):
    """Per-light GRU over (stop-line position, state) across history -> pool.
    Has a time dimension -- light state changes."""

    def __init__(self, cont_feat_dim=2, num_states=8, state_embed_dim=8, hidden_dim=64):
        super().__init__()
        self.state_embed = nn.Embedding(num_states, state_embed_dim)
        self.gru = nn.GRU(cont_feat_dim + state_embed_dim, hidden_dim, batch_first=True)
        self.pool = RelevancePool(hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, hist_xy, state_ids, mask):
        # hist_xy: (B, N, hist_len, 2), state_ids: (B, N, hist_len), mask: (B, N)
        B, N, T, _ = hist_xy.shape
        combined = torch.cat([hist_xy, self.state_embed(state_ids)], dim=-1)
        _, h_n = self.gru(combined.view(B * N, T, -1))
        h_n = h_n.squeeze(0).view(B, N, -1)
        return self.pool(h_n, mask)


class Stage1Model(nn.Module):
    """Full pipeline: three branches -> concat with ego (yaw, speed) -> decoder MLP
    -> predicted ego future trajectory (future_len steps, single-shot, cumsum'd)."""

    def __init__(self, hidden_dim=64, future_len=30, ego_dim=3):
        super().__init__()
        self.agent_branch = AgentBranch(hidden_dim=hidden_dim)
        self.map_branch = MapBranch(hidden_dim=hidden_dim)
        self.light_branch = TrafficLightBranch(hidden_dim=hidden_dim)
        self.future_len = future_len
        in_dim = self.agent_branch.out_dim + self.map_branch.out_dim + self.light_branch.out_dim + ego_dim
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, future_len * 2),
        )

    def forward(self, batch):
        agent_ctx, agent_w = self.agent_branch(batch["agent_hist"], batch["agent_class"], batch["agent_mask"])
        map_ctx, map_w = self.map_branch(batch["map_xy"], batch["map_type"], batch["map_mask"])
        light_ctx, light_w = self.light_branch(batch["light_hist_xy"], batch["light_state"], batch["light_mask"])
        x = torch.cat([agent_ctx, map_ctx, light_ctx, batch["ego_vec"]], dim=-1)
        offsets = self.decoder(x).view(-1, self.future_len, 2)
        pred = torch.cumsum(offsets, dim=1)
        return pred, {"agent_w": agent_w, "map_w": map_w, "light_w": light_w}
