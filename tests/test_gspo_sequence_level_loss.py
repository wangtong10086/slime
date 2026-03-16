import torch

from slime.utils.ppo_utils import (
    compute_gspo_kl,
    compute_gspo_sequence_kl,
    compute_gspo_sequence_policy_tensors,
    compute_policy_loss,
)


def test_gspo_sequence_level_kl_matches_expanded_path():
    full_log_probs = [
        torch.tensor([-0.2, -0.4, -0.1], dtype=torch.float32),
        torch.tensor([-0.3, -0.5], dtype=torch.float32),
    ]
    full_old_log_probs = [
        torch.tensor([-0.1, -0.6, -0.2], dtype=torch.float32),
        torch.tensor([-0.2, -0.4], dtype=torch.float32),
    ]
    local_log_probs = [tensor.clone() for tensor in full_log_probs]
    loss_masks = [torch.ones_like(tensor) for tensor in full_log_probs]

    expanded = compute_gspo_kl(full_log_probs, full_old_log_probs, local_log_probs, loss_masks)
    sequence_kl = compute_gspo_sequence_kl(full_log_probs, full_old_log_probs, loss_masks)

    expected = torch.cat([kl.expand_as(log_prob) for kl, log_prob in zip(sequence_kl, local_log_probs, strict=True)])

    assert torch.allclose(expanded, expected)


def test_gspo_sequence_level_policy_loss_matches_token_expansion():
    sequence_kl = [
        torch.tensor(0.1, dtype=torch.float32),
        torch.tensor(-0.05, dtype=torch.float32),
    ]
    advantages = [
        torch.tensor([0.3, -0.1, 0.2], dtype=torch.float32),
        torch.tensor([-0.2, 0.4], dtype=torch.float32),
    ]

    seq_pg_loss, seq_clipfrac = compute_gspo_sequence_policy_tensors(
        sequence_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    expanded_kl = torch.cat(
        [kl.expand_as(advantage) for kl, advantage in zip(sequence_kl, advantages, strict=True)],
        dim=0,
    )
    flat_advantages = torch.cat(advantages, dim=0)
    expanded_pg_loss, expanded_clipfrac = compute_policy_loss(
        expanded_kl,
        flat_advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    assert torch.allclose(torch.cat(seq_pg_loss, dim=0), expanded_pg_loss)
    assert torch.allclose(torch.cat(seq_clipfrac, dim=0), expanded_clipfrac)
