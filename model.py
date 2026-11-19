"""Neuro-GRAPE Model Entrypoint (Member 2 — Phase 5 Handoff).

Re-exports TrimodalNeuroGrapeModel and NeuroGrapeModel from src.models.model
with full interface and tensor dimension documentation.

Tensor Dimensions
-----------------
EEG:  (B, 63, 225) float32       -> (B, 128) L2-normalized
MEG:  (B, 272, 225) float32      -> (B, 128) L2-normalized
fMRI: (B, 1, 96, 96, 60) float32 -> (B, 128) L2-normalized
"""

from __future__ import annotations

from src.models.encoders import EEGEncoder, FMRIEncoder, MEGEncoder
from src.models.model import NeuroGrapeModel, TrimodalNeuroGrapeModel

__all__ = [
    "TrimodalNeuroGrapeModel",
    "NeuroGrapeModel",
    "EEGEncoder",
    "MEGEncoder",
    "FMRIEncoder",
]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch
    import torch.nn.functional as F

    REPO_ROOT = Path(__file__).resolve().parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.data.dataset import NeuroGrapeDataset
    from src.models.training import prepare_cross_modal_batch
    from src.models.losses import trimodal_contrastive_loss

    print("=" * 70)
    print("Neuro-GRAPE Trimodal Neural Representation Model Execution")
    print("=" * 70)

    # 1. Instantiate Model
    embed_dim = 128
    model = TrimodalNeuroGrapeModel(embed_dim=embed_dim)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eeg_params = sum(p.numel() for p in model.encoders["eeg"].parameters())
    meg_params = sum(p.numel() for p in model.encoders["meg"].parameters())
    fmri_params = sum(p.numel() for p in model.encoders["fmri"].parameters())

    print(f"\n[Architecture Summary]")
    print(f"  Embedding Dimension : {embed_dim}")
    print(f"  Total Parameters    : {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  EEG Encoder Params  : {eeg_params:,} (Input: [B, 63, 225])")
    print(f"  MEG Encoder Params  : {meg_params:,} (Input: [B, 272, 225])")
    print(f"  fMRI Encoder Params : {fmri_params:,} (Input: [B, 1, 96, 96, 60])")

    # 2. Load Checkpoint
    ckpt_path = REPO_ROOT / "checkpoints" / "baseline.pt"
    if ckpt_path.exists():
        ckpt = model.load_checkpoint(ckpt_path)
        print(f"\n[Checkpoint Loaded]")
        print(f"  Source file         : {ckpt_path.name}")
        if "epoch" in ckpt:
            print(f"  Trained Epochs      : {ckpt.get('epoch')}")
        if "val_loss" in ckpt:
            print(f"  Validation Loss     : {ckpt.get('val_loss'):.6f}")
    else:
        print(f"\n[Notice] No checkpoint found at {ckpt_path}; using initialized weights.")

    # 3. Load Multimodal Dataset
    inventory_path = REPO_ROOT / "data_inventory.csv"
    if inventory_path.exists():
        for split in ("test", "val", "train"):
            try:
                ds = NeuroGrapeDataset(manifest=inventory_path, split=split)
                batch, stim_ids = prepare_cross_modal_batch(ds)
                b_size = len(stim_ids["eeg"])

                print(f"\n[Running Forward Pass on '{split}' Split (N={b_size} Stimuli)]")
                active_mods = [m for m in ("eeg", "meg", "fmri") if m in batch]
                print(f"  Input Tensors:")
                for mod in active_mods:
                    print(f"    - {mod.upper():4s}: shape={tuple(batch[mod].shape)}, dtype={batch[mod].dtype}")

                with torch.no_grad():
                    embeddings = model(batch)
                    loss = trimodal_contrastive_loss(embeddings, stim_ids)

                print(f"  Output Latent Embeddings (L2 Sphere):")
                for mod in active_mods:
                    emb = embeddings[mod]
                    norms = torch.norm(emb, p=2, dim=-1)
                    is_unit = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
                    print(f"    - {mod.upper():4s}: shape={tuple(emb.shape)}, L2 norm={norms.mean().item():.4f} (unit_norm={is_unit})")

                # Cross-modal pairwise cosine similarities
                print(f"  Cross-Modal Pairwise Alignment (Mean Diagonal Cosine Sim):")
                from itertools import combinations
                for m1, m2 in combinations(active_mods, 2):
                    sim = F.cosine_similarity(embeddings[m1], embeddings[m2]).mean().item()
                    print(f"    - {m1.upper()} <-> {m2.upper()}  : {sim:+.4f}")
                print(f"  Contrastive Loss : {loss.item():.4f}")
                break
            except Exception as e:
                print(f"Could not load split '{split}': {e}")
    else:
        # Mock forward pass
        print("\n[Running Mock Forward Pass]")
        b_size = 4
        mock_batch = {
            "eeg": torch.randn(b_size, 63, 225),
            "meg": torch.randn(b_size, 272, 225),
            "fmri": torch.randn(b_size, 1, 96, 96, 60),
        }
        with torch.no_grad():
            embeddings = model(mock_batch)
            for mod in ("eeg", "meg", "fmri"):
                emb = embeddings[mod]
                print(f"  {mod.upper()}: shape={tuple(emb.shape)}")

    print("\n" + "=" * 70)
    print("Model execution completed successfully.")
    print("=" * 70)

