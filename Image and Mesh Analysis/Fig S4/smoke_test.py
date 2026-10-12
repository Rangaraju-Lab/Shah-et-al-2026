import numpy as np
import torch
import train as t

def main():
    t.set_seed(t.SEED)
    torch.set_num_threads(2)
    model = t.HierarchyConstrainedAttentionResUNet3D(base=t.BASE_CH)
    shape = (8, 32, 32)
    image = torch.randn(1, 1, *shape)
    labels = np.zeros((t.N_CLASSES, *shape), dtype=np.uint8)
    labels[t.OMM_IDX, 2:6, 8:24, 8:24] = 1
    labels[t.IBM_IDX, 3:5, 10:22, 10:22] = 1
    labels[t.IMS_IDX, 2:6, 8:10, 8:24] = 1
    labels[t.ER_IDX, :, 2:4, :] = 1
    labels[t.RIBO_IDX, 1, 5, 5] = 1
    target = torch.from_numpy(labels[None].astype(np.float32))
    boundary = torch.from_numpy(t.make_boundary_target(labels)[None])
    affinity = torch.from_numpy(t.make_affinity_target(labels)[None])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.LR, weight_decay=t.WEIGHT_DECAY)
    out = model(image)
    for key in ("seg_probs", "raw_probs", "raw_logits", "boundary_logits"):
        assert out[key].shape == (1, 5, *shape), key
    assert out["affinity_logits"].shape == (1, 12, *shape)
    p = out["seg_probs"]
    assert bool(torch.all(p[:, t.IBM_IDX] <= p[:, t.OMM_IDX]))
    assert bool(torch.all(p[:, t.IMS_IDX] <= p[:, t.OMM_IDX]))
    assert bool(torch.all(p[:, t.ER_IDX] + p[:, t.RIBO_IDX] <= 1 - p[:, t.OMM_IDX] + 1e-6))
    loss, parts = t.total_loss(out, target, torch.ones(1, 5), boundary, affinity, torch.ones(5), torch.ones(5))
    assert bool(torch.isfinite(loss)), parts
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    optimizer.step()
    count = sum(p.numel() for p in model.parameters())
    print(f"PASS: {count:,} parameters; segmentation/boundary/affinity shapes, hierarchy, loss, backward and optimizer step.")
    print("Synthetic CPU test only; original architecture and base width, smaller spatial input. No datasets or checkpoints written.")

if __name__ == "__main__":
    main()
