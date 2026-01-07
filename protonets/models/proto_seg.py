import torch
import torch.nn as nn
import torch.nn.functional as F

from protonets.models import register_model


class SmallSegEncoder(nn.Module):
    """
    Convolutional encoder.
    Produces feature maps B x F x H x W (same spatial size as input).
    """
    def __init__(self, in_ch: int, feat_dim: int = 64, hid_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hid_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # B,F,H,W


class ProtoSeg(nn.Module):
    """
    Prototypical Few-Shot Segmentation (episode-based).

    Handles both:
      - dataset returns episodic tensors [way, shot/query, C, H, W]
      - DataLoader adds an outer batch dim -> [B, way, shot/query, C, H, W]
    """

    def __init__(self,
                 feat_dim: int = 64,
                 hid_dim: int = 64,
                 use_two_stream: bool = True,
                 in_ch_single: int = 3):
        super().__init__()
        self.use_two_stream = use_two_stream
        self.in_ch_single = in_ch_single
        self.feat_dim = feat_dim

        if self.use_two_stream:
            self.encoder = SmallSegEncoder(in_ch=in_ch_single, feat_dim=feat_dim, hid_dim=hid_dim)
        else:
            self.encoder = SmallSegEncoder(in_ch=2 * in_ch_single, feat_dim=feat_dim, hid_dim=hid_dim)

    @staticmethod
    def _flatten_episode(x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - 6D: [B, way, n, C, H, W] -> [B*way*n, C, H, W]
          - 5D: [way, n, C, H, W]    -> [way*n, C, H, W]
          - 4D: [B, C, H, W]         -> unchanged
        """
        if x.ndim == 6:
            b, way, n, c, h, w = x.shape
            return x.reshape(b * way * n, c, h, w)
        if x.ndim == 5:
            way, n, c, h, w = x.shape
            return x.reshape(way * n, c, h, w)
        if x.ndim == 4:
            return x
        raise RuntimeError(
            f"Unexpected tensor rank {x.ndim} for image tensor (expected 4D/5D/6D). Shape={tuple(x.shape)}"
        )

    @staticmethod
    def _sanitize_class_ids(class_ids) -> torch.Tensor:
        """
        Accepts:
          - torch.Tensor [way] or [B,way] or any shape containing way ids
          - python list [way] or [[way]]
        Returns:
          - torch.LongTensor [way]
        """
        if isinstance(class_ids, torch.Tensor):
            t = class_ids.detach()
            if t.dtype != torch.long:
                t = t.long()
            return t.view(-1)
        if isinstance(class_ids, list):
            if len(class_ids) > 0 and isinstance(class_ids[0], list):
                flat = [int(x) for x in class_ids[0]]
            else:
                flat = [int(x) for x in class_ids]
            return torch.tensor(flat, dtype=torch.long)
        raise TypeError(f"Unsupported type for class_ids: {type(class_ids)}")

    @staticmethod
    def _build_episode_label_map(class_ids_1d: torch.Tensor) -> dict:
        class_ids_1d = class_ids_1d.view(-1)
        class_ids_list = [int(x) for x in class_ids_1d.cpu().tolist()]
        return {cid: i for i, cid in enumerate(class_ids_list)}

    @staticmethod
    def _map_mask_to_episode(mask: torch.Tensor, cid_to_ep: dict, ignore_index: int = 255) -> torch.Tensor:
        out = torch.full_like(mask, fill_value=ignore_index)
        for cid, epi in cid_to_ep.items():
            out[mask == cid] = epi
        return out

    @staticmethod
    def _flatten_mask_episode(mask: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - 5D: [B, way, n, H, W] -> [B*way*n, H, W]
          - 4D: [way, n, H, W]    -> [way*n, H, W]
          - 3D: [B, H, W]         -> unchanged
        """
        if mask.ndim == 5:
            b, way, n, h, w = mask.shape
            return mask.reshape(b * way * n, h, w)
        if mask.ndim == 4:
            way, n, h, w = mask.shape
            return mask.reshape(way * n, h, w)
        if mask.ndim == 3:
            return mask
        raise RuntimeError(
            f"Unexpected mask rank {mask.ndim} for mask tensor (expected 3D/4D/5D). Shape={tuple(mask.shape)}"
        )

    def _encode(self, img):
        """
        Encodes either:
          - dict {"pre": ..., "post": ...}
          - concatenated tensor
        """
        if isinstance(img, dict):
            pre = self._flatten_episode(img["pre"])
            post = self._flatten_episode(img["post"])
            pre_f = self.encoder(pre)
            post_f = self.encoder(post)
            return (pre_f + post_f) * 0.5

        x = self._flatten_episode(img)
        return self.encoder(x)

    def loss(self, sample):
        class_ids = self._sanitize_class_ids(sample["class_ids"])
        cid_to_ep = self._build_episode_label_map(class_ids)

        xs_img = sample["xs_img"]
        xs_msk = sample["xs_msk"]
        xq_img = sample["xq_img"]
        xq_msk = sample["xq_msk"]

        # Encode support/query images
        xs_feat = self._encode(xs_img)  # [Bs, F, H, W]
        xq_feat = self._encode(xq_img)  # [Bq, F, H, W]

        # Flatten masks to match
        xs_msk_flat = self._flatten_mask_episode(xs_msk)  # [Bs,H,W]
        xq_msk_flat = self._flatten_mask_episode(xq_msk)  # [Bq,H,W]

        way = int(class_ids.numel())

        ignore_index = 255
        yq = self._map_mask_to_episode(xq_msk_flat, cid_to_ep, ignore_index=ignore_index)  # [Bq,H,W]

        # Build prototypes
        prototypes = []
        for epi in range(way):
            cid = int(class_ids[epi].item())
            mask = (xs_msk_flat == cid).float()  # [Bs,H,W]
            denom = mask.sum().clamp(min=1.0)

            masked = xs_feat * mask.unsqueeze(1)      # [Bs,F,H,W]
            proto = masked.sum(dim=(0, 2, 3)) / denom # [F]
            prototypes.append(proto)

        prototypes = torch.stack(prototypes, dim=0)  # [way,F]

        # Pixel-wise logits by distance to prototypes
        Bq, Fd, H, W = xq_feat.shape
        feat_pix = xq_feat.permute(0, 2, 3, 1).contiguous().view(-1, Fd)  # [Bq*H*W,F]

        dist = (feat_pix.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)  # [Bq*H*W,way]
        logits = (-dist).view(Bq, H, W, way).permute(0, 3, 1, 2).contiguous()       # [Bq,way,H,W]

        loss_val = F.cross_entropy(logits, yq, ignore_index=ignore_index)

        with torch.no_grad():
            pred = logits.argmax(dim=1)  # [Bq,H,W]
            valid = (yq != ignore_index)
            if valid.any():
                acc_val = (pred[valid] == yq[valid]).float().mean()
            else:
                acc_val = torch.tensor(0.0, device=logits.device)

        return loss_val, {"loss": float(loss_val.item()), "acc": float(acc_val.item())}


@register_model("protonet_seg")
def load_protonet_seg(**kwargs):
    feat_dim = int(kwargs.get("feat_dim", 64))
    hid_dim = int(kwargs.get("hid_dim", 64))
    use_two_stream = bool(kwargs.get("use_two_stream", True))
    in_ch_single = int(kwargs.get("in_ch_single", 3))

    return ProtoSeg(
        feat_dim=feat_dim,
        hid_dim=hid_dim,
        use_two_stream=use_two_stream,
        in_ch_single=in_ch_single,
    )
