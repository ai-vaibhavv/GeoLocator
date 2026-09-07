import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGeoCellGeoNet(nn.Module):
    """HGNetV2-B0 backbone + two geo-cell + coordinate offset head."""

    def __init__(
        self,
        arch,
        pretrained_path,
        cell_lat,
        cell_lng,
        cell2_lat,
        cell2_lng,
        dropout=0.3,
        target_size=224,
        task_weight_init_lambda=1.0,
    ):
        """Build the network around a timm backbone and a fixed pair of partitions.

        Args:
            arch: timm architecture name for the backbone.
            pretrained_path: .pth of ImageNet weights for that backbone, or None
                to start from random initialisation.
            cell_lat, cell_lng: centroids of the coarse partition, one per cell.
            cell2_lat, cell2_lng: centroids of the fine partition.
            dropout: dropout inside the shared stem and the offset head.
            target_size: spatial size the backbone consumes; the image is resized
                to this after the detail conv.
            task_weight_init_lambda: initial value of the regression loss weight.
        """
        super().__init__()
        self.target_size = target_size
        self.log_var_reg = nn.Parameter(
            torch.tensor(math.log(float(task_weight_init_lambda)))
        )
        self.log_var_cell = nn.Parameter(torch.zeros(()))
        self.log_var_cell2 = nn.Parameter(torch.zeros(()))
        self.detail_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.detail_conv[-1].weight)
        nn.init.zeros_(self.detail_conv[-1].bias)
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        if pretrained_path is not None:
            self.backbone.load_state_dict(
                torch.load(pretrained_path, map_location="cpu")
            )
        with torch.no_grad():
            out_dim = self._features(torch.zeros(2, 3, target_size, target_size)).shape[
                1
            ]
        if out_dim == 0:
            raise ValueError(f"{arch} produced a 0-dim feature vector")
        stem_dim = 128

        def proj(width):
            """One projection block: out_dim -> width, Hardswish, Dropout."""
            return nn.Sequential(
                nn.Linear(out_dim, width),
                nn.Hardswish(inplace=True),
                nn.Dropout(dropout),
            )

        self.cell_stem = proj(stem_dim)
        self.cell2_stem = self.cell_stem
        self.cell_out = nn.Linear(stem_dim, len(cell_lat))
        self.cell2_out = nn.Linear(stem_dim, len(cell2_lat))
        self.gate = nn.Linear(out_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.offset_head = nn.Sequential(*proj(stem_dim), nn.Linear(stem_dim, 2))
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        self.register_buffer("cell_lat", torch.tensor(cell_lat, dtype=torch.float32))
        self.register_buffer("cell_lng", torch.tensor(cell_lng, dtype=torch.float32))
        self.register_buffer("cell2_lat", torch.tensor(cell2_lat, dtype=torch.float32))
        self.register_buffer("cell2_lng", torch.tensor(cell2_lng, dtype=torch.float32))

    def _features(self, x):
        return self.backbone.forward_head(
            self.backbone.forward_features(x), pre_logits=True
        )

    def forward(self, x):
        """Map a batch of images to coordinates.

        Args:
            x: a (batch, 3, H, W) tensor of normalised images.

        Returns:
            (lat, lng, cell_logits, cell2_logits, gate): the predicted
            coordinates, the coarse and fine cell logits, and the per-image
            blend weight. This 5-tuple is a contract the training loop and
            predict.py rely on.
        """
        x = x + self.detail_conv(x)
        x = F.interpolate(
            x,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        feat = self._features(x)
        self.last_feat = feat
        cell_logits = self.cell_out(self.cell_stem(feat))
        probs = torch.softmax(cell_logits, dim=1)
        cell2_logits = self.cell2_out(self.cell2_stem(feat))
        probs2 = torch.softmax(cell2_logits, dim=1)
        g = torch.sigmoid(self.gate(feat)).squeeze(1)
        anchor_lat = g * (probs2 @ self.cell2_lat) + (1.0 - g) * (probs @ self.cell_lat)
        anchor_lng = g * (probs2 @ self.cell2_lng) + (1.0 - g) * (probs @ self.cell_lng)
        offset = self.offset_head(feat)
        return (
            anchor_lat + offset[:, 0],
            anchor_lng + offset[:, 1],
            cell_logits,
            cell2_logits,
            g,
        )
