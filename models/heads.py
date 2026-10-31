"""Segmentation heads.
  flat  : original, one score per class.
  hier  : tree over ALL classes (first hierarchical attempt).
  hier2 : normal softmax for background / efflorescence / cracks / "spalling family",
          and a small severity tree inside the spalling family only, following the
          dataset definitions: rebar exposed? -> corroded? -> Moderate vs Delamination.

Both return [B, n_classes, H, W] "logits" in class-ID order, so the loss,
metrics and evaluate.py work unchanged. (The hierarchical head returns
log-probabilities; softmax of log-probabilities gives the probabilities back.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SPALL_LEVELS = ["Minor spalling", "Moderate spalling", "Delamination"]  # severity ladder, low -> high
REBAR = "Rebar corrosion"
OTHER_DAMAGE = ["Efflorescence", "Microcrack", "Structural crack"]


class FlatHead(nn.Module):
    """Original head: one 1x1 conv, one score per class."""

    def __init__(self, in_channels, class_names):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, len(class_names), kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class HierarchicalHead(nn.Module):
    """Tree of questions (all 1x1 convs on the same decoder features):

        A  damage vs background
        B  (damage) spalling family vs other damage
        C  (other)  which one: Efflorescence / Microcrack / Structural crack
        D  (spalling) rebar exposed? -> Rebar corrosion
        E  (spalling, no rebar) level: Minor / Moderate / Delamination

    P(class) = product of the answers along its path; the 8 probabilities sum to 1.
    """

    def __init__(self, in_channels, class_names):
        super().__init__()
        self.class_names = list(class_names)
        self.idx = {n: i for i, n in enumerate(self.class_names)}
        self.other = [n for n in OTHER_DAMAGE if n in self.idx]
        self.levels = [n for n in SPALL_LEVELS if n in self.idx]
        for n in ["Background", REBAR] + self.other + self.levels:
            if n not in self.idx:
                raise ValueError(f"Hierarchical head needs class '{n}' (got {self.class_names}).")
        self.node_a = nn.Conv2d(in_channels, 1, 1)
        self.node_b = nn.Conv2d(in_channels, 1, 1)
        self.node_c = nn.Conv2d(in_channels, len(self.other), 1)
        self.node_d = nn.Conv2d(in_channels, 1, 1)
        self.node_e = nn.Conv2d(in_channels, len(self.levels), 1)

    def forward(self, x):
        ls = F.logsigmoid
        a, b, d = self.node_a(x), self.node_b(x), self.node_d(x)     # [B,1,H,W]
        c = F.log_softmax(self.node_c(x), dim=1)                     # [B,|other|,H,W]
        e = F.log_softmax(self.node_e(x), dim=1)                     # [B,|levels|,H,W]

        log_dmg = ls(a)
        log_spall = log_dmg + ls(b)               # damage AND spalling family
        log_other = log_dmg + ls(-b)              # damage AND not spalling family

        out = [None] * len(self.class_names)
        out[self.idx["Background"]] = ls(-a)[:, 0]
        out[self.idx[REBAR]] = (log_spall + ls(d))[:, 0]
        for i, n in enumerate(self.other):
            out[self.idx[n]] = (log_other + c[:, i:i + 1])[:, 0]
        for j, n in enumerate(self.levels):
            out[self.idx[n]] = (log_spall + ls(-d) + e[:, j:j + 1])[:, 0]
        return torch.stack(out, dim=1)


class SeverityHead(nn.Module):
    """hier2. Same number of parameters as the flat head.

    Step 1 (ordinary softmax, 1x1 conv): Background / Efflorescence / Microcrack /
            Structural crack / SPALLING-FAMILY.
    Step 2 (only inside the spalling family, three 1x1 convs):
            rebar exposed?  no  -> Minor spalling
                            yes -> corroded?  yes -> Rebar corrosion
                                              no  -> Moderate spalling vs Delamination (major)
    P(class) = product along its path; probabilities sum to 1.
    """

    def __init__(self, in_channels, class_names):
        super().__init__()
        self.class_names = list(class_names)
        self.idx = {n: i for i, n in enumerate(self.class_names)}
        for n in ["Background", REBAR] + SPALL_LEVELS:
            if n not in self.idx:
                raise ValueError(f"Severity head needs class '{n}' (got {self.class_names}).")
        self.top_names = ["Background"] + [n for n in OTHER_DAMAGE if n in self.idx]
        self.top = nn.Conv2d(in_channels, len(self.top_names) + 1, 1)   # last channel = spalling family
        self.exposed = nn.Conv2d(in_channels, 1, 1)
        self.corroded = nn.Conv2d(in_channels, 1, 1)
        self.major = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        ls = F.logsigmoid
        top = F.log_softmax(self.top(x), dim=1)
        log_spall = top[:, -1:]                                   # [B,1,H,W]
        exposed, corroded, major = self.exposed(x), self.corroded(x), self.major(x)

        log_exposed = log_spall + ls(exposed)
        log_clean = log_exposed + ls(-corroded)                   # exposed, not corroded

        out = [None] * len(self.class_names)
        for i, n in enumerate(self.top_names):
            out[self.idx[n]] = top[:, i]
        out[self.idx["Minor spalling"]] = (log_spall + ls(-exposed))[:, 0]
        out[self.idx[REBAR]] = (log_exposed + ls(corroded))[:, 0]
        out[self.idx["Moderate spalling"]] = (log_clean + ls(-major))[:, 0]
        out[self.idx["Delamination"]] = (log_clean + ls(major))[:, 0]
        return torch.stack(out, dim=1)


def build_head(head, in_channels, class_names):
    if head == "flat":
        return FlatHead(in_channels, class_names)
    if head == "hier":
        return HierarchicalHead(in_channels, class_names)
    if head == "hier2":
        return SeverityHead(in_channels, class_names)
    raise ValueError(f"Unknown head '{head}'. Choose 'flat', 'hier' or 'hier2'.")
