"""Dependency-free SVG charts of the spectrum (written to results/core_spectrum/plots/).

Three projections of the 4-D frontier: drift vs RAM, decode vs RAM, prefill vs decode.
Entries on the live frontier are highlighted; superseded winners stay visible in grey.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .core_frontier import Spectrum

W, H, ML, MR, MT, MB = 640, 420, 70, 24, 44, 56
BG, TEXT, SUBTLE = "#0a0c0f", "#e9ecf0", "#8a96a6"
FRONT, OLD, GRID = "#00E5FF", "#5d6773", "#1e242c"


@dataclass(frozen=True)
class Panel:
    slug: str
    title: str
    x: str
    y: str
    xlabel: str
    ylabel: str
    ylog: bool = False
    xlog: bool = False


PANELS = (
    Panel("accuracy_vs_ram", "Drift vs memory (lower-left is better)", "ram_gib", "rp_kl",
          "peak RAM (GiB)", "RP-KL drift (log)", ylog=True),
    Panel("decode_vs_ram", "Decode speed vs memory (upper-left is better)", "ram_gib", "gen_rel",
          "peak RAM (GiB)", "decode speed vs baseline"),
    Panel("prefill_vs_decode", "Prefill vs decode (upper-right is better)", "gen_rel", "prefill_rel",
          "decode speed vs baseline", "prefill speed vs baseline"),
)


def _scale(values: list[float], log: bool, lo_px: float, hi_px: float):
    vals = [math.log10(v) if log else v for v in values]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or (abs(hi) * 0.05 or 0.5)
    lo, hi = lo - pad, hi + pad
    return (lambda v: lo_px + ((math.log10(v) if log else v) - lo) / (hi - lo) * (hi_px - lo_px)), lo, hi


def _ticks(lo: float, hi: float, log: bool, n: int = 5) -> list[tuple[float, str]]:
    out = []
    for i in range(n):
        t = lo + (hi - lo) * i / (n - 1)
        v = 10 ** t if log else t
        out.append((v, f"{v:.3g}"))
    return out


def render_panel(spectrum: Spectrum, panel: Panel) -> str:
    entries = spectrum.entries
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="system-ui, sans-serif" font-size="12">'
            f'<rect width="{W}" height="{H}" rx="14" fill="{BG}"/>'
            f'<text x="{W / 2}" y="26" text-anchor="middle" font-size="15" font-weight="600" fill="{TEXT}">{escape(panel.title)}</text>')
    if not entries:
        return head + f'<text x="{W / 2}" y="{H / 2}" text-anchor="middle" fill="{SUBTLE}">no entries yet</text></svg>'

    active = {e.point.candidate_id for e in spectrum.active()}
    xs = [getattr(e.point, panel.x) for e in entries]
    ys = [getattr(e.point, panel.y) for e in entries]
    fx, xlo, xhi = _scale(xs, panel.xlog, ML, W - MR)
    fy, ylo, yhi = _scale(ys, panel.ylog, H - MB, MT)
    parts = [head]
    for v, label in _ticks(xlo, xhi, panel.xlog):
        x = fx(v)
        parts.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{H - MB}" stroke="{GRID}"/>'
                     f'<text x="{x:.1f}" y="{H - MB + 16}" text-anchor="middle" fill="{SUBTLE}">{label}</text>')
    for v, label in _ticks(ylo, yhi, panel.ylog):
        y = fy(v)
        parts.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{W - MR}" y2="{y:.1f}" stroke="{GRID}"/>'
                     f'<text x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{SUBTLE}">{label}</text>')
    parts.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 14}" text-anchor="middle" fill="{TEXT}">{escape(panel.xlabel)}</text>')
    parts.append(f'<text transform="translate(16 {(MT + H - MB) / 2}) rotate(-90)" text-anchor="middle" fill="{TEXT}">'
                 f'{escape(panel.ylabel)}</text>')

    ordered = sorted(entries, key=lambda e: e.point.candidate_id in active)   # frontier drawn last, on top
    for e in ordered:
        on = e.point.candidate_id in active
        cx, cy = fx(getattr(e.point, panel.x)), fy(getattr(e.point, panel.y))
        cls = "front" if on else "old"
        parts.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{6 if on else 4}" '
                     f'fill="{FRONT if on else OLD}" fill-opacity="{1.0 if on else 0.7}">'
                     f'<title>{escape(e.point.name)}{" (" + escape(e.tier) + ")" if e.tier else ""}</title></circle>')
        if on:
            parts.append(f'<text x="{cx + 9:.1f}" y="{cy - 7:.1f}" fill="{TEXT}">{escape(e.point.name[:22])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_plots(spectrum: Spectrum, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for panel in PANELS:
        p = out / f"{panel.slug}.svg"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(render_panel(spectrum, panel), encoding="utf-8")
        tmp.replace(p)
        paths.append(p)
    return paths
