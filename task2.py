#!/usr/bin/env python3
"""
任务二：复用 task1 图像，测量电池片间距（像素与毫米）。
假定：单片电池在图像中的竖直跨度与物理高度 210 mm 对应（用行带高度标定）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

TASK1 = Path(__file__).resolve().parent / "task1"
CELL_HEIGHT_MM = 210.0


def smooth_1d(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a.astype(np.float64)
    k = np.ones(win, dtype=np.float64) / win
    return np.convolve(a.astype(np.float64), k, mode="same")


def greedy_peaks(sig: np.ndarray, min_dist: int, rel: float) -> list[int]:
    s = sig.astype(np.float64).copy()
    mx = float(s.max()) if s.size else 0.0
    thr = mx * rel
    out: list[int] = []
    while True:
        i = int(np.argmax(s))
        if s[i] < thr:
            break
        out.append(i)
        lo, hi = max(0, i - min_dist), min(len(s), i + min_dist + 1)
        s[lo:hi] = 0.0
    return sorted(out)


def find_busbar_row_mask(row_mean: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    thr = np.percentile(row_mean, percentile)
    return row_mean > thr


def cell_row_intervals(h: int, busbar_mask: np.ndarray, min_h: int = 350) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    y = 0
    while y < h:
        if busbar_mask[y]:
            while y < h and busbar_mask[y]:
                y += 1
            continue
        y0 = y
        while y < h and not busbar_mask[y]:
            y += 1
        if y - y0 >= min_h:
            intervals.append((y0, y))
    return intervals


def vertical_seams_x(gray_strip: np.ndarray, mindist: int) -> list[int]:
    blur = cv2.GaussianBlur(gray_strip, (5, 21), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    proj = gx.sum(axis=0)
    p = smooth_1d(proj, max(15, mindist // 6))
    return greedy_peaks(p, mindist, 0.28)


def thin_horizontal_lines(
    gray: np.ndarray,
    min_row_dist: int | None = None,
    peak_rel: float = 0.32,
) -> tuple[list[int], list[int]]:
    """
    细横线：对 |∂I/∂y| 做行投影取峰。返回 (峰所在行 y, 各峰在半高上的跨行像素数)。
    """
    h, _w = gray.shape
    if min_row_dist is None:
        min_row_dist = max(35, h // 32)
    blur = cv2.GaussianBlur(gray, (21, 5), 0)
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    p = smooth_1d(gy.mean(axis=1), max(7, min_row_dist // 5))
    ys = greedy_peaks(p.copy(), min_row_dist, peak_rel)
    base = float(np.percentile(p, 28))
    spans: list[int] = []
    for y in ys:
        vmax = float(p[y])
        if vmax <= base + 1.0:
            spans.append(1)
            continue
        thr = base + 0.5 * (vmax - base)
        lo = hi = y
        while lo > 0 and p[lo - 1] >= thr:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr:
            hi += 1
        spans.append(int(hi - lo + 1))
    return ys, spans


def thin_vertical_lines(
    gray: np.ndarray,
    min_col_dist: int | None = None,
    peak_rel: float = 0.32,
    max_peak_span: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    细竖线：对 |∂I/∂x| 做列投影取峰。返回 (峰所在列 x, 各峰在半高上的跨列像素数)。
    max_peak_span：若给定，丢弃半高宽超过该值的峰（相机4 右缘常见宽峰伪检测，真缝约 34–40 px）。
    """
    _h, w = gray.shape
    if min_col_dist is None:
        min_col_dist = max(35, w // 32)
    blur = cv2.GaussianBlur(gray, (5, 21), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    p = smooth_1d(gx.mean(axis=0), max(7, min_col_dist // 5))
    xs = greedy_peaks(p.copy(), min_col_dist, peak_rel)
    base = float(np.percentile(p, 28))
    spans: list[int] = []
    for x in xs:
        vmax = float(p[x])
        if vmax <= base + 1.0:
            spans.append(1)
            continue
        thr = base + 0.5 * (vmax - base)
        lo = hi = x
        while lo > 0 and p[lo - 1] >= thr:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr:
            hi += 1
        spans.append(int(hi - lo + 1))
    if max_peak_span is not None:
        pairs = [(x, s) for x, s in zip(xs, spans) if s <= max_peak_span]
        if pairs:
            xs = [a for a, _ in pairs]
            spans = [b for _, b in pairs]
    return xs, spans


@dataclass(frozen=True)
class BusbarColumnEdges:
    """竖直汇流条在列均值剖面上的内外缘（突变小=靠峰一侧，突变大=靠背景一侧）。"""

    x_left_big: int
    x_left_small: int
    x_right_small: int
    x_right_big: int
    x_peak: int
    width_inner_px: int
    thr_big: float
    thr_small: float
    base: float
    peak_val: float


def busbar_search_roi_cam14(w: int, mode: Literal["left", "right"]) -> tuple[int, int]:
    """相机 1/4：在左/右缘 ROI 内找列均值峰，再定汇流条宽度。"""
    if mode == "left":
        return 0, max(48, int(w * 0.22))
    return min(w - 48, int(w * 0.78)), w


# 相机 1 左缘图 _0/_4/_8：ROI 最左侧常有与细竖缝同量级的窄尖峰，应用 min_peak_run_px 忽略之
CAM1_LEFT_BROAD_PEAK_SLOTS: frozenset[int] = frozenset({0, 4, 8})


def _busbar_peak_rel_prefer_broad(
    seg: np.ndarray,
    base: float,
    min_run_px: int,
) -> int:
    """在 ROI 剖面 seg 上，在局部极大中优先选「坡宽 ≥ min_run_px」的最高峰，避免窄尖峰当汇流条峰。"""
    n = len(seg)
    if n < 5 or min_run_px <= 1:
        return int(np.argmax(seg))
    idxs: list[int] = []
    for i in range(1, n - 1):
        if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            idxs.append(i)
    if not idxs:
        return int(np.argmax(seg))
    cand: list[tuple[float, int]] = []
    for j in idxs:
        h = float(seg[j])
        if h <= base + 2.0:
            continue
        thr = base + 0.30 * (h - base)
        lo, hi = j, j
        while lo > 0 and float(seg[lo - 1]) >= thr:
            lo -= 1
        while hi < n - 1 and float(seg[hi + 1]) >= thr:
            hi += 1
        run = hi - lo + 1
        if run >= min_run_px:
            cand.append((h, j))
    if not cand:
        return int(np.argmax(seg))
    cand.sort(key=lambda t: (-t[0], -t[1]))
    return cand[0][1]


def busbar_edges_from_column_mean(
    gray: np.ndarray,
    x_search_lo: int,
    x_search_hi: int,
    *,
    frac_big: float = 0.18,
    frac_small: float = 0.62,
    smooth_win: int | None = None,
    base_pct: float = 12.0,
    min_peak_run_px: int | None = None,
) -> tuple[np.ndarray, BusbarColumnEdges]:
    """
    全图列均值剖面（可先平滑）。在 [x_search_lo, x_search_hi) 内找峰，用两档高度阈值定缘：
    - **thr_big**（突变大，离基线较近）：外缘脚点；
    - **thr_small**（突变小，更靠峰顶）：内缘脚点。
    **汇流条区域**取两内缘之间 [x_left_small, x_right_small]，像素宽度为 ``width_inner_px``。
    min_peak_run_px：若给定，峰在 ROI 内须满足近似坡宽 ≥ 该值（像素），用于剔除左侧窄尖峰。
    返回 (平滑后全宽剖面, 边缘参数)，剖面长度 = 图像宽。
    """
    col = gray.mean(axis=0).astype(np.float64)
    w = gray.shape[1]
    x_lo = int(np.clip(x_search_lo, 0, w - 1))
    x_hi = int(np.clip(x_search_hi, x_lo + 2, w))
    span = x_hi - x_lo
    if smooth_win is None:
        sw = int(np.clip(span // 20, 11, 31))
        if sw % 2 == 0:
            sw += 1
        if min_peak_run_px is not None:
            sw = min(sw, 13)
            if sw % 2 == 0:
                sw -= 1
            sw = max(sw, 7)
    else:
        sw = max(3, int(smooth_win))
        if sw % 2 == 0:
            sw += 1
    p_full = smooth_1d(col, sw)
    seg = p_full[x_lo:x_hi]
    base = float(np.percentile(seg, base_pct))
    if min_peak_run_px is not None and min_peak_run_px > 1:
        imax_rel = _busbar_peak_rel_prefer_broad(seg, base, int(min_peak_run_px))
    else:
        imax_rel = int(np.argmax(seg))
    x_peak = x_lo + imax_rel
    peak_val = float(p_full[x_peak])
    if peak_val <= base + 1.0:
        mid = (x_lo + x_hi) // 2
        z = BusbarColumnEdges(
            mid, mid, mid, mid, mid, 1, float("nan"), float("nan"), base, peak_val
        )
        return p_full, z
    thr_big = base + float(frac_big) * (peak_val - base)
    thr_small = base + float(frac_small) * (peak_val - base)
    if thr_small <= thr_big + 1.0:
        thr_small = thr_big + max(1.0, 0.02 * (peak_val - base))

    x_left_big: int | None = None
    x_left_small: int | None = None
    for i in range(x_lo, x_peak + 1):
        if x_left_big is None and p_full[i] >= thr_big:
            x_left_big = i
        if x_left_small is None and p_full[i] >= thr_small:
            x_left_small = i

    if x_left_big is None:
        x_left_big = x_lo
    if x_left_small is None:
        x_left_small = x_peak

    x_right_big: int | None = None
    x_right_small: int | None = None
    for i in range(x_hi - 1, x_peak - 1, -1):
        if x_right_big is None and p_full[i] >= thr_big:
            x_right_big = i
        if x_right_small is None and p_full[i] >= thr_small:
            x_right_small = i

    if x_right_big is None:
        x_right_big = x_hi - 1
    if x_right_small is None:
        x_right_small = x_peak

    w_in = max(1, int(x_right_small) - int(x_left_small) + 1)
    z = BusbarColumnEdges(
        int(x_left_big),
        int(x_left_small),
        int(x_right_small),
        int(x_right_big),
        int(x_peak),
        w_in,
        thr_big,
        thr_small,
        base,
        peak_val,
    )
    return p_full, z


def busbar_search_roi_row_top_bottom(
    h: int, position: Literal["top", "bottom"]
) -> tuple[int, int]:
    """
    相机 1/4 可见的横向外缘汇流条：在 **上缘或下缘** 各取高度约 max(48, 0.22h)（且至少 3 行、保证 ROI 有效）的条带内搜行均值峰。
    """
    h = max(1, int(h))
    if h < 4:
        return 0, h
    band = max(48, int(h * 0.22))
    band = min(band, h - 2)
    band = max(band, min(3, h))
    if position == "top":
        return 0, min(h, band)
    y_lo = max(0, h - band)
    if y_lo + 2 > h:
        y_lo = max(0, h - 3)
    return y_lo, h


def busbar_edges_from_row_mean(
    gray: np.ndarray,
    y_search_lo: int,
    y_search_hi: int,
    *,
    frac_big: float = 0.18,
    frac_small: float = 0.62,
    smooth_win: int | None = None,
    base_pct: float = 12.0,
    min_peak_run_px: int | None = None,
) -> tuple[np.ndarray, BusbarColumnEdges]:
    """
    行均值剖面上的双阈值缘，与 `busbar_edges_from_column_mean` 同逻辑。
    返回 (平滑后全高剖面, 边缘参数)；各 ``x_*`` 字段在此表示 **行号 y**。
    """
    row = gray.mean(axis=1).astype(np.float64)
    hh = gray.shape[0]
    y_lo = int(np.clip(y_search_lo, 0, hh - 1))
    y_hi = int(np.clip(y_search_hi, y_lo + 2, hh))
    span = y_hi - y_lo
    if smooth_win is None:
        sw = int(np.clip(span // 20, 11, 31))
        if sw % 2 == 0:
            sw += 1
        if min_peak_run_px is not None:
            sw = min(sw, 13)
            if sw % 2 == 0:
                sw -= 1
            sw = max(sw, 7)
    else:
        sw = max(3, int(smooth_win))
        if sw % 2 == 0:
            sw += 1
    p_full = smooth_1d(row, sw)
    seg = p_full[y_lo:y_hi]
    base = float(np.percentile(seg, base_pct))
    if min_peak_run_px is not None and min_peak_run_px > 1:
        imax_rel = _busbar_peak_rel_prefer_broad(seg, base, int(min_peak_run_px))
    else:
        imax_rel = int(np.argmax(seg))
    y_peak = y_lo + imax_rel
    peak_val = float(p_full[y_peak])
    if peak_val <= base + 1.0:
        mid = (y_lo + y_hi) // 2
        z = BusbarColumnEdges(
            mid, mid, mid, mid, mid, 1, float("nan"), float("nan"), base, peak_val
        )
        return p_full, z
    thr_big = base + float(frac_big) * (peak_val - base)
    thr_small = base + float(frac_small) * (peak_val - base)
    if thr_small <= thr_big + 1.0:
        thr_small = thr_big + max(1.0, 0.02 * (peak_val - base))

    y_left_big: int | None = None
    y_left_small: int | None = None
    for i in range(y_lo, y_peak + 1):
        if y_left_big is None and p_full[i] >= thr_big:
            y_left_big = i
        if y_left_small is None and p_full[i] >= thr_small:
            y_left_small = i

    if y_left_big is None:
        y_left_big = y_lo
    if y_left_small is None:
        y_left_small = y_peak

    y_right_big: int | None = None
    y_right_small: int | None = None
    for i in range(y_hi - 1, y_peak - 1, -1):
        if y_right_big is None and p_full[i] >= thr_big:
            y_right_big = i
        if y_right_small is None and p_full[i] >= thr_small:
            y_right_small = i

    if y_right_big is None:
        y_right_big = y_hi - 1
    if y_right_small is None:
        y_right_small = y_peak

    w_in = max(1, int(y_right_small) - int(y_left_small) + 1)
    z = BusbarColumnEdges(
        int(y_left_big),
        int(y_left_small),
        int(y_right_small),
        int(y_right_big),
        int(y_peak),
        w_in,
        thr_big,
        thr_small,
        base,
        peak_val,
    )
    return p_full, z


def _first_row_top_y_from_strip_brightness(
    gray: np.ndarray,
    y_sobel_top: int,
    y_first_row_gap: int,
) -> int:
    """
    在首条 Sobel 横峰与第一行行缝之间，对左侧条带做行均值平滑：先取汇流条亮带峰值 y_p，
    再按相对行缝附近电池片区亮度的累计大幅下降，取第一行越过阈值处作为 **汇流条底边缘**
    （亮带过渡到电池片区域的下沿），而非外框棱边。
    """
    h, w = gray.shape
    x0, x1 = max(0, int(w * 0.05)), min(w, int(w * 0.16))
    strip = gray[:, x0:x1].astype(np.float64)
    rm = strip.mean(axis=1)
    span = max(30, y_first_row_gap - y_sobel_top)
    k = int(np.clip(span // 25, 11, 31))
    if k % 2 == 0:
        k += 1
    sm = np.convolve(rm, np.ones(k) / k, mode="same")
    lo = min(y_sobel_top + 25, y_first_row_gap - 100)
    hi = max(lo + 50, y_first_row_gap - 40)
    lo = max(0, lo)
    hi = min(h - 1, hi)
    if lo >= hi:
        return y_sobel_top
    seg = sm[lo:hi]
    y_p = int(lo + int(np.argmax(seg)))
    peak = float(sm[y_p])
    med = float(np.median(seg))
    if peak < med * 1.2:
        return y_sobel_top
    y_lo_ref = max(y_p + 30, y_first_row_gap - 120)
    y_hi_ref = min(h, y_first_row_gap - 20)
    if y_lo_ref >= y_hi_ref:
        return y_sobel_top
    sm_ref = float(np.median(sm[y_lo_ref:y_hi_ref]))
    # 相对行缝区域亮度的落差；0.35≈「已完成汇流条→片区过渡」的比例，下限防噪声
    drop_need = max(18.0, 0.35 * (peak - sm_ref))
    for y in range(y_p + 5, y_first_row_gap - 20):
        if peak - float(sm[y]) >= drop_need:
            return int(y)
    return y_sobel_top


def first_cell_row_height_px(gray: np.ndarray) -> tuple[int, int, int]:
    """
    第一行电池片竖直高度 h（像素）：底边 y1 为第一条行缝（最上两条 thin_horizontal_lines 峰中较低者）；
    顶边 y0 在「整幅图顶部可见」时由左侧条带亮度定位 **汇流条底边缘**，否则退回 Sobel 最上峰。
    """
    ys, _ = thin_horizontal_lines(gray)
    ys_ord = sorted(ys)
    if len(ys_ord) < 2:
        raise ValueError(
            f"检测到的横线少于 2 条（{len(ys_ord)}），无法量第一行高度 h"
        )
    y_sobel_top, y1 = int(ys_ord[0]), int(ys_ord[1])
    h_img = gray.shape[0]
    gap0 = y1 - y_sobel_top
    y0 = y_sobel_top
    # 仅当最上 Sobel 峰靠近画面上沿时细化（裁切到面板中部时首峰不是「顶框」）
    if y_sobel_top < int(0.35 * h_img) and gap0 >= 80:
        y0 = _first_row_top_y_from_strip_brightness(gray, y_sobel_top, y1)
        h_px = y1 - y0
        min_h = min(200, max(80, int(0.25 * gap0)))
        if h_px < min_h:
            y0 = y_sobel_top
    return y0, y1, y1 - y0


def valley_width_px(
    profile: np.ndarray,
    center_idx: int,
    frac: float = 0.55,
) -> float:
    """在 1D 剖面（已平滑）上，估计中心低谷宽度（FWHM 类：低于两侧基准 * frac 的连续宽度）。"""
    if profile.size < 5:
        return float("nan")
    p = profile.astype(np.float64)
    c = int(np.clip(center_idx, 0, p.size - 1))
    half = min(25, p.size // 4)
    lo = max(0, c - half)
    hi = min(p.size, c + half + 1)
    vmin = float(p[lo:hi].min())
    left_ref = float(p[0 : min(8, p.size)].mean())
    right_ref = float(p[max(0, p.size - 8) :].mean())
    base = max(left_ref, right_ref, vmin + 1.0)
    thr = vmin + (base - vmin) * frac
    x = c
    while x > 0 and p[x] <= thr:
        x -= 1
    left = x
    x = c
    while x < p.size - 1 and p[x] <= thr:
        x += 1
    right = x
    w = float(max(0, right - left))
    return w if w > 0.5 else float("nan")


def horizontal_gap_at_seam(
    gray: np.ndarray, ya: int, yb: int, x_seam: int, half_win: int = 72
) -> float:
    h, w = gray.shape
    x0 = max(0, x_seam - half_win)
    x1 = min(w, x_seam + half_win + 1)
    patch = gray[ya:yb, x0:x1]
    if patch.size == 0:
        return float("nan")
    prof = patch.mean(axis=0)
    prof = smooth_1d(prof, max(5, half_win // 14))
    cx = x_seam - x0
    return valley_width_px(prof, cx)


def bright_peak_x(col_mean: np.ndarray, x0: int, x1: int) -> int | None:
    seg = col_mean[x0:x1]
    if seg.size < 3:
        return None
    i = int(np.argmax(seg)) + x0
    if float(col_mean[i]) < float(np.median(col_mean)) + 2.0:
        return None
    return i


def bright_plateau_width_px(
    col_mean: np.ndarray,
    x0: int,
    x1: int,
    frac: float = 0.52,
    max_span: int | None = None,
) -> float:
    """
    在列均值剖面 [x0, x1) 内，沿峰值取亮度台地宽度，用于估计竖直汇流条（粗竖线）宽度（px）。
    与 task1 中「宽竖区」物理含义一致；外缘条与中间条应同量级。
    max_span：峰值两侧最多扩展列数，防止宽 ROI 下台地与整片高亮连成一片。
    """
    x0 = max(0, min(int(x0), col_mean.size - 1))
    x1 = max(x0 + 3, min(int(x1), col_mean.size))
    seg = col_mean[x0:x1].astype(np.float64)
    win = min(31, max(5, (len(seg) // 3) | 1))
    s = smooth_1d(seg, win)
    vmin = float(np.percentile(s, 20))
    vmax = float(np.max(s))
    if vmax < vmin + 2.0:
        return float("nan")
    thr = vmin + frac * (vmax - vmin)
    k = int(np.argmax(s))
    if max_span is None:
        max_span = max(24, min(240, len(s) // 2))
    left, right = k, k
    while (
        left > 0
        and s[left - 1] >= thr
        and (k - (left - 1)) <= max_span
    ):
        left -= 1
    while (
        right < len(s) - 1
        and s[right + 1] >= thr
        and ((right + 1) - k) <= max_span
    ):
        right += 1
    return float(max(1.0, right - left + 1))


def aggregate_row_col_mean(gray: np.ndarray, ya: int, yb: int) -> np.ndarray:
    return gray[ya:yb, :].mean(axis=0)


@dataclass
class FrameStats:
    horiz_gaps_px: list[float] = field(default_factory=list)
    vert_gaps_px: list[float] = field(default_factory=list)
    """c) 相机 1、4：外缘竖直汇流条（粗竖线）宽度"""
    edge_busbar_width_px: list[float] = field(default_factory=list)
    """d) 相机 2、3：中间竖直汇流条宽度"""
    mid_busbar_width_px: list[float] = field(default_factory=list)
    band_heights_px: list[float] = field(default_factory=list)


def camera_index_from_suffix(fname: str) -> int | None:
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", fname, re.I)
    if not m:
        return None
    return int(m.group(1)) % 4


def image_slot_from_suffix(fname: str) -> int | None:
    """`*_0_N.jpg` 中的序号 N（0..11）。"""
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", fname, re.I)
    return int(m.group(1)) if m else None


def process_image(path: Path, gray: np.ndarray, stats: FrameStats) -> None:
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    bus_mask = find_busbar_row_mask(row_mean, 99.0)
    rows = cell_row_intervals(h, bus_mask, min_h=350)
    if len(rows) < 1:
        return

    mindist_x = max(120, int(w * 0.065))
    cam = camera_index_from_suffix(path.name)

    for ri, (ya, yb) in enumerate(rows):
        stats.band_heights_px.append(float(yb - ya))
        strip = gray[ya:yb, :]
        seams = vertical_seams_x(strip, mindist_x)

        for sx in seams:
            g = horizontal_gap_at_seam(gray, ya, yb, sx)
            if not np.isnan(g) and 0.5 < g < 120:
                stats.horiz_gaps_px.append(float(g))

        if ri < len(rows) - 1:
            yb_cur = yb
            ya_next = rows[ri + 1][0]
            vg = float(ya_next - yb_cur)
            if 5 < vg < min(400, h * 0.25):
                stats.vert_gaps_px.append(vg)

    # c) d) 与 task2.ipynb 一致：整幅列均值 + 双阈值内缘宽度，每图仅 1 个样本
    if cam is not None:
        slot = image_slot_from_suffix(path.name)
        if cam == 0:
            x0, x1 = busbar_search_roi_cam14(w, "left")
            min_run = (
                32
                if slot is not None and slot in CAM1_LEFT_BROAD_PEAK_SLOTS
                else None
            )
            _, e = busbar_edges_from_column_mean(
                gray, x0, x1, min_peak_run_px=min_run
            )
            stats.edge_busbar_width_px.append(float(e.width_inner_px))
        elif cam == 3:
            x0, x1 = busbar_search_roi_cam14(w, "right")
            _, e = busbar_edges_from_column_mean(gray, x0, x1)
            stats.edge_busbar_width_px.append(float(e.width_inner_px))
        elif cam in (1, 2):
            _, e = busbar_edges_from_column_mean(gray, 0, w)
            stats.mid_busbar_width_px.append(float(e.width_inner_px))


def aggregate_sets(task1_root: Path | None = None) -> dict[str, Any]:
    root = Path(task1_root) if task1_root is not None else TASK1
    all_h: list[float] = []
    all_v: list[float] = []
    all_outer: list[float] = []
    all_inner: list[float] = []
    bands: list[float] = []

    for set_dir in sorted(root.glob("set*")):
        if not set_dir.is_dir():
            continue
        st = FrameStats()
        for img_path in sorted(set_dir.glob("*.jpg")):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            process_image(img_path, gray, st)
        all_h.extend(st.horiz_gaps_px)
        all_v.extend(st.vert_gaps_px)
        all_outer.extend(st.edge_busbar_width_px)
        all_inner.extend(st.mid_busbar_width_px)
        bands.extend(st.band_heights_px)

    median_band = float(np.median(bands)) if bands else float("nan")
    scale = CELL_HEIGHT_MM / median_band if median_band > 1e-6 else float("nan")

    def mm_range(vals: list[float]) -> tuple[float, float, float, float]:
        if not vals:
            return float("nan"), float("nan"), float("nan"), float("nan")
        a = np.asarray(vals, dtype=np.float64)
        mn_px, mx_px = float(a.min()), float(a.max())
        return mn_px, mx_px, mn_px * scale, mx_px * scale

    mn_h_px, mx_h_px, mn_h_mm, mx_h_mm = mm_range(all_h)
    mn_v_px, mx_v_px, mn_v_mm, mx_v_mm = mm_range(all_v)
    mn_o_px, mx_o_px, mn_o_mm, mx_o_mm = mm_range(all_outer)
    mn_i_px, mx_i_px, mn_i_mm, mx_i_mm = mm_range(all_inner)

    return {
        "assumption_cell_height_mm": CELL_HEIGHT_MM,
        "median_cell_row_band_height_px": median_band,
        "scale_mm_per_px": scale,
        "a_horizontal_cell_gap_px": {"min": mn_h_px, "max": mx_h_px},
        "a_horizontal_cell_gap_mm": {"min": mn_h_mm, "max": mx_h_mm},
        "b_vertical_cell_gap_px": {"min": mn_v_px, "max": mx_v_px},
        "b_vertical_cell_gap_mm": {"min": mn_v_mm, "max": mx_v_mm},
        "c_edge_busbar_width_cam1_cam4_px": {"min": mn_o_px, "max": mx_o_px},
        "c_edge_busbar_width_cam1_cam4_mm": {"min": mn_o_mm, "max": mx_o_mm},
        "d_mid_busbar_width_cam2_cam3_px": {"min": mn_i_px, "max": mx_i_px},
        "d_mid_busbar_width_cam2_cam3_mm": {"min": mn_i_mm, "max": mx_i_mm},
        "counts": {
            "horizontal_gaps": len(all_h),
            "vertical_gaps": len(all_v),
            "c_edge_busbar_width_samples": len(all_outer),
            "d_mid_busbar_width_samples": len(all_inner),
            "band_height_samples": len(bands),
        },
    }


def print_task2_report(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """在终端/Notebook 中打印任务二四项结果；若未传入则现算。"""
    if results is None:
        results = aggregate_sets()

    def fmt_px_mm(pxk: str, mmk: str, title: str) -> None:
        px = results[pxk]
        mm = results[mmk]
        print(title)
        print(
            f"  最小: {px['min']:.2f} px  ≈  {mm['min']:.3f} mm"
        )
        print(
            f"  最大: {px['max']:.2f} px  ≈  {mm['max']:.3f} mm"
        )

    print("标定：假定单片电池竖直跨度 = "
          f"{results['assumption_cell_height_mm']} mm")
    print(
        f"行带高度中位数 = {results['median_cell_row_band_height_px']:.1f} px，"
        f"比例尺 = {results['scale_mm_per_px']:.6f} mm/px"
    )
    print()

    fmt_px_mm(
        "a_horizontal_cell_gap_px",
        "a_horizontal_cell_gap_mm",
        "a) 相邻两片电池（左右并排）之间的间隙（细竖缝宽度）",
    )
    print()
    fmt_px_mm(
        "b_vertical_cell_gap_px",
        "b_vertical_cell_gap_mm",
        "b) 相邻两片电池（上下叠放）之间的间隙（行间竖向间距）",
    )
    print()
    fmt_px_mm(
        "c_edge_busbar_width_cam1_cam4_px",
        "c_edge_busbar_width_cam1_cam4_mm",
        "c) 相机 1、4：外缘竖直汇流条（粗竖线）宽度",
    )
    print()
    fmt_px_mm(
        "d_mid_busbar_width_cam2_cam3_px",
        "d_mid_busbar_width_cam2_cam3_mm",
        "d) 相机 2、3：中间竖直汇流条宽度（应与 c 同量级）",
    )
    print()
    print("样本计数:", results["counts"])
    return results


def results_to_markdown(results: dict[str, Any]) -> str:
    """供 Notebook `display(Markdown(...))` 渲染结果表。"""
    c = results["counts"]
    lines = [
        "### 任务二 · 结果汇总",
        "",
        f"- **标定**：单片电池竖直跨度 = **{results['assumption_cell_height_mm']:.0f} mm**",
        f"- **行带高度中位数**：**{results['median_cell_row_band_height_px']:.1f} px**",
        f"- **比例尺**：**{results['scale_mm_per_px']:.6f} mm/px**",
        "",
        "| 项目 | 最小 (px) | 最大 (px) | 最小 (mm) | 最大 (mm) |",
        "|------|-----------|-----------|-----------|-----------|",
    ]

    def row(label: str, pxk: str, mmk: str) -> None:
        px, mm = results[pxk], results[mmk]
        lines.append(
            f"| {label} | {px['min']:.2f} | {px['max']:.2f} | "
            f"{mm['min']:.3f} | {mm['max']:.3f} |"
        )

    row("a) 左右并排间隙", "a_horizontal_cell_gap_px", "a_horizontal_cell_gap_mm")
    row("b) 上下叠放间隙", "b_vertical_cell_gap_px", "b_vertical_cell_gap_mm")
    row(
        "c) 相机1、4 外缘汇流条宽度",
        "c_edge_busbar_width_cam1_cam4_px",
        "c_edge_busbar_width_cam1_cam4_mm",
    )
    row(
        "d) 相机2、3 中间汇流条宽度",
        "d_mid_busbar_width_cam2_cam3_px",
        "d_mid_busbar_width_cam2_cam3_mm",
    )
    lines.extend(
        [
            "",
            "**样本计数**："
            f" 左右缝 {c['horizontal_gaps']}，"
            f" 行间 {c['vertical_gaps']}，"
            f" c 外缘条宽 {c['c_edge_busbar_width_samples']}，"
            f" d 中间条宽 {c['d_mid_busbar_width_samples']}，"
            f" 行带 {c['band_height_samples']}",
            "",
            "完整 JSON 与脚本同目录：`task2_results.json`（运行下一格代码会覆盖更新）。",
        ]
    )
    return "\n".join(lines)


def save_results_json(results: dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    out = aggregate_sets()
    out_path = Path(__file__).resolve().parent / "task2_results.json"
    save_results_json(out, out_path)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
