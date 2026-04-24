"""
task2.ipynb 配套：标定、细/粗竖线与横线可视化、3×4 拼图。
细横/竖线检测在本地实现，避免依赖 task2.py 版本差异导致 ImportError。
标定与其它统计仍复用 task2 中的工具函数。
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from task2 import (
    CAM1_LEFT_BROAD_PEAK_SLOTS,
    CELL_HEIGHT_MM,
    TASK1,
    BusbarColumnEdges,
    busbar_edges_from_column_mean,
    busbar_edges_from_row_mean,
    busbar_search_roi_cam14,
    busbar_search_roi_row_top_bottom,
    cell_row_intervals,
    find_busbar_row_mask,
    first_cell_row_height_px,
    greedy_peaks,
    smooth_1d,
)
from task2_gap_measurement import (
    compute_lines,
    measure_hline_gap_gradient,
    measure_vline_gap_gradient,
)

# 上下叠放示意图：这些槽位在画面中只取两条片间细横缝（去掉顶框/过宽伪峰）
TWO_THIN_HORIZONTAL_STACK_SLOTS: frozenset[int] = frozenset(
    {0, 1, 2, 3, 8, 9, 10, 11}
)


def thin_horizontal_lines(
    gray: np.ndarray,
    min_row_dist: int | None = None,
    peak_rel: float = 0.32,
) -> tuple[list[int], list[int]]:
    """细横线：|∂I/∂y| 行投影峰 + 半高宽（跨行像素）。"""
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


THIN_HORIZ_MAX_SPAN_FOR_STACK_PAIR = 72
# 非相机 4：竖缝半高宽上限（整幅 Sobel 在 set2/3/4 上易出现极宽伪峰）
THIN_VLINE_MAX_SPAN_DEFAULT: int = 96
# 横剖面尖峰：峰值须 ≥ 局部背景（鞍部）× 该倍数
THIN_VLINE_PEAK_BG_RATIO: float = 1.5
# 突变路径：依次用 peak_bg_ratio 减去这些增量做重试（先严后松），直到检出至少一个峰
THIN_VLINE_MUTATION_BG_RELAX_DELTAS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.28)
# 尖峰搜索左右留白（相对全宽），避开最边缘
THIN_VLINE_INTERIOR_MARGIN_FRAC: float = 0.065
# 突变检测：轻平滑窗口 = max(3, min(min_col_dist//该除数, 9))（奇数），保留骑在宽边坡上的窄峰
THIN_VLINE_MUTATION_FINE_SMOOTH_DIV: int = 10
# 某条缝半高宽 > 中位数×该因子时，在本档内改选更窄的尖峰
THIN_VLINE_SPAN_VS_MEDIAN_MAX: float = 2.05


def expected_horizontal_seam_count(image_slot: int | None) -> int | None:
    """
    整板 3×4 中每张 tile 可见的 **细横缝条数**（与统计一致）：
    上行 4 张、下行 4 张各 **2** 条；中行 4 张 **3** 条。
    """
    if image_slot is None or int(image_slot) < 0:
        return None
    return 3 if int(image_slot) // 4 == 1 else 2


def thin_horizontal_lines_stacked_pair(
    gray: np.ndarray,
    image_slot: int | None,
    *,
    stack_slots: frozenset[int] | None = None,
    thin_max_span: int = THIN_HORIZ_MAX_SPAN_FOR_STACK_PAIR,
    peak_rel: float = 0.30,
) -> tuple[list[int], list[int]]:
    """
    **通用 b）横线条数**：按槽位所在 **行** 强制 **2 或 3** 条（12 张里 4×2 + 4×3 + 4×2）。

    - **上行 / 下行**（序号 0–3、8–11）：``thin_horizontal_lines_universal`` 后优先取
      半高宽 ≤ ``thin_max_span`` 的峰，按 y **取最大的 2 条**（片间横缝靠近画面下端）；
      细峰不足则对全部峰 **合并到 2 条**，仍不足则略放宽 ``peak_rel`` 再试。
    - **中行**（4–7）：合并或放宽直至 **3 条**。

    ``stack_slots`` 仅保留兼容旧调用，逻辑已按槽位行号统一，无需再传。
    """
    _ = stack_slots  # 兼容旧 notebook 签名
    exp = expected_horizontal_seam_count(image_slot)
    ys, spans = thin_horizontal_lines_universal(gray, peak_rel=peak_rel)

    if exp is None:
        return ys, spans

    row = int(image_slot) // 4

    def _fallback_more_peaks() -> tuple[list[int], list[int]]:
        rel_fb = max(0.18, float(peak_rel) * 0.88)
        return thin_horizontal_lines_universal(gray, peak_rel=rel_fb)

    if row != 1:
        thin = [(y, s) for y, s in zip(ys, spans) if s <= thin_max_span]
        thin_sorted = sorted(thin, key=lambda t: t[0])
        if len(thin_sorted) >= exp:
            picked = thin_sorted[-exp:]
            return [int(t[0]) for t in picked], [int(t[1]) for t in picked]
        if len(ys) >= exp:
            y2, s2 = _merge_closest_peak_pairs(list(ys), list(spans), exp)
            return y2, s2
        ys3, sp3 = _fallback_more_peaks()
        if len(ys3) >= exp:
            if len(ys3) > exp:
                ys3, sp3 = _merge_closest_peak_pairs(ys3, sp3, exp)
            return ys3, sp3
        return ys, spans

    # 中行：3 条
    if len(ys) > exp:
        return _merge_closest_peak_pairs(list(ys), list(spans), exp)
    if len(ys) < exp:
        ys4, sp4 = _fallback_more_peaks()
        if len(ys4) >= exp:
            ys, spans = ys4, sp4
            if len(ys) > exp:
                return _merge_closest_peak_pairs(ys, spans, exp)
    return ys, spans


def paint_vertical_gap_inner_edges(
    bgr: np.ndarray, edges: list[BusbarColumnEdges], thickness: int = 2
) -> np.ndarray:
    """竖缝：两内缘竖线（与 c 的金区边界同义）。"""
    out = bgr.copy()
    hh, w = out.shape[:2]
    for e in edges:
        for xv in (e.x_left_small, e.x_right_small):
            xi = int(np.clip(xv, 0, w - 1))
            cv2.line(out, (xi, 0), (xi, hh - 1), (0, 200, 255), thickness)
    return out


def paint_horizontal_gap_inner_edges(
    bgr: np.ndarray, edges: list[BusbarColumnEdges], thickness: int = 2
) -> np.ndarray:
    """横缝：两内缘横线；``BusbarColumnEdges`` 坐标为行号。"""
    out = bgr.copy()
    h_img, ww = out.shape[:2]
    for e in edges:
        for yv in (e.x_left_small, e.x_right_small):
            yi = int(np.clip(yv, 0, h_img - 1))
            cv2.line(out, (0, yi), (ww - 1, yi), (0, 200, 255), thickness)
    return out


def _greedy_peaks_robust(sig: np.ndarray, min_dist: int, rel: float) -> list[int]:
    """
    与 ``task2.greedy_peaks`` 相同的非极大抑制贪心取峰，但阈值用
    ``percentile(sig, 99.5) * rel`` 代替 ``max(sig) * rel``，避免单边假峰拉满 max 后压掉真缝。
    """
    s = sig.astype(np.float64).copy()
    roof = float(np.percentile(s, 99.5))
    if roof <= 1e-12:
        return []
    thr = roof * rel
    out: list[int] = []
    while True:
        i = int(np.argmax(s))
        if s[i] < thr:
            break
        out.append(i)
        lo, hi = max(0, i - min_dist), min(len(s), i + min_dist + 1)
        s[lo:hi] = 0.0
    return sorted(out)


def _local_env_neighbor_min(s: np.ndarray, i: int, W: int) -> float:
    """峰 ``i`` 左右邻域（不含 ``i``，窗口 ``W``）内剖面最小值，作局部「背景/谷」参考。"""
    n = len(s)
    chunks: list[np.ndarray] = []
    if i > 0:
        lo = max(0, i - W)
        chunks.append(s[lo:i])
    if i + 1 < n:
        hi = min(n, i + W + 1)
        chunks.append(s[i + 1 : hi])
    if not chunks:
        return float(s[i])
    return float(np.min(np.concatenate(chunks)))


def _greedy_peaks_local_prominence(
    sig: np.ndarray,
    min_dist: int,
    prom_rel: float,
    *,
    prom_abs_floor: float | None = None,
) -> list[int]:
    """
    贪心取峰：**相对邻域背景** 的突起即可，不设全局幅度屋顶。
    接受条件：``p[i] - env(i) >= max(prom_abs_floor, prom_rel * p[i])``，
    其中 ``env(i)`` 为左右 ``min_dist`` 窗口内（不含峰心）的最小值。
    真竖线在 |gx| 剖面上是 **比两旁谷值高出一截** 的尖峰；汇流条大边往往 **很宽**，
    宽窄用半高宽另判，不在此用全局分位压阈值。
    """
    s = sig.astype(np.float64).copy()
    n = len(s)
    if n < 3:
        return []
    if prom_abs_floor is None:
        prom_abs_floor = max(1e-9, float(np.percentile(s, 8)) * 0.04)
    out: list[int] = []
    while True:
        i = int(np.argmax(s))
        v = float(s[i])
        if v <= 1e-12:
            break
        env = _local_env_neighbor_min(s, i, min_dist)
        prom = v - env
        need = max(float(prom_abs_floor), float(prom_rel) * v)
        if prom < need:
            break
        out.append(i)
        lo, hi = max(0, i - min_dist), min(n, i + min_dist + 1)
        s[lo:hi] = 0.0
    return sorted(out)


def _vertical_seam_column_keep_mask(
    gray: np.ndarray,
    cell_row_weights: np.ndarray,
    *,
    bright_pct: float = 99.0,
    margin_frac: float = 0.26,
    dilate_half_width: int | None = None,
) -> np.ndarray:
    """
    左/右缘 **竖直汇流条** 在列均值上极亮；其边缘在 |∂I/∂x| 上形成 **极宽** 峰，
    取半高宽时会到数百像素并占掉 6 条缝名额。此处仅在 **左右边带** 内找高亮列，
    横向膨胀后把 gx 置零，使投影与半高宽只反映 **片间细竖缝**。
    """
    h, w = gray.shape
    rw = np.asarray(cell_row_weights, dtype=np.float64).reshape(h)
    ws = float(rw.sum()) + 1e-6
    col_mean = (gray.astype(np.float64) * rw[:, None]).sum(axis=0) / ws
    thr = float(np.percentile(col_mean, bright_pct))
    bright = col_mean >= thr
    margin = int(max(12, min(int(w * margin_frac), int(w * 0.38))))
    xs = np.arange(w, dtype=np.int32)
    in_band = (xs < margin) | (xs >= w - margin)
    bus = bright & in_band
    if dilate_half_width is None:
        dilate_half_width = max(20, w // 72)
    if dilate_half_width > 0 and np.any(bus):
        k = int(max(3, dilate_half_width * 2 + 1))
        m = bus.astype(np.uint8).reshape(1, -1)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        m = cv2.dilate(m, kern).ravel().astype(bool)
    else:
        m = bus
    keep = (~m).astype(np.float64)
    return keep


def _vertical_gx_column_mean_raw(
    gray: np.ndarray,
    min_col_dist: int | None,
    *,
    row_weights: np.ndarray | None = None,
    column_keep: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """|gx| 列加权均值，**未**做列向 smooth_1d（供突变检测做多尺度平滑）。"""
    h, w = gray.shape
    if min_col_dist is None:
        min_col_dist = max(35, w // 32)
    blur = cv2.GaussianBlur(gray, (5, 21), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    if column_keep is not None:
        ck = np.asarray(column_keep, dtype=np.float64).reshape(w)
        gx = gx.astype(np.float64) * ck[None, :]
    if row_weights is not None:
        rw = np.asarray(row_weights, dtype=np.float64).reshape(h)
        ws = float(rw.sum())
        if ws > 1e-6:
            p_raw = (gx.astype(np.float64) * rw[:, None]).sum(axis=0) / ws
        else:
            p_raw = gx.mean(axis=0).astype(np.float64)
    else:
        p_raw = gx.mean(axis=0).astype(np.float64)
    return p_raw.astype(np.float64), min_col_dist


def _vertical_gx_column_profile(
    gray: np.ndarray,
    min_col_dist: int | None,
    *,
    row_weights: np.ndarray | None = None,
    column_keep: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """与细竖线可视化/粗检测相同的平滑列剖面 ``p``（|gx| 加权列均值 + smooth_1d）。"""
    p_raw, min_col_dist = _vertical_gx_column_mean_raw(
        gray, min_col_dist, row_weights=row_weights, column_keep=column_keep
    )
    p = smooth_1d(p_raw, max(7, min_col_dist // 5))
    return p, min_col_dist


def _spans_at_peaks_on_profile(p: np.ndarray, xs: list[int]) -> list[int]:
    """在剖面 ``p`` 上各峰 **半高宽**（跨列像素）。细竖缝为窄峰；汇流条边为宽肩台。"""
    base = float(np.percentile(p, 28))
    spans: list[int] = []
    for x in xs:
        vmax = float(p[x])
        if vmax <= base + 1.0:
            spans.append(1)
            continue
        thr_h = base + 0.5 * (vmax - base)
        lo = hi = x
        while lo > 0 and p[lo - 1] >= thr_h:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr_h:
            hi += 1
        spans.append(int(hi - lo + 1))
    return spans


def _local_bg_saddle(p: np.ndarray, i: int, W: int) -> float:
    """峰 ``i`` 两侧窗口内的谷底较大者，作局部「背景」参考（鞍部）。"""
    n = len(p)
    lo = max(0, i - W)
    hi = min(n, i + W + 1)
    left_min = float(np.min(p[lo:i])) if i > lo else float(p[i])
    right_min = float(np.min(p[i + 1 : hi])) if i + 1 < hi else float(p[i])
    return max(left_min, right_min)


def _unimodal_mutation_extent(p: np.ndarray, i: int) -> tuple[int, int] | None:
    """
    从局部极大 ``i`` 向左右扩展到 **突变起止**：左侧非降、右侧非升，整体为 **单峰无凹槽**
    （顶可为平肩）。若区间内出现违反单调的「凹口」则返回 ``None``。
    """
    n = len(p)
    L = i
    while L > 0 and float(p[L - 1]) <= float(p[L]):
        L -= 1
    R = i
    while R < n - 1 and float(p[R + 1]) <= float(p[R]):
        R += 1
    eps = 1e-9
    for k in range(L, i):
        if float(p[k]) > float(p[k + 1]) + eps:
            return None
    for k in range(i, R):
        if float(p[k]) < float(p[k + 1]) - eps:
            return None
    return L, R


def _mutation_peak_candidates_on_profile(
    p: np.ndarray,
    min_col_dist: int,
    peak_bg_ratio: float,
    margin: int,
    max_mutation_width: int | None,
) -> list[tuple[int, int, int, int, float]]:
    """在一条剖面 ``p`` 上收集突变尖峰候选 ``(i,L,R,width,height)``。"""
    n = len(p)
    W_bg = max(5, min_col_dist // 3)
    out: list[tuple[int, int, int, int, float]] = []
    for i in range(1, n - 1):
        if i < margin or i > n - 1 - margin:
            continue
        if not (
            float(p[i]) >= float(p[i - 1]) and float(p[i]) >= float(p[i + 1])
        ):
            continue
        bg = _local_bg_saddle(p, i, W_bg)
        if bg < 1e-12:
            continue
        if float(p[i]) < float(peak_bg_ratio) * bg:
            continue
        ext = _unimodal_mutation_extent(p, i)
        if ext is None:
            continue
        L, R = ext
        width = int(R - L + 1)
        if max_mutation_width is not None and width > int(max_mutation_width):
            continue
        out.append((i, L, R, width, float(p[i])))
    return out


def _thin_vertical_mutation_peaks_all(
    gray: np.ndarray,
    row_weights: np.ndarray,
    column_keep: np.ndarray,
    *,
    peak_bg_ratio: float,
    interior_margin_frac: float = THIN_VLINE_INTERIOR_MARGIN_FRAC,
    max_mutation_width: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    **横剖面突变尖峰（不固定条数）**：在 **轻平滑** 与 **原粗平滑** 两条剖面上分别取候选，
    再合并 NMS。**原因**：窄缝若落在宽汇流条边的缓坡上，强平滑后往往 **不再是局部极大**，
    只在更细的 ``p_fine`` 上仍呈尖峰；双尺度可减少「离群细缝检不出」的情况。

    仍须 ``p[i] ≥ peak_bg_ratio × 鞍部背景`` 且单峰无凹槽；**统计跨像素** 为 ``R-L+1``。
    合并时 **优先保留轻平滑剖面上的峰**（先加入 fine，再加入 coarse 补缺）。
    """
    p_raw, min_col_dist = _vertical_gx_column_mean_raw(
        gray, None, row_weights=row_weights, column_keep=column_keep
    )
    n = len(p_raw)
    if n < 5:
        return [], []
    margin = max(int(n * interior_margin_frac), min_col_dist, 2)
    w_fine = max(3, min(min_col_dist // THIN_VLINE_MUTATION_FINE_SMOOTH_DIV, 9))
    if w_fine % 2 == 0:
        w_fine += 1
    p_fine = smooth_1d(p_raw.copy(), w_fine)
    p_coarse = smooth_1d(p_raw.copy(), max(7, min_col_dist // 5))

    c_fine = _mutation_peak_candidates_on_profile(
        p_fine, min_col_dist, peak_bg_ratio, margin, max_mutation_width
    )
    c_coarse = _mutation_peak_candidates_on_profile(
        p_coarse, min_col_dist, peak_bg_ratio, margin, max_mutation_width
    )
    c_fine.sort(key=lambda t: -t[4])
    c_coarse.sort(key=lambda t: -t[4])
    chosen: list[tuple[int, int, int, int, float]] = []
    for c in c_fine:
        i = c[0]
        if any(abs(i - ch[0]) < min_col_dist for ch in chosen):
            continue
        chosen.append(c)
    for c in c_coarse:
        i = c[0]
        if any(abs(i - ch[0]) < min_col_dist for ch in chosen):
            continue
        chosen.append(c)
    chosen.sort(key=lambda t: t[0])
    xs = [int(c[0]) for c in chosen]
    spans = [int(c[3]) for c in chosen]
    return xs, spans


def _bin_x_range(
    x0: int, x1: int, expected_n: int, k: int
) -> tuple[int, int]:
    width = x1 - x0
    a = x0 + int(k * width / expected_n)
    b = x0 + int((k + 1) * width / expected_n)
    if k == expected_n - 1:
        b = x1
    return a, b


def _pick_sharp_peak_in_bin(
    p: np.ndarray,
    a: int,
    b: int,
    min_col_dist: int,
    peak_bg_ratio: float,
    *,
    max_half_width: int | None = None,
) -> tuple[int, int] | None:
    """
    在 ``[a,b]`` 内找 **尖峰**（严格局部极大），且 ``p[i] >= peak_bg_ratio * 局部背景``。
    若有 ``max_half_width``，只保留半高宽不超过该值的候选；择优 **峰值更高**，
    并列时更 **窄**（跨像素更接近同类尖峰）。
    """
    n = len(p)
    a = int(max(1, a))
    b = int(min(n - 2, b))
    if b <= a + 1:
        return None
    W_bg = max(5, min_col_dist // 3)
    best: tuple[float, float, int, int] | None = None
    for i in range(a + 1, b):
        if not (float(p[i]) >= float(p[i - 1]) and float(p[i]) >= float(p[i + 1])):
            continue
        bg = _local_bg_saddle(p, i, W_bg)
        if bg < 1e-12:
            continue
        if float(p[i]) < float(peak_bg_ratio) * bg:
            continue
        sp = int(_spans_at_peaks_on_profile(p, [i])[0])
        if max_half_width is not None and sp > int(max_half_width):
            continue
        score_peak = float(p[i])
        score_narrow = -float(sp)
        tup = (score_peak, score_narrow, i, sp)
        if best is None or tup[:2] > best[:2]:
            best = tup
    if best is None:
        return None
    return best[2], best[3]


def _thin_vertical_profile_uniform_sharp_peaks(
    gray: np.ndarray,
    row_weights: np.ndarray,
    column_keep: np.ndarray,
    *,
    expected_n: int,
    peak_bg_ratio: float = THIN_VLINE_PEAK_BG_RATIO,
    interior_margin_frac: float = THIN_VLINE_INTERIOR_MARGIN_FRAC,
    span_vs_median_max: float = THIN_VLINE_SPAN_VS_MEDIAN_MAX,
    max_half_width: int | None = None,
) -> tuple[list[int], list[int]]:
    """
    **横剖面尖峰定义（细竖线）**：

    - 剖面 ``p(x)`` 与 ``thin_vertical_lines_universal`` 主路径相同（电池行加权 |gx| + 压边带）；
    - 中间区域按 ``expected_n`` **等分**，每档 **至多一个** 尖峰，体现 **均匀出现**；
    - 尖峰须为局部极大，且 **峰值 ≥ peak_bg_ratio × 局部鞍部背景**（默认 1.5）；
    - 默认先取档内 **最高** 尖峰，半高宽与同类 **相差过大** 时在档内改选更窄者；
    - 返回各尖峰列号与半高宽（跨像素），供拼图竖线与统计。
    """
    if expected_n < 1:
        return [], []
    p, min_col_dist = _vertical_gx_column_profile(
        gray, None, row_weights=row_weights, column_keep=column_keep
    )
    w = len(p)
    margin = max(int(w * interior_margin_frac), min_col_dist, 2)
    x0, x1 = margin, w - margin
    if x1 - x0 < expected_n * 4:
        return [], []

    xs: list[int] = []
    spans: list[int] = []

    for k in range(expected_n):
        a, b = _bin_x_range(x0, x1, expected_n, k)
        picked = None
        for relax in (0.0, 0.08, 0.16):
            ratio = max(1.05, float(peak_bg_ratio) - relax)
            picked = _pick_sharp_peak_in_bin(p, a, b, min_col_dist, ratio)
            if picked is not None:
                break
        if picked is None:
            continue
        xi, spi = picked
        xs.append(xi)
        spans.append(spi)

    if len(xs) != expected_n:
        return xs, spans

    W_bg = max(5, min_col_dist // 3)
    for _ in range(expected_n * 2):
        med_sp = float(np.median(spans))
        outlier_k = [
            k
            for k, s in enumerate(spans)
            if float(s) > float(span_vs_median_max) * med_sp
            or (max_half_width is not None and float(s) > float(max_half_width))
        ]
        if not outlier_k:
            break
        k = int(outlier_k[0])
        a, b = _bin_x_range(x0, x1, expected_n, k)
        span_limit = int(
            max(12.0, med_sp * 1.72, float(np.median(spans)) * 1.72)
        )
        if max_half_width is not None:
            span_limit = min(span_limit, int(max_half_width))
        span_limit = min(span_limit, 130)
        alt: tuple[float, float, int, int] | None = None
        for i in range(max(1, a + 1), min(w - 2, b)):
            if not (float(p[i]) >= float(p[i - 1]) and float(p[i]) >= float(p[i + 1])):
                continue
            bg = _local_bg_saddle(p, i, W_bg)
            if bg < 1e-12:
                continue
            if float(p[i]) < float(peak_bg_ratio) * bg * 0.92:
                continue
            sp = int(_spans_at_peaks_on_profile(p, [i])[0])
            if sp > span_limit:
                continue
            if any(j != k and abs(i - xs[j]) < min_col_dist for j in range(len(xs))):
                continue
            tup = (float(p[i]), -float(sp), i, sp)
            if alt is None or tup[:2] > alt[:2]:
                alt = tup
        if alt is None:
            break
        xs[k] = alt[2]
        spans[k] = alt[3]

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [spans[i] for i in order]


def _wide_vertical_seam_indices(
    spans: list[int],
    max_half_width: int | None,
    *,
    vs_median_ratio: float = 2.15,
    floor_px: int = 86,
    adaptive_ceiling_px: float = 125.0,
) -> list[int]:
    """
    判定「过宽」峰：若给定 ``max_half_width`` 则 ``span > max_half_width``；
    否则 ``span > max(floor_px, ratio * median)`` 且 **不超过** ``adaptive_ceiling_px`` 的软上限，
    避免「全员大肩台」时中位数被抬高导致谁也判不成宽峰。
    """
    if not spans:
        return []
    med = float(np.median(spans))
    if max_half_width is not None:
        cap = float(max_half_width)
    else:
        cap = max(float(floor_px), vs_median_ratio * med)
        cap = min(cap, float(adaptive_ceiling_px))
    return [i for i, s in enumerate(spans) if float(s) > cap]


def _narrow_halfwidth_cap_for_candidates(
    spans: list[int],
    skip_idx: int,
    max_half_width: int | None,
) -> float:
    """替换边带峰时，候选缝的半高宽须不超过此值（略宽于其余真缝的中位数，或物理上限）。"""
    others = [spans[j] for j in range(len(spans)) if j != skip_idx]
    med = float(np.median(others)) if others else 42.0
    if max_half_width is not None:
        return float(max_half_width)
    return max(36.0, min(130.0, med * 1.7))


def _refine_wide_vertical_seam_peaks(
    gray: np.ndarray,
    row_weights: np.ndarray | None,
    column_keep: np.ndarray | None,
    xs: list[int],
    spans: list[int],
    *,
    peak_prominence: float,
    max_half_width: int | None,
    target_n: int,
) -> tuple[list[int], list[int]]:
    """
    在已为 ``target_n`` 条的峰中，将 **半高宽异常大**（边带/汇流条肩台）的峰挖掉重搜，
    用 **突起度合格** 且 **半高宽与细缝一致** 的候选替换；**不**用全局幅度阈值删峰。
    """
    if not xs or len(xs) != len(spans) or len(xs) != target_n:
        return xs, spans
    p, min_col_dist = _vertical_gx_column_profile(
        gray, None, row_weights=row_weights, column_keep=column_keep
    )
    xs = list(xs)
    spans = list(spans)
    for _ in range(target_n * 3):
        wide_idx = _wide_vertical_seam_indices(spans, max_half_width)
        if not wide_idx:
            break
        i = int(max(wide_idx, key=lambda j: spans[j]))
        narrow_cap = _narrow_halfwidth_cap_for_candidates(spans, i, max_half_width)
        x_bad = xs[i]
        R = int(max(min_col_dist * 2, spans[i] * 0.5, 140))
        p_masked = p.copy()
        lo, hi = max(0, x_bad - R), min(len(p), x_bad + R + 1)
        p_masked[lo:hi] = 0.0
        prom_try = max(0.05, float(peak_prominence) * 0.88)
        cand_x = _greedy_peaks_local_prominence(p_masked.copy(), min_col_dist, prom_try)
        cand_spans = _spans_at_peaks_on_profile(p, [int(x) for x in cand_x])
        others = [xs[j] for j in range(len(xs)) if j != i]
        best: tuple[float, int, int] | None = None
        for x, s in zip(cand_x, cand_spans):
            if float(s) > narrow_cap:
                continue
            if any(abs(int(x) - ox) < min_col_dist for ox in others):
                continue
            score = float(p[int(x)])
            if best is None or score > best[0]:
                best = (score, int(x), int(s))
        if best is None:
            break
        _, x_new, s_new = best
        xs[i] = x_new
        spans[i] = s_new
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    return [xs[k] for k in order], [spans[k] for k in order]


def _thin_vertical_core(
    gray: np.ndarray,
    min_col_dist: int | None,
    peak_rel: float,
    max_peak_span: int | None,
    robust_peaks: bool,
    *,
    row_weights: np.ndarray | None = None,
    column_keep: np.ndarray | None = None,
) -> tuple[list[int], list[int]]:
    """
    row_weights：形状 (h,) 非负，对各行 gx 加权后再做列均值（和为 0 时退回全图均值）。
    用于只在「电池片区」行上积分，避免多行带各自取峰导致竖缝条数翻倍。

    column_keep：形状 (w,) 非负，逐列乘到 gx 上（典型为 0/1，去掉左右汇流条带）。
    """
    p, min_col_dist = _vertical_gx_column_profile(
        gray, min_col_dist, row_weights=row_weights, column_keep=column_keep
    )
    if robust_peaks:
        xs = _greedy_peaks_local_prominence(p.copy(), min_col_dist, peak_rel)
    else:
        xs = greedy_peaks(p.copy(), min_col_dist, peak_rel)
    spans = _spans_at_peaks_on_profile(p, xs)
    if max_peak_span is not None:
        pairs = [(x, s) for x, s in zip(xs, spans) if s <= max_peak_span]
        if pairs:
            xs = [a for a, _ in pairs]
            spans = [b for _, b in pairs]
    return xs, spans


def _merge_closest_peak_pairs(
    xs: list[int],
    spans: list[int],
    target_n: int,
) -> tuple[list[int], list[int]]:
    """按 x 排序后反复合并「相邻间距最小」的一对，直到剩 target_n 条（面板每图 6 竖缝等先验）。"""
    if len(xs) <= target_n:
        return xs, spans
    pairs: list[list[float]] = [[float(x), float(s)] for x, s in zip(xs, spans)]
    while len(pairs) > target_n:
        pairs.sort(key=lambda t: t[0])
        if len(pairs) < 2:
            break
        best_i = 0
        best_gap = pairs[1][0] - pairs[0][0]
        for i in range(1, len(pairs) - 1):
            g = pairs[i + 1][0] - pairs[i][0]
            if g < best_gap:
                best_gap = g
                best_i = i
        a, b = pairs[best_i], pairs[best_i + 1]
        merged = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
        pairs = pairs[:best_i] + [merged] + pairs[best_i + 2 :]
    out_x = [int(round(t[0])) for t in pairs]
    out_s = [int(max(1, round(t[1]))) for t in pairs]
    return out_x, out_s


def vertical_seam_horizontal_profiles(
    gray: np.ndarray,
    *,
    smooth_gx_win: int | None = None,
    mask_edge_busbars: bool = True,
) -> dict[str, np.ndarray]:
    """
    **细竖线 · 横剖面**（沿 **x** 的一条曲线，不是「某一行像素」而是 **电池区所有行** 在列向上的加权平均）：

    - 用行均值 99% 分位得 **汇流条行**，在 **非汇流条行** 上对每列求加权平均；
    - 给出 **灰度**、**255−灰**、**|∂I/∂x|**（Sobel-x，与 ``thin_vertical_lines`` 同模糊核）列曲线；
    - ``gx_smooth`` 与 ``thin_vertical_lines_universal`` 内部平滑尺度一致（除非传入 ``smooth_gx_win``）。
    - ``mask_edge_busbars=True`` 时，|gx| 列与通用检测相同，**压掉左右汇流条带**，便于与红虚线对照。

    规律上：**竖缝**处常见 **灰度谷**、**255−灰峰**、**|gx| 峰**；检出线应落在 |gx| 平滑剖面的峰附近。
    """
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    bus = find_busbar_row_mask(row_mean, 99.0)
    cell = (~bus).astype(np.float64)
    ws = float(cell.sum()) + 1e-6
    g = gray.astype(np.float64)
    inv = 255.0 - g
    gray_mean = (g * cell[:, None]).sum(axis=0) / ws
    inv_mean = (inv * cell[:, None]).sum(axis=0) / ws
    blur = cv2.GaussianBlur(gray, (5, 21), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)).astype(np.float64)
    if mask_edge_busbars and ws > 32:
        ck = _vertical_seam_column_keep_mask(gray, cell)
        gx = gx * ck[None, :]
    gx_mean = (gx * cell[:, None]).sum(axis=0) / ws
    min_col_dist = max(35, w // 32)
    sw = max(7, min_col_dist // 5)
    if smooth_gx_win is not None:
        sw = max(3, int(smooth_gx_win))
        if sw % 2 == 0:
            sw += 1
    gx_smooth = smooth_1d(gx_mean, sw)
    x = np.arange(w, dtype=np.float64)
    return {
        "x": x,
        "gray_mean": gray_mean,
        "inv_mean": inv_mean,
        "gx_mean": gx_mean,
        "gx_smooth": gx_smooth,
    }


def horizontal_seam_vertical_profiles(
    gray: np.ndarray,
    *,
    smooth_gy_win: int | None = None,
) -> dict[str, np.ndarray]:
    """
    **细横线 · 纵剖面**（沿 **y/行**）：与 ``task2_gap_measurement`` / ``compute_lines`` 一致，
    对 ``GaussianBlur(5,5)`` 后 ``|∂I/∂y|`` 在 **汇流条行置零** 后按列取平均，得行曲线；
    另给每行灰度、255−灰 的列均值。用于与 ``thin_horizontal_lines_gap_gradient`` 检出的 y 对照。
    """
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    bus = find_busbar_row_mask(row_mean, 99.0)
    g = gray.astype(np.float64)
    gray_mean = g.mean(axis=1)
    inv_mean = (255.0 - g).mean(axis=1)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)).astype(np.float64)
    gy[bus, :] = 0.0
    gy_mean = gy.mean(axis=1)
    min_row_dist = max(35, h // 32)
    sw = max(7, min_row_dist // 5)
    if smooth_gy_win is not None:
        sw = max(3, int(smooth_gy_win))
        if sw % 2 == 0:
            sw += 1
    gy_smooth = smooth_1d(gy_mean.astype(np.float64), sw)
    y = np.arange(h, dtype=np.float64)
    return {
        "y": y,
        "gray_mean": gray_mean,
        "inv_mean": inv_mean,
        "gy_mean": gy_mean,
        "gy_smooth": gy_smooth,
    }


def _cluster_x_peaks(pairs: list[tuple[int, int]], sep: int) -> list[tuple[int, int]]:
    """将跨行带重复的竖缝 x 合并：sep 内取中位 x、中位半高宽。"""
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda t: t[0])
    clusters: list[list[tuple[int, int]]] = [[pairs[0]]]
    for x, s in pairs[1:]:
        if x - clusters[-1][-1][0] > sep:
            clusters.append([(x, s)])
        else:
            clusters[-1].append((x, s))
    out: list[tuple[int, int]] = []
    for cl in clusters:
        xs = [a for a, _ in cl]
        ss = [b for _, b in cl]
        out.append((int(np.median(xs)), int(np.median(ss))))
    return out


def thin_vertical_lines(
    gray: np.ndarray,
    min_col_dist: int | None = None,
    peak_rel: float = 0.32,
    max_peak_span: int | None = None,
) -> tuple[list[int], list[int]]:
    """细竖线：|∂I/∂x| 列投影峰 + 半高宽（跨列像素）。max_peak_span 可滤掉过宽的伪峰。"""
    return _thin_vertical_core(
        gray, min_col_dist, peak_rel, max_peak_span, robust_peaks=False, row_weights=None
    )


def thin_vertical_lines_universal(
    gray: np.ndarray,
    *,
    peak_rel: float = 0.14,
    peak_bg_ratio: float = THIN_VLINE_PEAK_BG_RATIO,
    mutation_bg_relax_deltas: tuple[float, ...] | None = None,
    max_peak_span: int | None = None,
    merge_sep_px: int | None = None,
    min_strip_h: int | None = None,
    expected_vertical_count: int | None = 6,
) -> tuple[list[int], list[int]]:
    """
    **通用细竖线（推荐 a））**：电池区行加权 ``|∂I/∂x|`` 列剖面 + 左右竖汇流条列掩膜。

    **主路径（电池区足够时）**：突变尖峰须 **形状**（单峰无凹槽）+ **幅度**
    ``p[峰] ≥ peak_bg_ratio × 鞍部背景``（默认 ``THIN_VLINE_PEAK_BG_RATIO``，常 1.5）。
    按 ``mutation_bg_relax_deltas``（默认 ``THIN_VLINE_MUTATION_BG_RELAX_DELTAS``）对
    ``peak_bg_ratio`` **由严到松** 重试，**取最先非空** 的检出（通常条数更少、伪峰更少；
    若全空则再走回退）。可调 ``peak_bg_ratio`` 或 ``mutation_bg_relax_deltas`` 做折中。

    **跨像素** 为突变区间 ``R-L+1``；**不强制条数**；NMS 去近距重复。

    ``max_peak_span``：若给定，作 **突变宽度上限**（约 ``3×``）。

    **回退**：突变路径全空时，退回 ``peak_rel`` 邻域突起 + 取峰。

    ``expected_vertical_count``：兼容旧调用，**不用于强制条数**。

    ``merge_sep_px`` / ``min_strip_h``：仅回退路径、电池区行过少时用。
    """
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    bus = find_busbar_row_mask(row_mean, 99.0)
    cell_rows = (~bus).astype(np.float64)
    n_cell = int(cell_rows.sum())
    col_keep = (
        _vertical_seam_column_keep_mask(gray, cell_rows)
        if n_cell > 32
        else np.ones(w, dtype=np.float64)
    )

    xs: list[int] = []
    spans: list[int] = []
    mw: int | None = None
    if max_peak_span is not None:
        mw = max(120, int(max_peak_span * 3))

    if n_cell > 32:
        relax_seq = (
            mutation_bg_relax_deltas
            if mutation_bg_relax_deltas is not None
            else THIN_VLINE_MUTATION_BG_RELAX_DELTAS
        )
        for rdelta in relax_seq:
            ratio = max(1.02, float(peak_bg_ratio) - float(rdelta))
            xs, spans = _thin_vertical_mutation_peaks_all(
                gray,
                cell_rows,
                col_keep,
                peak_bg_ratio=ratio,
                interior_margin_frac=THIN_VLINE_INTERIOR_MARGIN_FRAC,
                max_mutation_width=mw,
            )
            if xs:
                break

    if not xs:
        if n_cell >= max(64, h // 25):
            xs, spans = _thin_vertical_core(
                gray,
                None,
                peak_rel,
                None,
                robust_peaks=True,
                row_weights=cell_rows,
                column_keep=col_keep,
            )
        else:
            mh = min_strip_h if min_strip_h is not None else max(220, min(360, int(0.22 * h)))
            rows = cell_row_intervals(h, bus, min_h=mh)
            sep = merge_sep_px if merge_sep_px is not None else max(18, min(42, w // 28))
            acc: list[tuple[int, int]] = []
            if rows:
                for ya, yb in rows:
                    strip = gray[ya:yb, :]
                    x2, sp2 = _thin_vertical_core(
                        strip,
                        None,
                        peak_rel,
                        None,
                        robust_peaks=True,
                        column_keep=col_keep,
                    )
                    acc.extend(zip(x2, sp2))
                merged = _cluster_x_peaks(acc, sep)
            else:
                x2, sp2 = _thin_vertical_core(
                    gray, None, peak_rel, None, robust_peaks=True, column_keep=col_keep
                )
                merged = list(zip(x2, sp2))
            xs = [t[0] for t in merged]
            spans = [t[1] for t in merged]

    return xs, spans


def thin_vertical_lines_gap_gradient(bgr: np.ndarray) -> tuple[list[int], list[int]]:
    """
    **细竖线位置 + 跨像素**（与 ``task2_gap_measurement.py`` 一致）：

    - ``compute_lines``：列能量峰得 x，并剔除 **宽竖线** 簇内峰；
    - ``measure_vline_gap_gradient``：列均值 **带符号梯度** 的正峰与负峰间距 → **间隙跨像素**
      （非 |gx| 半高宽）。
    """
    gray, xs_thin, _wide_vspan, _ys = compute_lines(bgr)
    xs_out: list[int] = []
    spans: list[int] = []
    for x in xs_thin:
        gap, _xl, _xr = measure_vline_gap_gradient(gray, int(x))
        xs_out.append(int(x))
        spans.append(int(max(1, gap)))
    return xs_out, spans


def thin_horizontal_lines_gap_gradient(
    bgr: np.ndarray,
    image_slot: int | None = None,
    *,
    thin_max_span: int = THIN_HORIZ_MAX_SPAN_FOR_STACK_PAIR,
    peak_rel: float = 0.30,
) -> tuple[list[int], list[int]]:
    """
    **细横线 y + 跨像素**（与 ``task2_gap_measurement.py`` 一致）：

    - ``compute_lines`` 在 **行能量** 峰上得 y，并已 **剔除一道粗横条能量带**（与竖向 ``wide_vspan`` 对偶，弱化边缘粗横线）；
    - ``measure_hline_gap_gradient``：行均值 **带符号梯度** 正峰与负峰间距 → **横缝跨像素**；
    - 传入 ``image_slot``（``Path`` 的 ``_slot_key``）时，按整板 3×4 与 ``thin_horizontal_lines_stacked_pair``
      相同规则保留 **上行/下行 2 条、中行 3 条**（细跨距优先、非中行取 **y 最大** 的若干条）。
    """
    gray, _xs, _wide_vspan, ys = compute_lines(bgr)
    ys_out: list[int] = []
    spans: list[int] = []
    for y in sorted(ys):
        gap, _t, _b = measure_hline_gap_gradient(gray, int(y))
        ys_out.append(int(y))
        spans.append(int(max(1, gap)))

    exp = expected_horizontal_seam_count(image_slot)
    if exp is None:
        return ys_out, spans

    ys = ys_out
    row = int(image_slot) // 4

    def _fallback_more_peaks() -> tuple[list[int], list[int]]:
        rel_fb = max(0.18, float(peak_rel) * 0.88)
        ys_u, _sp = thin_horizontal_lines_universal(gray, peak_rel=rel_fb)
        yo: list[int] = []
        sp: list[int] = []
        for y in sorted(ys_u):
            gap, _, _ = measure_hline_gap_gradient(gray, int(y))
            yo.append(int(y))
            sp.append(int(max(1, gap)))
        return yo, sp

    if row != 1:
        thin = [(y, s) for y, s in zip(ys, spans) if s <= thin_max_span]
        thin_sorted = sorted(thin, key=lambda t: t[0])
        if len(thin_sorted) >= exp:
            picked = thin_sorted[-exp:]
            return [int(t[0]) for t in picked], [int(t[1]) for t in picked]
        if len(ys) >= exp:
            y2, s2 = _merge_closest_peak_pairs(list(ys), list(spans), exp)
            return y2, s2
        ys3, sp3 = _fallback_more_peaks()
        if len(ys3) >= exp:
            if len(ys3) > exp:
                ys3, sp3 = _merge_closest_peak_pairs(ys3, sp3, exp)
            return ys3, sp3
        return ys, spans

    if len(ys) > exp:
        return _merge_closest_peak_pairs(list(ys), list(spans), exp)
    if len(ys) < exp:
        ys4, sp4 = _fallback_more_peaks()
        if len(ys4) >= exp:
            ys, spans = ys4, sp4
            if len(ys) > exp:
                return _merge_closest_peak_pairs(ys, spans, exp)
    return ys, spans


def thin_horizontal_lines_universal(
    gray: np.ndarray,
    *,
    peak_rel: float = 0.30,
) -> tuple[list[int], list[int]]:
    """
    **通用细横线（推荐 b））**：对 **汇流条行**（行均值亮于 99% 分位）在 ``|∂I/∂y|`` 上
    整行置零后再做行均值投影，削弱汇流条大横边造成的假横缝峰；取峰同用 robust 屋顶。
    """
    h, w = gray.shape
    if h < 8 or w < 8:
        return [], []
    min_row_dist = max(35, h // 32)
    row_mean = gray.mean(axis=1)
    bus = find_busbar_row_mask(row_mean, 99.0)
    blur = cv2.GaussianBlur(gray, (21, 5), 0)
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    gy = np.asarray(gy, dtype=np.float64)
    gy[bus, :] = 0.0
    p = smooth_1d(gy.mean(axis=1), max(7, min_row_dist // 5))
    ys = _greedy_peaks_robust(p.copy(), min_row_dist, peak_rel)
    base = float(np.percentile(p, 28))
    spans: list[int] = []
    for y in ys:
        vmax = float(p[y])
        if vmax <= base + 1.0:
            spans.append(1)
            continue
        thr_h = base + 0.5 * (vmax - base)
        lo = hi = y
        while lo > 0 and p[lo - 1] >= thr_h:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr_h:
            hi += 1
        spans.append(int(hi - lo + 1))
    return ys, spans


def _slot_key(p: Path) -> int:
    m = re.search(r"_(\d+)\.(jpg|jpeg|png)$", p.name, re.I)
    return int(m.group(1)) if m else -1


def task1_set_paths(set_name: str) -> list[Path]:
    """``task1/{set_name}`` 下按 `_序号` 排序的 jpg。"""
    root = TASK1 / set_name
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jpg"), key=_slot_key)


def set1_paths() -> list[Path]:
    """按 `_序号` 排序的 12 张图（``task1/set1``）。"""
    return task1_set_paths("set1")


# 相机4 图右缘在 |gx| 投影上易出现半高宽极大的伪峰；真细缝半高宽约 34–40
CAM4_THIN_VLINE_MAX_PEAK_SPAN = 55
# set1 文件名 `*_0_N.jpg` 的序号 N；与 cam_paths(4) 一致
CAM4_IMAGE_SLOTS: tuple[int, ...] = (3, 7, 11)


def cam_paths(cam: Literal[1, 2, 3, 4], set_name: str = "set1") -> list[Path]:
    idx = {1: (0, 4, 8), 2: (1, 5, 9), 3: (2, 6, 10), 4: (3, 7, 11)}[cam]
    by_n = {_slot_key(p): p for p in task1_set_paths(set_name)}
    return [by_n[n] for n in idx if n in by_n]


def cam14_horizontal_edge_kinds(slot_n: int) -> list[Literal["top", "bottom"]]:
    """
    相机 1、4 外缘图：按槽位在整板 3×4 中的行决定是否补充 **横向** 外缘汇流条剖面。
    上行 tile（序号 0–3）→ 上缘；下行 tile（8–11）→ 下缘；中行（4–7）仅保留左右竖向，不切行向 ROI。
    """
    row = int(slot_n) // 4
    if row == 0:
        return ["top"]
    if row == 2:
        return ["bottom"]
    return []


def median_cell_band_height_px(paths: list[Path] | None = None) -> float:
    """多图行带高度中位数 → 用于 210mm 标定。"""
    paths = paths or set1_paths()
    bands: list[float] = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        h = g.shape[0]
        row_mean = g.mean(axis=1)
        bm = find_busbar_row_mask(row_mean, 99.0)
        for ya, yb in cell_row_intervals(h, bm, min_h=350):
            bands.append(float(yb - ya))
    return float(np.median(bands)) if bands else float("nan")


def mm_per_px(paths: list[Path] | None = None) -> tuple[float, float]:
    """返回 (行带高度中位数 px, mm/px)。"""
    med = median_cell_band_height_px(paths)
    scale = CELL_HEIGHT_MM / med if med > 1e-6 else float("nan")
    return med, scale


def first_row_calibrated_mm_per_px(set_name: str = "set1") -> float:
    """该集排序首张图 `first_cell_row_height_px` → ``210 mm / h``（notebook 标定）。"""
    paths = task1_set_paths(set_name)
    if not paths:
        return float("nan")
    fp = paths[0]
    gray = cv2.cvtColor(load_bgr(fp), cv2.COLOR_BGR2GRAY)
    _y0, _y1, h_px = first_cell_row_height_px(gray)
    return float(CELL_HEIGHT_MM / h_px) if h_px > 1e-6 else float("nan")


def minmax_px_mm_with_source(
    pairs: list[tuple[str, float]],
    mm_per_px: float,
    title: str,
) -> None:
    """打印最小/最大像素值及换算 mm，并标注对应样本键（多为文件名或文件名#缝序）。"""
    pairs_f = [(n, float(v)) for n, v in pairs if math.isfinite(v) and v > 0]
    if not pairs_f:
        print(f"{title}: 无有效样本")
        return
    if not math.isfinite(mm_per_px) or mm_per_px <= 0:
        print(f"{title}: mm/px 无效，跳过换算")
        return
    vals = np.asarray([v for _, v in pairs_f], dtype=np.float64)
    i_mn = int(vals.argmin())
    i_mx = int(vals.argmax())
    mn, mx = float(vals[i_mn]), float(vals[i_mx])
    nm_mn, nm_mx = pairs_f[i_mn][0], pairs_f[i_mx][0]
    print(f"{title}")
    print(f"  最小: {mn:.2f} px  ≈  {mn * mm_per_px:.3f} mm  （{nm_mn}）")
    print(f"  最大: {mx:.2f} px  ≈  {mx * mm_per_px:.3f} mm  （{nm_mx}）")


def collect_a_halfwidth_pairs(set_name: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for p in task1_set_paths(set_name):
        bgr = load_bgr(p)
        _xs, spans = thin_vertical_lines_gap_gradient(bgr)
        for si, s in enumerate(spans):
            out.append((f"{p.name}#竖{si}", float(s)))
    return out


def collect_b_halfwidth_pairs(set_name: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for p in task1_set_paths(set_name):
        bgr = load_bgr(p)
        _ys, spans = thin_horizontal_lines_gap_gradient(bgr, _slot_key(p))
        for si, s in enumerate(spans):
            out.append((f"{p.name}#横{si}", float(s)))
    return out


def collect_c_width_pairs(set_name: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for p in cam_paths(1, set_name) + cam_paths(4, set_name):
        g = cv2.cvtColor(load_bgr(p), cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        sn = _slot_key(p)
        mode = "left" if sn % 4 == 0 else "right"
        x0, x1 = busbar_search_roi_cam14(w, mode)
        min_run = (
            32 if mode == "left" and sn in CAM1_LEFT_BROAD_PEAK_SLOTS else None
        )
        _prof, e = busbar_edges_from_column_mean(
            g, x0, x1, min_peak_run_px=min_run
        )
        if not math.isnan(e.thr_big) and e.width_inner_px > 0:
            out.append((f"{p.name}:外缘竖向", float(e.width_inner_px)))
        for hk in cam14_horizontal_edge_kinds(sn):
            try:
                y0, y1 = busbar_search_roi_row_top_bottom(h, hk)
                min_run_h = (
                    32
                    if mode == "left" and sn in CAM1_LEFT_BROAD_PEAK_SLOTS
                    else None
                )
                _pr, er = busbar_edges_from_row_mean(
                    g, y0, y1, min_peak_run_px=min_run_h
                )
                if math.isnan(er.thr_big) or er.width_inner_px <= 0:
                    continue
                tag = "上缘横向" if hk == "top" else "下缘横向"
                out.append((f"{p.name}:{tag}", float(er.width_inner_px)))
            except Exception:
                continue
    return out


def collect_d_width_pairs(set_name: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for p in cam_paths(2, set_name) + cam_paths(3, set_name):
        g = cv2.cvtColor(load_bgr(p), cv2.COLOR_BGR2GRAY)
        _prof, e = busbar_edges_from_column_mean(g, 0, g.shape[1])
        out.append((p.name, float(e.width_inner_px)))
    return out


def print_abcd_minmax_summary_for_set(set_name: str) -> None:
    """a 为 task2_gap_measurement 梯度跨距；b 为通用细横线半高宽；c/d 不变。"""
    mm = first_row_calibrated_mm_per_px(set_name)
    print(f"======== {set_name} · 四项极值（a 梯度跨距 / b–d 同前）========")
    print(f"标定：该集排序首张 → mm/px = {mm:.6f}")
    minmax_px_mm_with_source(
        collect_a_halfwidth_pairs(set_name), mm, "a) 左右并排·细竖缝梯度跨距（列梯度±峰间距）"
    )
    minmax_px_mm_with_source(
        collect_b_halfwidth_pairs(set_name),
        mm,
        "b) 上下叠放·细横缝梯度跨距（行梯度±峰间距，compute_lines y）",
    )
    minmax_px_mm_with_source(
        collect_c_width_pairs(set_name),
        mm,
        "c) 相机1/4·外缘汇流条宽（内缘，含上下横补）",
    )
    minmax_px_mm_with_source(
        collect_d_width_pairs(set_name), mm, "d) 相机2/3·中间汇流条宽（内缘）"
    )
    print()


def _hsv_color(i: int, n: int) -> tuple[int, int, int]:
    h = int(179 * i / max(n, 1)) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def paint_vertical_lines(bgr: np.ndarray, xs: list[int], thickness: int = 2) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    n = len(xs)
    for i, x in enumerate(xs):
        x = int(np.clip(x, 0, w - 1))
        cv2.line(out, (x, 0), (x, h - 1), _hsv_color(i, n), thickness)
    return out


def paint_horizontal_lines(bgr: np.ndarray, ys: list[int], thickness: int = 2) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    n = len(ys)
    for i, y in enumerate(ys):
        y = int(np.clip(y, 0, h - 1))
        cv2.line(out, (0, y), (w - 1, y), _hsv_color(i, n), thickness)
    return out


def paint_thick_vertical_bands(
    bgr: np.ndarray,
    xs: list[int],
    spans: list[int],
    alpha: float = 0.4,
) -> np.ndarray:
    """按跨列宽度画半透明竖条。"""
    out = bgr.copy()
    h, w = out.shape[:2]
    n = len(xs)
    for i, (x, sw) in enumerate(zip(xs, spans)):
        half = max(1, int(sw) // 2)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, 0), (x1 - 1, h - 1), _hsv_color(i, n), -1)
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    return out


def thick_vertical_lines_brightness(
    gray: np.ndarray,
    mode: Literal["left", "right", "center"],
    *,
    span_level_frac: float = 0.5,
    peak_rel: float = 0.38,
) -> tuple[list[int], list[int]]:
    """
    粗竖线：列均值亮度在 ROI 内取峰（汇流条偏亮），再算「水平宽度」。
    span_level_frac 越小，阈值越靠近基底，量出的跨列宽度越大（0.12 量级可反映汇流条全宽约 120+ px）。
    """
    _h, w = gray.shape
    col_m = gray.mean(axis=0)
    if mode == "left":
        x0, x1 = 0, max(48, int(w * 0.22))
        min_d = max(45, (x1 - x0) // 5)
    elif mode == "right":
        x0, x1 = min(w - 48, int(w * 0.78)), w
        min_d = max(45, (x1 - x0) // 5)
    else:
        x0, x1 = int(w * 0.36), int(w * 0.64)
        min_d = max(55, w // 14)
    seg = col_m[x0:x1].astype(np.float64)
    if seg.size < 5:
        return [], []
    p = smooth_1d(seg, min(25, max(9, seg.size // 6)))
    peaks = greedy_peaks(p.copy(), min_d, peak_rel)
    xs = [x0 + int(pk) for pk in peaks]
    base = float(np.percentile(p, 22))
    lvl = float(np.clip(span_level_frac, 0.05, 0.95))
    spans: list[int] = []
    for pk in peaks:
        vmax = float(p[pk])
        if vmax <= base + 1.0:
            spans.append(1)
            continue
        thr = base + lvl * (vmax - base)
        lo = hi = int(pk)
        while lo > 0 and p[lo - 1] >= thr:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr:
            hi += 1
        spans.append(int(hi - lo + 1))
    return xs, spans


def thick_vertical_busbar_fullwidth(
    gray: np.ndarray,
    *,
    min_span: int = 120,
    max_span_frac: float = 0.22,
    span_level_frac: float = 0.12,
    peak_rel: float = 0.38,
) -> tuple[list[int], list[int]]:
    """
    全幅列均值：找 **一根** 最亮的汇流条粗竖线（跨列宽度用 span_level_frac 度量），
    且宽度 ∈ [min_span, w * max_span_frac]，用于相机 2/3（汇流条不一定在画面正中）。
    """
    _h, w = gray.shape
    col_m = gray.mean(axis=0)
    seg = col_m.astype(np.float64)
    p = smooth_1d(seg, min(25, max(9, w // 40)))
    min_d = max(55, w // 14)
    peaks = greedy_peaks(p.copy(), min_d, peak_rel)
    base = float(np.percentile(p, 22))
    lvl = float(np.clip(span_level_frac, 0.05, 0.95))
    max_span_px = int(max(min_span + 1, w * max_span_frac))
    best: tuple[int, int, float] | None = None
    for pk in peaks:
        pk = int(pk)
        vmax = float(p[pk])
        if vmax <= base + 1.0:
            continue
        thr = base + lvl * (vmax - base)
        lo = hi = pk
        while lo > 0 and p[lo - 1] >= thr:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] >= thr:
            hi += 1
        span = int(hi - lo + 1)
        if span < min_span or span > max_span_px:
            continue
        if best is None or vmax > best[2]:
            best = (pk, span, vmax)
    if best is None:
        return [], []
    x, span, _ = best
    return [int(x)], [int(span)]


def mosaic_grid(
    bgr_list: list[np.ndarray],
    nrows: int,
    ncols: int,
    pad: int = 4,
    bg: tuple[int, int, int] = (40, 40, 40),
    max_side: int | None = 2000,
) -> np.ndarray:
    """nrows×ncols 拼图，顺序行优先。max_side：输出最长边上限，便于 Notebook 显示。"""
    if len(bgr_list) != nrows * ncols:
        raise ValueError(f"需要 {nrows * ncols} 张，当前 {len(bgr_list)}")
    hs = [im.shape[0] for im in bgr_list]
    ws = [im.shape[1] for im in bgr_list]
    ch, cw = min(hs), min(ws)
    cells = [cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA) for im in bgr_list]
    gh = nrows * ch + (nrows - 1) * pad
    gw = ncols * cw + (ncols - 1) * pad
    grid = np.full((gh, gw, 3), bg, dtype=np.uint8)
    for k, cell in enumerate(cells):
        r, c = k // ncols, k % ncols
        y = r * (ch + pad)
        x = c * (cw + pad)
        grid[y : y + ch, x : x + cw] = cell
    if max_side and max(grid.shape[0], grid.shape[1]) > max_side:
        sc = max_side / max(grid.shape[0], grid.shape[1])
        grid = cv2.resize(
            grid,
            (int(grid.shape[1] * sc), int(grid.shape[0] * sc)),
            interpolation=cv2.INTER_AREA,
        )
    return grid


def mosaic_3x4(bgr_list: list[np.ndarray], **kw) -> np.ndarray:
    """12 张：3×4。"""
    return mosaic_grid(bgr_list, 3, 4, **kw)


def load_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return im
