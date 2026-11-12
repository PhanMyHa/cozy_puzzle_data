"""
Chuyển ảnh PNG thành SVG chia vùng màu cho game tô màu.

Mỗi vùng là 1 <path>:
    fill        = xám nhạt (chưa tô)
    data-color  = màu gốc của vùng (hiện ra khi tô đúng)
    data-index  = số thứ tự màu trong bảng màu (1..K) -> "màu tương đối"

Cài thư viện:  pip install pillow numpy scipy opencv-python

Cách dùng:
    python png_to_svg_regions.py item.png -o item.svg --colors 8
    python png_to_svg_regions.py item.png -o item.svg --colors 12 --min-area 80 --numbers

Tham số:
    --colors    : số màu trong bảng màu (K)
    --min-area  : vùng nhỏ hơn N pixel sẽ được gộp vào vùng bên cạnh
    --epsilon   : độ đơn giản hóa đường viền (lớn = ít điểm, góc cạnh hơn)
    --max-size  : thu nhỏ ảnh nếu cạnh dài hơn N pixel (0 = giữ nguyên)
    --numbers   : thêm số thứ tự màu vào giữa mỗi vùng (kiểu tô theo số)
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

EIGHT = np.ones((3, 3), int)


def load(path, max_size, alpha_thresh):
    img = Image.open(path).convert("RGBA")
    if max_size and max(img.size) > max_size:
        s = max_size / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    arr = np.array(img)
    return arr[:, :, :3], arr[:, :, 3] > alpha_thresh


def quantize(rgb, mask, k):
    """K-means trong không gian Lab -> bản đồ cụm (-1 = ngoài item)."""
    smooth = cv2.bilateralFilter(rgb, 9, 40, 9)
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    pts = lab[mask]
    k = max(1, min(k, len(pts)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, _ = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cmap = np.full(mask.shape, -1, np.int32)
    cmap[mask] = lbl.flatten()
    return cmap, k


def label_regions(cmap, k):
    """Tách mỗi cụm màu thành các vùng liền nhau. Trả về (region_map, số vùng)."""
    region = np.zeros(cmap.shape, np.int32)
    n = 0
    for c in range(k):
        cc, m = ndimage.label(cmap == c, structure=EIGHT)
        region[cc > 0] = cc[cc > 0] + n
        n += m
    return region, n


def absorb_small(cmap, mask, k, min_area):
    """Vùng quá nhỏ -> nhận màu của pixel hợp lệ gần nhất."""
    for _ in range(3):
        region, n = label_regions(cmap, k)
        areas = np.bincount(region.ravel(), minlength=n + 1)
        small = (areas[region] < min_area) & mask
        valid = mask & ~small
        if not small.any() or not valid.any():
            break
        iy, ix = ndimage.distance_transform_edt(~valid, return_distances=False,
                                                return_indices=True)
        cmap = cmap.copy()
        cmap[small] = cmap[iy[small], ix[small]]
    return cmap


def to_path(m, eps, ox, oy):
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in contours:
        if len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, eps, True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2)
        parts.append("M" + " L".join(f"{x + ox} {y + oy}" for x, y in pts) + "Z")
    return " ".join(parts)


def convert(inp, out, colors, min_area, epsilon, max_size, numbers,
            stroke_width=1.0, stroke_color="#222222", alpha_thresh=10):
    rgb, mask = load(inp, max_size, alpha_thresh)
    h, w = mask.shape

    cmap, k = quantize(rgb, mask, colors)
    cmap = absorb_small(cmap, mask, k, min_area)
    region, n = label_regions(cmap, k)

    # Màu gốc của từng cụm = trung bình pixel gốc
    palette = []
    for c in range(k):
        sel = cmap == c
        if sel.any():
            r, g, b = rgb[sel].mean(axis=0).round().astype(int)
            palette.append("#%02x%02x%02x" % (r, g, b))
        else:
            palette.append("#000000")

    areas = np.bincount(region.ravel(), minlength=n + 1)
    slices = ndimage.find_objects(region)
    order = sorted(range(1, n + 1), key=lambda i: -areas[i])  # to vẽ trước, nhỏ đè lên

    kernel = np.ones((3, 3), np.uint8)
    paths, texts, counts = [], [], {}
    for rid in order:
        sl = slices[rid - 1]
        if sl is None or areas[rid] == 0:
            continue
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, h)
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, w)
        m = (region[y0:y1, x0:x1] == rid).astype(np.uint8)
        cidx = int(cmap[y0:y1, x0:x1][m > 0][0])

        # nở 1px để các vùng chồng nhẹ lên nhau, không bị khe hở trắng
        mg = cv2.dilate(m, kernel) & mask[y0:y1, x0:x1].astype(np.uint8)
        d = to_path(mg, epsilon, x0, y0)
        if not d:
            continue

        index = cidx + 1
        counts[index] = counts.get(index, 0) + 1
        paths.append(
            f'<path id="r{rid}" class="region" d="{d}" fill="#d9d9d9" '
            f'fill-rule="evenodd" data-index="{index}" data-color="{palette[cidx]}"'
            + (f' stroke="{stroke_color}" stroke-width="{stroke_width}" '
               f'stroke-linejoin="round"' if stroke_width > 0 else "")
            + "/>"
        )

        if numbers:
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            yy, xx = np.unravel_index(dist.argmax(), dist.shape)
            if dist[yy, xx] >= 4:
                fs = int(min(max(dist[yy, xx] * 1.1, 6), 30))
                texts.append(
                    f'<text class="num" x="{xx + x0}" y="{yy + y0}" font-size="{fs}" '
                    f'text-anchor="middle" dominant-baseline="central" '
                    f'fill="#555" pointer-events="none">{index}</text>'
                )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">',
        '<g id="regions">', *paths, "</g>",
    ]
    if texts:
        svg += ['<g id="numbers">', *texts, "</g>"]
    svg.append("</svg>")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    pal_path = os.path.splitext(out)[0] + "_palette.json"
    with open(pal_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"index": i + 1, "color": c, "regions": counts.get(i + 1, 0)}
             for i, c in enumerate(palette)],
            f, indent=2,
        )

    print(f"Đã lưu {out}: {len(paths)} vùng, {k} màu")
    print(f"Bảng màu: {pal_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default="output.svg")
    p.add_argument("--colors", type=int, default=8)
    p.add_argument("--min-area", type=int, default=60)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--max-size", type=int, default=1024)
    p.add_argument("--numbers", action="store_true")
    p.add_argument("--stroke-width", type=float, default=1.0,
                   help="độ dày viền (0 = không viền)")
    p.add_argument("--stroke-color", default="#222222")
    a = p.parse_args()
    convert(a.input, a.output, a.colors, a.min_area, a.epsilon, a.max_size,
            a.numbers, a.stroke_width, a.stroke_color)