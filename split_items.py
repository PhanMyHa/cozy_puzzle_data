"""
Tách các item riêng biệt trong 1 ảnh đã xóa nền (PNG trong suốt) thành nhiều ảnh.

Cài thư viện:  pip install pillow numpy scipy

Cách dùng:
    python split_items.py input.png -o output
    python split_items.py input.png -o output --merge-gap 15 --padding 10 --min-area 500

Tham số:
    --alpha-thresh : pixel có alpha > ngưỡng này được coi là thuộc item (mặc định 10)
    --merge-gap    : gộp các mảnh gần nhau trong khoảng N pixel thành 1 item
                     (hữu ích khi 1 item gồm nhiều mảnh rời, vd: chữ, dấu chấm...)
    --min-area     : bỏ qua các cụm nhỏ hơn N pixel (loại bỏ nhiễu/bụi)
    --padding      : chừa thêm N pixel viền trong suốt quanh mỗi item
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def split_items(input_path, output_dir, alpha_thresh=10, merge_gap=0,
                min_area=100, padding=0):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Mask các pixel thuộc item
    mask = arr[:, :, 3] > alpha_thresh

    # Gộp các mảnh gần nhau (nếu cần) bằng cách giãn mask
    if merge_gap > 0:
        grouping = ndimage.binary_dilation(
            mask, structure=np.ones((3, 3), bool), iterations=merge_gap
        )
    else:
        grouping = mask

    # Gán nhãn từng cụm liền nhau (8 hướng)
    labels, n = ndimage.label(grouping, structure=np.ones((3, 3), int))
    slices = ndimage.find_objects(labels)

    items = []
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        area = int(mask[sl][labels[sl] == idx].sum())
        if area < min_area:
            continue
        items.append((idx, sl))

    # Sắp xếp từ trên xuống dưới, trái sang phải
    items.sort(key=lambda t: (t[1][0].start, t[1][1].start))

    saved = 0
    for idx, sl in items:
        y0 = max(sl[0].start - padding, 0)
        y1 = min(sl[0].stop + padding, h)
        x0 = max(sl[1].start - padding, 0)
        x1 = min(sl[1].stop + padding, w)

        crop = arr[y0:y1, x0:x1].copy()

        # Xóa phần của item khác nằm lọt trong khung crop
        belongs = labels[y0:y1, x0:x1] == idx
        if merge_gap > 0:
            # giữ cả pixel viền mờ nằm trong vùng giãn của item này
            belongs &= True
        crop[~belongs, 3] = 0

        saved += 1
        out_path = os.path.join(output_dir, f"item_{saved:03d}.png")
        Image.fromarray(crop, "RGBA").save(out_path)
        print(f"Đã lưu {out_path}  ({x1 - x0}x{y1 - y0})")

    print(f"\nTổng cộng: {saved} item -> thư mục '{output_dir}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tách item từ ảnh PNG nền trong suốt")
    p.add_argument("input", help="đường dẫn ảnh PNG đã tách nền")
    p.add_argument("-o", "--output", default="output", help="thư mục lưu kết quả")
    p.add_argument("--alpha-thresh", type=int, default=10)
    p.add_argument("--merge-gap", type=int, default=0)
    p.add_argument("--min-area", type=int, default=100)
    p.add_argument("--padding", type=int, default=0)
    a = p.parse_args()

    split_items(a.input, a.output, a.alpha_thresh, a.merge_gap,
                a.min_area, a.padding)