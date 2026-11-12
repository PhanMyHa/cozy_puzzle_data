from pathlib import Path
from PIL import Image

# Thư mục chứa các folder ảnh: cat_puzzle, cat_slide, ...
ROOT_DIR = Path(__file__).resolve().parent

# Chất lượng WebP (0-100)
WEBP_QUALITY = 90

# False: giữ ảnh gốc sau khi chuyển đổi
# True: xóa ảnh gốc sau khi đã tạo WebP thành công
DELETE_ORIGINAL = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_folder(folder: Path):
    image_files = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.name.lower()
    )

    if not image_files:
        return

    print(f"\n[{folder.name}] Tìm thấy {len(image_files)} ảnh")

    for index, source in enumerate(image_files, start=1):
        destination = folder / f"{index:03d}.webp"

        try:
            with Image.open(source) as image:
                # Giữ ảnh có nền trong suốt; chuyển mode phù hợp với WebP
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

                image.save(
                    destination,
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    method=6
                )

            if DELETE_ORIGINAL:
                source.unlink()

            print(f"  OK: {source.name} -> {destination.name}")

        except Exception as error:
            print(f"  LỖI: {source.name}: {error}")


def main():
    if not ROOT_DIR.exists():
        print(f"Không tìm thấy thư mục: {ROOT_DIR}")
        return

    folders = sorted(
        (path for path in ROOT_DIR.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower()
    )

    if not folders:
        print("Không tìm thấy folder con nào.")
        return

    for folder in folders:
        convert_folder(folder)

    print("\nHoàn tất chuyển đổi.")


if __name__ == "__main__":
    main()
