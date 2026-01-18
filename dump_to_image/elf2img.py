import struct
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# ============================================================
# 固定設定（論文準拠）
# ============================================================
IMG_SIZE = 256
TARGET_PIXELS = IMG_SIZE * IMG_SIZE

# ============================================================
# ELF → ロード可能メモリ領域（PT_LOAD）の抽出
# ============================================================
def extract_raw_memory(data: bytes) -> bytes:
    """
    ELF64 Program Header Table を解析し、
    PT_LOAD セグメントのみを連結して raw memory bytes を生成
    """
    # ELF64 header fields
    e_phoff     = struct.unpack_from("<Q", data, 0x20)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36)[0]
    e_phnum     = struct.unpack_from("<H", data, 0x38)[0]

    raw_bytes = bytearray()

    for i in range(e_phnum):
        ph_start = e_phoff + i * e_phentsize

        # p_type
        p_type = struct.unpack_from("<I", data, ph_start)[0]

        # PT_LOAD == 1
        if p_type == 1:
            p_offset = struct.unpack_from("<Q", data, ph_start + 8)[0]
            p_filesz = struct.unpack_from("<Q", data, ph_start + 32)[0]

            if p_filesz > 0:
                raw_bytes.extend(data[p_offset : p_offset + p_filesz])

    return bytes(raw_bytes)

# ============================================================
# Raw memory → RGB image array（論文準拠）
# ============================================================
def memory_to_image_array(mem_bytes: bytes) -> np.ndarray:
    """
    - 4 byte = 1 digital word
    - ARGB と解釈
    - Alpha を破棄し RGB のみ使用
    - 256×256 に切り詰め or 0-padding
    """
    arr = np.frombuffer(mem_bytes, dtype=np.uint8)

    # 4-byte alignment
    n = (arr.size // 4) * 4
    if n == 0:
        raise ValueError("memory dump too small")

    arr = arr[:n]

    # [N,4] digital words
    words = arr.reshape(-1, 4)

    # Ignore Alpha channel
    rgb = words[:, 1:4]

    num_pixels = rgb.shape[0]

    if num_pixels < TARGET_PIXELS:
        pad = np.zeros((TARGET_PIXELS - num_pixels, 3), dtype=np.uint8)
        rgb = np.vstack([rgb, pad])
    else:
        rgb = rgb[:TARGET_PIXELS]

    return rgb.reshape(IMG_SIZE, IMG_SIZE, 3).astype(np.uint8)

# ============================================================
# 指定 ID ディレクトリを処理（ファイル名に ID 付与）
# ============================================================
def process_by_id(input_root: Path, output_root: Path, target_id: str):
    input_dir  = input_root / target_id
    output_dir = output_root / target_id
    output_dir.mkdir(parents=True, exist_ok=True)

    elf_files = sorted(input_dir.glob("*.elf"))
    if not elf_files:
        print(f"[!] No ELF files in {input_dir}")
        return

    print(f"[*] Processing ID={target_id} | {len(elf_files)} ELF files")

    for elf_path in elf_files:
        try:
            with open(elf_path, "rb") as f:
                data = f.read()

            # ELF → raw memory
            mem_bytes = extract_raw_memory(data)

            # raw memory → image
            img_arr = memory_to_image_array(mem_bytes)
            img = Image.fromarray(img_arr, mode="RGB")

            # ★ ファイル名に ID を付加
            out_path = output_dir / f"{target_id}_{elf_path.stem}.jpg"
            img.save(out_path, format="JPEG", quality=95)

            print(f"[+] {elf_path.name} -> {out_path.name}")

        except Exception as e:
            print(f"[!] Failed {elf_path.name}: {e}")

    print(f"[*] Done ID={target_id}")

# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELF memory dump to RGB image (paper-aligned, ID-aware)"
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Root directory containing ID folders (e.g., D:\\dump)"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for output images"
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Target ID folder name (e.g., id02)"
    )

    args = parser.parse_args()

    process_by_id(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        target_id=args.id
    )

#  C:/Users/ace09/AppData/Local/Programs/Python/Python312/python.exe d:/mem/elf2img.py --input_root D:\dump\ --output_root D:\dump\images256 --id id02
