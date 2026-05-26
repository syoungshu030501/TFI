"""
将 train 目录按 8:2 比例分成 train_split 和 val 目录。
使用符号链接（symlink）避免重复占用磁盘空间。
随机种子固定为 42，确保可复现。

原始数据结构:
  train/
  ├── Black/  (800 samples)
  │   ├── Caption/  (.md)
  │   ├── Image/    (.jpg/.png)
  │   └── Mask/     (.png)
  └── White/  (200 samples)
      ├── Caption/  (.md)
      └── Image/    (.jpg)

输出结构:
  train_split/  (80%)
  ├── Black/
  │   ├── Caption/
  │   ├── Image/
  │   └── Mask/
  └── White/
      ├── Caption/
      └── Image/

  val/  (20%)
  ├── Black/
  │   ├── Caption/
  │   ├── Image/
  │   └── Mask/
  └── White/
      ├── Caption/
      └── Image/

注意: submit_example.csv 中的 500 张图全部对应 test 目录，与训练集无关，无需分割。
"""

import os
import random
import shutil
from pathlib import Path

# ======================== 配置 ========================
SEED = 42
TRAIN_RATIO = 0.8
BASE_DIR = Path(__file__).resolve().parent
ORIG_TRAIN_DIR = BASE_DIR / "train"
NEW_TRAIN_DIR = BASE_DIR / "train_split"
VAL_DIR = BASE_DIR / "val"
USE_SYMLINK = True  # True: 使用符号链接节省空间; False: 复制文件
# =====================================================

random.seed(SEED)


def get_sample_ids(category_dir: Path) -> list:
    """
    从 Image 目录中获取所有样本的 stem（去掉扩展名的文件名），
    同时记录原始扩展名，以便正确链接/复制 Image 文件。
    返回 [(stem, ext), ...] 列表。
    """
    image_dir = category_dir / "Image"
    samples = []
    for fname in sorted(os.listdir(image_dir)):
        stem, ext = os.path.splitext(fname)
        samples.append((stem, ext))
    return samples


def create_link_or_copy(src: Path, dst: Path):
    """创建符号链接或复制文件"""
    if USE_SYMLINK:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def split_and_link(category: str, subdirs: list):
    """
    对某个类别（Black/White）执行分割和链接/复制。
    
    Args:
        category: "Black" 或 "White"
        subdirs: 该类别下需要处理的子目录列表，如 ["Image", "Caption", "Mask"]
    """
    orig_cat_dir = ORIG_TRAIN_DIR / category
    samples = get_sample_ids(orig_cat_dir)
    
    # 随机打乱
    random.shuffle(samples)
    
    # 按 8:2 分割
    split_idx = int(len(samples) * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"\n{'='*50}")
    print(f"类别: {category}")
    print(f"  总样本数: {len(samples)}")
    print(f"  训练集:   {len(train_samples)}")
    print(f"  验证集:   {len(val_samples)}")
    
    # 为每个子目录创建目标文件夹并链接/复制文件
    for subdir in subdirs:
        orig_subdir = orig_cat_dir / subdir
        if not orig_subdir.exists():
            print(f"  [跳过] {subdir} 目录不存在")
            continue
        
        # 确定该子目录下文件的扩展名映射
        # Image 子目录用样本自身的扩展名，Caption 用 .md，Mask 用 .png
        ext_map = {}
        for fname in os.listdir(orig_subdir):
            stem, ext = os.path.splitext(fname)
            ext_map[stem] = ext
        
        # 创建 train_split 和 val 对应目录
        train_subdir = NEW_TRAIN_DIR / category / subdir
        val_subdir = VAL_DIR / category / subdir
        train_subdir.mkdir(parents=True, exist_ok=True)
        val_subdir.mkdir(parents=True, exist_ok=True)
        
        # 链接/复制 train 样本
        train_count = 0
        for stem, _ in train_samples:
            if stem in ext_map:
                src_file = orig_subdir / f"{stem}{ext_map[stem]}"
                dst_file = train_subdir / f"{stem}{ext_map[stem]}"
                if not dst_file.exists():
                    create_link_or_copy(src_file, dst_file)
                train_count += 1
        
        # 链接/复制 val 样本
        val_count = 0
        for stem, _ in val_samples:
            if stem in ext_map:
                src_file = orig_subdir / f"{stem}{ext_map[stem]}"
                dst_file = val_subdir / f"{stem}{ext_map[stem]}"
                if not dst_file.exists():
                    create_link_or_copy(src_file, dst_file)
                val_count += 1
        
        print(f"  {subdir}: train={train_count}, val={val_count}")


def verify_split():
    """验证分割结果的正确性"""
    print(f"\n{'='*50}")
    print("验证分割结果...")
    
    for category in ["Black", "White"]:
        orig_cat_dir = ORIG_TRAIN_DIR / category
        subdirs_to_check = [d for d in os.listdir(orig_cat_dir) 
                           if (orig_cat_dir / d).is_dir()]
        
        for subdir in subdirs_to_check:
            orig_files = set(os.listdir(orig_cat_dir / subdir))
            
            train_dir = NEW_TRAIN_DIR / category / subdir
            val_dir_path = VAL_DIR / category / subdir
            
            train_files = set(os.listdir(train_dir)) if train_dir.exists() else set()
            val_files = set(os.listdir(val_dir_path)) if val_dir_path.exists() else set()
            
            # 检查无交集
            overlap = train_files & val_files
            if overlap:
                print(f"  [错误] {category}/{subdir}: train 和 val 有 {len(overlap)} 个重叠文件!")
            
            # 检查并集 == 原始集合
            union = train_files | val_files
            if union != orig_files:
                missing = orig_files - union
                extra = union - orig_files
                if missing:
                    print(f"  [错误] {category}/{subdir}: 缺少 {len(missing)} 个文件")
                if extra:
                    print(f"  [错误] {category}/{subdir}: 多出 {len(extra)} 个文件")
            else:
                print(f"  [通过] {category}/{subdir}: "
                      f"orig={len(orig_files)}, "
                      f"train={len(train_files)}, "
                      f"val={len(val_files)}, "
                      f"无重叠，无遗漏")
    
    # 验证 Image/Caption/Mask 之间的对应关系
    print(f"\n检查文件对应关系...")
    for split_name, split_dir in [("train_split", NEW_TRAIN_DIR), ("val", VAL_DIR)]:
        for category in ["Black", "White"]:
            cat_dir = split_dir / category
            if not cat_dir.exists():
                continue
            
            # 获取各子目录的 stem 集合
            stems = {}
            for subdir in os.listdir(cat_dir):
                subdir_path = cat_dir / subdir
                if subdir_path.is_dir():
                    stems[subdir] = set(
                        os.path.splitext(f)[0] for f in os.listdir(subdir_path)
                    )
            
            # Image 和 Caption 的 stem 应完全一致
            if "Image" in stems and "Caption" in stems:
                if stems["Image"] == stems["Caption"]:
                    print(f"  [通过] {split_name}/{category}: Image 与 Caption 完全对应")
                else:
                    diff = stems["Image"].symmetric_difference(stems["Caption"])
                    print(f"  [错误] {split_name}/{category}: Image 与 Caption 不对应，差异 {len(diff)} 个")
            
            # Image 和 Mask 的 stem 应完全一致（仅 Black 有 Mask）
            if "Image" in stems and "Mask" in stems:
                if stems["Image"] == stems["Mask"]:
                    print(f"  [通过] {split_name}/{category}: Image 与 Mask 完全对应")
                else:
                    diff = stems["Image"].symmetric_difference(stems["Mask"])
                    print(f"  [错误] {split_name}/{category}: Image 与 Mask 不对应，差异 {len(diff)} 个")


def main():
    print("开始分割训练集...")
    print(f"随机种子: {SEED}")
    print(f"分割比例: train={TRAIN_RATIO}, val={1-TRAIN_RATIO}")
    print(f"使用符号链接: {USE_SYMLINK}")
    print(f"原始训练目录: {ORIG_TRAIN_DIR}")
    print(f"新训练目录:   {NEW_TRAIN_DIR}")
    print(f"验证集目录:   {VAL_DIR}")
    
    # Black: Image, Caption, Mask
    split_and_link("Black", ["Image", "Caption", "Mask"])
    
    # White: Image, Caption (无 Mask)
    split_and_link("White", ["Image", "Caption"])
    
    # 验证
    verify_split()
    
    # 统计 submit_example.csv
    print(f"\n{'='*50}")
    print("关于 submit_example.csv:")
    print("  该 CSV 中的 500 张图片全部对应 test 目录，与训练集无关，无需分割。")
    
    print(f"\n{'='*50}")
    print("分割完成!")
    print(f"  train_split 目录: {NEW_TRAIN_DIR}")
    print(f"  val 目录:         {VAL_DIR}")


if __name__ == "__main__":
    main()
