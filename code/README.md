# code/ — vendored 第三方依赖

本目录用于放置 TFI 训练 / 推理需要 clone 的第三方仓库（未纳入主仓库 Git 历史）。

## verl（FIPO 训练框架）

```bash
git clone https://github.com/volcengine/verl.git code/verl
cd code/verl
git checkout f9ec463   # 项目最后一次验证的 commit
pip install -e .
```

切换 commit 后将 `train/fipo/verl_patches/` 下的补丁文件按需复制 / 软链到 `code/verl/` 对应位置（详见 [train/fipo/README.md](../train/fipo/README.md)）。
