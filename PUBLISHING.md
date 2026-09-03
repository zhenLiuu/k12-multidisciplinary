# GitHub 与 Hugging Face 发布步骤

本地逻辑数据已经冻结。发布时 GitHub 与 Hugging Face 使用两个同名或易于对应的仓库，例如 `k12-multidisciplinary-multimodal-reasoning`。

## 1. 本地构建与验证

```bash
conda run -n bmmr python scripts/build_release.py
conda run -n bmmr python scripts/verify_release_artifacts.py
conda run -n bmmr python scripts/prepare_github_release.py
```

Hugging Face 上传目录为 `dist/v0.1.0/huggingface/`，GitHub 推送目录为 `dist/v0.1.0/github/`。不要直接把工作目录作为 GitHub 仓库，以免发布内部中间文件或大文件。

## 2. GitHub

1. 在 GitHub 网页新建一个空仓库，不要勾选自动创建 README、LICENSE 或 `.gitignore`。
2. 先完成第 3 节的 HF 仓库创建与元数据定稿，再推送 GitHub 包。
3. 在精简发布目录初始化并推送：

```bash
cd dist/v0.1.0/github
git init
git add .
git commit -m "Release v0.1.0"
git branch -M main
git remote add origin git@github.com:zhenLiuu/k12-multidisciplinary.git
git push -u origin main
```

4. 数据上传并验证后，在 GitHub 创建 `v0.1.0` Release；release notes 使用 `CHANGELOG.md` 的对应内容。

## 3. Hugging Face

1. 注册或登录 Hugging Face，在头像菜单中选择 **New Dataset**。
2. 创建一个 dataset repository；建议先设为 private，验证完成后再切换 public。
3. 使用定稿脚本写入两个实际仓库标识并刷新 manifest；不要手工修改已构建的 HF README：

```bash
conda run -n bmmr python scripts/finalize_release_metadata.py --hf-repo zhenliuu/k12-multidisciplinary --github-repo zhenLiuu/k12-multidisciplinary
conda run -n bmmr python scripts/verify_release_artifacts.py
```
4. 登录并上传：

```bash
conda run -n bmmr hf auth login
HF_XET_HIGH_PERFORMANCE=1 conda run -n bmmr hf upload \
  zhenliuu/k12-multidisciplinary \
  dist/v0.1.0/huggingface \
  --repo-type dataset
```

同一个上传目录会保存可恢复的上传缓存；上传中断后重新执行同一命令。不要并行启动多个上传进程。

## 4. 公开前检查

1. 从 Hugging Face 下载到一个全新目录并核对 `SHA256SUMS`。
2. 确认 Dataset Viewer 能展示 `raw` 和 `test`。
3. 使用 `load_dataset("zhenliuu/k12-multidisciplinary")` 实测加载两个 split。
4. 随机选择含图记录，通过 `images/index.parquet` 找到 tar 并成功解码图片。
5. 把 Hugging Face 数据集链接补到 GitHub README，再推送最终文档提交。
6. 将 HF 仓库切换为 public，然后创建 GitHub `v0.1.0` Release。
