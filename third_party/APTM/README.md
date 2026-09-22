# APTM 外部源码位置

本目录不提交 APTM 源码或模型权重。正式运行前，将官方仓库
`https://github.com/Shuyu-XJTU/APTM` 的源码放到这里，推荐固定到：

- 仓库：`Shuyu-XJTU/APTM`
- 已核验 commit：`b20025445471eb4b743164a6faec29a9eccfb293`
- 本项目使用的 54 个 MALS prompt 来源：`trains.py` 中的 `attr` 列表

本目录已经包含由本项目管理的这个 `README.md`，因此不能执行
`git clone ... third_party/APTM`。请先 clone 到临时目录，再复制官方源码；不要复制
官方仓库的 `.git` 和 `README.md`。

Windows PowerShell：

```powershell
$aptmSource = Join-Path $env:TEMP ("APTM-" + [guid]::NewGuid())
git clone https://github.com/Shuyu-XJTU/APTM.git $aptmSource
git -C $aptmSource checkout b20025445471eb4b743164a6faec29a9eccfb293
Get-ChildItem -Force -LiteralPath $aptmSource |
  Where-Object { $_.Name -notin @('.git', 'README.md') } |
  Copy-Item -Destination .\third_party\APTM -Recurse -Force
Remove-Item -LiteralPath $aptmSource -Recurse -Force
```

Linux：

```bash
aptm_source_dir="$(mktemp -d)"
git clone https://github.com/Shuyu-XJTU/APTM.git "$aptm_source_dir/APTM"
git -C "$aptm_source_dir/APTM" checkout b20025445471eb4b743164a6faec29a9eccfb293
find "$aptm_source_dir/APTM" -mindepth 1 -maxdepth 1 \
  ! -name .git ! -name README.md \
  -exec cp -a {} third_party/APTM/ \;
rm -rf -- "$aptm_source_dir"
```

服务器不需要联网：可以在 Windows 完成临时 clone，把得到的官方源码目录通过
XFTP 合并到服务器的 `third_party/APTM/`。权重不要放在本目录，按
`configs/attributes.yaml` 放到 `checkpoints/`。

准备完成后运行 `python scripts/preflight_aptm.py`。该命令会依据
`metadata/aptm_source.json` 校验固定 commit 的关键源码文件，能同时兼容 Windows
和 Linux 的换行格式。
