# APTM 外部源码位置

本目录不提交 APTM 源码或模型权重。正式运行前，将官方仓库
`https://github.com/Shuyu-XJTU/APTM` 的源码放到这里，推荐固定到：

- 仓库：`Shuyu-XJTU/APTM`
- 已核验 commit：`b20025445471eb4b743164a6faec29a9eccfb293`
- 本项目使用的 54 个 MALS prompt 来源：`trains.py` 中的 `attr` 列表

可以在网络条件较好的机器 clone 后通过 XFTP 整目录传输，也可以直接 clone：

```bash
git clone https://github.com/Shuyu-XJTU/APTM.git third_party/APTM
git -C third_party/APTM checkout b20025445471eb4b743164a6faec29a9eccfb293
```

权重不要放在本目录，按 `configs/attributes.yaml` 放到 `checkpoints/`。

