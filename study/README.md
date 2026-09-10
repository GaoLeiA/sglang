# SGLang 学习笔记

这个目录用于保存个人源码阅读、算子整理和架构学习笔记。

## 文档索引

| 文档                                                            | 内容                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------ |
| [架构学习指南](sglang_architecture.md)                          | 请求生命周期、调度、模型执行、缓存和分布式架构               |
| [Triton 算子指南](sglang_triton_operators.md)                   | 调用链、Kernel 启动、目录分布和函数索引                      |
| [Triton 融合算子清单 (vLLM / Omni / SGLang)](Triton%20fused%20ops.md) | 跨框架 Triton 融合算子（FLA/GDN、Attention、MoE、LoRA 等）全量对照 |
| [Qwen CUDA／Triton 算子清单](qwen3_6_sglang_custom_tritonop.md) | 对照 vLLM 参考清单，整理 SGLang 相关算子、条件分支和调用位置 |

## 仓库维护方式

学习仓库的远程地址：

- `origin`：<https://github.com/GaoLeiA/sglang>
- `upstream`：<https://github.com/sgl-project/sglang>

建议让 `main` 跟随官方源码，把个人笔记保存在 `study` 分支。这样可以在 GitHub 上同步 fork 的 `main`，再将同步后的代码合入学习分支。

本目录复制到新仓库后，需要先提交笔记，首次推送时发布 `study` 分支：

```powershell
git switch study
git add study
git commit -m "docs: add SGLang study notes"
git push -u origin study
```

以上是维护命令说明，不表示这些提交和推送已经执行。

## 定期同步

如果你已通过 GitHub 将官方代码同步到 fork 的 `main`，在本地执行：

```powershell
git fetch origin
git switch study
git merge origin/main
```

也可以直接从官方仓库获取并合入学习分支：

```powershell
git fetch upstream
git switch study
git merge upstream/main
```

这两种方式按实际情况选择一种。先提交或暂存正在编辑的内容；若发生冲突，保留需要的笔记并按 Git 提示解决，完成后再推送学习分支：

```powershell
git push origin study
```

日常维护学习分支使用 merge 保留个人提交，不需要把学习分支强制重置到官方 `main`。

后续日常维护与同步命令备忘
当您在 GitHub 上将官方 sgl-project/sglang 代码同步到您的 fork GaoLeiA/sglang 后，在本地终端执行以下命令即可同步至 study 分支：

```powershell
# 1. 获取 origin 最新代码并合并到 study 分支
git fetch origin
git switch study
git merge origin/main

# 2. 推送更新后的 study 分支
git push origin study
```


## 源码版本说明

现有三份专题文档基于提交 `03d06a764e4a83268eefd1bafc676418f7269c89` 整理。新克隆的 fork 可能位于不同提交，后续同步也会改变源码布局。

- 文件相对链接可直接导航当前 checkout，但文中的行号和调用链属于各文档注明的源码快照。
- 同步后如发现文件移动、后端分派或接口变化，应更新对应笔记并记录新的基线提交。
- 不要仅凭旧清单中的算子名称，认定新版本或其他模型会执行同一路径。

新增笔记时，建议记录研究问题、模型或后端条件、源码 commit、关键文件与调用链，以及验证方式。
