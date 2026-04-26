# 常用 Git 命令

## 初始化与配置

```bash
git init                          # 初始化新仓库
git clone <url>                   # 克隆远程仓库
git config --global user.name "名字"
git config --global user.email "邮箱"
```

## 查看状态

```bash
git status                        # 查看工作区状态
git log                           # 查看提交历史
git log --oneline                 # 简洁方式查看提交历史
git diff                          # 查看未暂存的修改
git diff --staged                 # 查看已暂存的修改
```

## 暂存与提交

```bash
git add <文件>                    # 暂存指定文件
git add .                         # 暂存所有修改
git commit -m "提交信息"          # 提交暂存内容
git commit -am "提交信息"         # 暂存并提交所有已跟踪文件的修改
```

## 分支操作

```bash
git branch                        # 列出所有本地分支
git branch <分支名>               # 创建新分支
git checkout <分支名>             # 切换分支
git checkout -b <分支名>          # 创建并切换到新分支
git merge <分支名>                # 合并指定分支到当前分支
git branch -d <分支名>            # 删除分支
```

## 远程操作

```bash
git remote -v                     # 查看远程仓库
git remote add origin <url>       # 添加远程仓库
git fetch                         # 获取远程更新（不合并）
git pull                          # 拉取并合并远程更新
git push origin <分支名>          # 推送到远程分支
git push -u origin <分支名>       # 推送并设置上游分支
```

## 撤销操作

```bash
git restore <文件>                # 撤销工作区的修改
git restore --staged <文件>       # 取消暂存
git reset --soft HEAD~1           # 撤销上一次提交，保留修改到暂存区
git reset --hard HEAD~1           # 撤销上一次提交，丢弃所有修改
git revert <commit>               # 创建一个新提交来撤销指定提交
```

## 标签

```bash
git tag                           # 列出所有标签
git tag <标签名>                  # 创建轻量标签
git tag -a <标签名> -m "说明"     # 创建附注标签
git push origin <标签名>          # 推送标签到远程
```

## 储藏

```bash
git stash                         # 储藏当前工作区修改
git stash list                    # 列出所有储藏
git stash pop                     # 恢复最近一次储藏并删除它
git stash drop                    # 删除最近一次储藏
```
