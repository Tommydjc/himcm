# Remit 旁路桥接

[Remit](https://github.com/zhou2030109-glitch/Remit) 在**仓库外**运行。本目录只约定导入/导出与人审门闩，不包含 Remit 源码。

## 安装（仓外）

```text
git clone https://github.com/zhou2030109-glitch/Remit.git
cd Remit
docker compose -f docker-compose.release.yml up -d --build
```

或按上游 README 使用 `win_start.bat` / mac 脚本。打开上游文档中的本地端口，配置自有模型 API Key（勿写入本仓）。

## 目录契约

| Remit 侧 | 人审后写入本仓 |
| --- | --- |
| 导入题面/数据 | `YYYY_P_slug/data/raw/`（只增，不覆盖已冻结 raw） |
| 代码提案 | `YYYY_P_slug/src/`（指定文件，禁止盲覆盖） |
| 数值表 | `YYYY_P_slug/results/`（**仅**确认后） |
| LaTeX / 章节 | `YYYY_P_slug/paper/` |
| 计划与对话摘要 | `YYYY_P_slug/paper/notes/` + `prompts/cursor_log.md` |

## 项目说明粘贴块

```text
权威数字只认本项目 results/*.csv。
先读 GoHiMCM wiki/playbooks/contest-hub.md 与 problem-type-router.md。
禁止引用其他年份 CSV 数字。
基线方法先于深度模型。
关键 AI 步骤记 prompts/cursor_log.md。
未审核不得写入 results/。
```

## Wiki 只读技能包（建议）

复制或挂载（只读）：

- `wiki/playbooks/contest-hub.md`
- `wiki/playbooks/problem-type-router.md`
- `wiki/concepts/cross-year-failure-modes.md`
- `wiki/playbooks/4h-combat-playbook.md`
- `wiki/playbooks/spec-driven-cursor-coding.md`

**不要**挂载各年 `results/*.csv` 进 Remit 知识库。

## 相关 Wiki

- [remit-assisted-loop](../../wiki/playbooks/remit-assisted-loop.md)
- [agent-stack-roles](../../wiki/comparisons/agent-stack-roles.md)
