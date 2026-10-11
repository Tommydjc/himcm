# MM-Agent 旁路桥接（许可门闩）

上游：<https://github.com/usail-hkust/LLM-MM-Agent>（NeurIPS 2025 MM-Agent）。  
本目录**默认不包含**上游源码。启用任何代码或 Skill 前，必须完成许可门闩。

## 许可门闩

1. 阅读上游根目录 `LICENSE` 与 README 中的许可说明（可能与 GitHub 徽标不一致，以文件为准）。
2. 判断本课程/竞赛场景：仅本地使用、是否再分发、是否商业授课。
3. 编辑同目录 [`LICENSE_GATE.md`](LICENSE_GATE.md)，将 `status` 改为下列之一：
   - `BLOCKED`（默认）：禁止调用 MM 代码与 Skill。
   - `SKILLS_ONLY`：仅允许官方文档所述 Cursor/Codex Skill 安装，不 vendoring 源码。
   - `ALLOWED`：允许仓外克隆并按桥接约定调用（仍禁止写入权威 CSV）。

未改门闩文件前，Wiki 中 MM playbook 保持 `draft`，赛时只用 Contest Hub。

## 放行后约定

| 输入 | 输出 |
| --- | --- |
| 新题文本、`data/raw/` | `YYYY_P_slug/paper/notes/` 分析草稿 |
| （可选）Wiki 路由摘要 | 方法候选列表（须再经 A–F 过滤） |

禁止：

- 自动写入 `results/*.csv`
- 把旧年 CSV 喂给 Agent
- submodule 整仓进 GoHiMCM 根目录

## 与 HMML

映射见 [`wiki/comparisons/hmml-vs-gohimcm.md`](../../wiki/comparisons/hmml-vs-gohimcm.md)。

## 相关 Wiki

- [mm-agent-assisted-loop](../../wiki/playbooks/mm-agent-assisted-loop.md)
- [agent-stack-roles](../../wiki/comparisons/agent-stack-roles.md)
