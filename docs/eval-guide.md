# Evaluation Guide

> **The `minimal_harness.eval` module has been removed in 0.7.0.**
> Use the
> [`mh-gateway` eval HTTP API](https://github.com/J0ey1iu/mh-gateway)
> for batch agent evaluation campaigns. That package owns the
> canonical implementation (`run_batch_eval` +
> `POST /api/v1/eval/batch`).
>
> This document is kept as historical reference. Code samples no
> longer work.

评测模块（`minimal_harness.eval`）曾用于对单 Agent 进行批量效果评测 �?并发执行、全链路事件采集、实时落盘和可视�?HTML 报告�?

�?0.7.0 起，该模块已�?SDK 中移除。评测现在属于服�?网关层，关注点包括：

- 持久化（评测结果要落库、可查询�?
- 权限（M2M 鉴权、用户隔离）
- HTTP API 暴露（`POST /api/v1/eval/batch`�?

请迁移到 `mh-gateway.eval`（含 `run_batch_eval`、`EvalResultStorage`、`/api/v1/eval/batch` HTTP 路由）。该实现是当前唯一推荐的评测入口�?

如果需要在进程内运行小批量评测（无服务、无 HTTP），请直接构�?`SimpleAgent` + `Middleware` 自管事件采集；SDK 仍提供完整的事件流与中间件能力�?
