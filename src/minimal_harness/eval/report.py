from __future__ import annotations

import json
from typing import Any

from .types import EvalSummary


def generate_html_report(summary: EvalSummary, output_path: str) -> None:
    data = _summary_to_report_data(summary)
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    data_json = data_json.replace("</", "<\\/")
    html = _HTML_TEMPLATE.replace("DATA_GOES_HERE", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _summary_to_report_data(summary: EvalSummary) -> dict[str, Any]:
    return {
        "taskName": summary.task_name,
        "description": summary.description,
        "agentId": summary.agent_metadata_id,
        "total": summary.total_runs,
        "completed": summary.completed,
        "failed": summary.failed,
        "interrupted": summary.interrupted,
        "totalTime": round(summary.total_time, 2),
        "avgTime": round(summary.avg_time, 2),
        "totalTokens": summary.total_tokens,
        "totalCost": round(summary.total_cost, 6)
        if summary.total_cost is not None
        else None,
        "runs": [_run_to_report_data(r) for r in summary.runs],
    }


def _run_to_report_data(r: Any) -> dict[str, Any]:
    tu = r.token_usage
    return {
        "runId": r.run_id,
        "input": r.input_text,
        "status": r.status,
        "timeTaken": round(r.time_taken, 2) if r.time_taken is not None else None,
        "error": r.error,
        "response": r.response,
        "llmCalls": r.llm_call_count,
        "toolCalls": r.tool_call_count,
        "exceeded": r.exceeded,
        "inputTokens": tu.input_tokens if tu else 0,
        "outputTokens": tu.output_tokens if tu else 0,
        "totalTokens": tu.total_tokens if tu else 0,
        "totalCost": round(tu.total_cost, 6)
        if tu and tu.total_cost is not None
        else None,
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval Report</title>
<style>
* { margin:0; padding:0; box-sizing:border-box }
body { background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif; padding:24px; line-height:1.6 }
h1 { font-size:1.8rem; font-weight:700; margin-bottom:4px }
h2 { font-size:1.2rem; font-weight:600; margin-bottom:12px; color:#94a3b8 }
.header { margin-bottom:24px }
.header .sub { color:#64748b; font-size:0.9rem }
.dashboard { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:28px }
.card { background:#1e293b; border-radius:10px; padding:16px; text-align:center }
.card .value { font-size:1.6rem; font-weight:700 }
.card .label { font-size:0.8rem; color:#94a3b8; margin-top:4px }
.card.green .value { color:#4ade80 }
.card.red .value { color:#f87171 }
.card.yellow .value { color:#facc15 }
.card.blue .value { color:#60a5fa }
.card.purple .value { color:#c084fc }
.toolbar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap }
.toolbar input, .toolbar select { background:#1e293b; border:1px solid #334155; color:#e2e8f0; padding:8px 12px; border-radius:6px; font-size:0.9rem }
.toolbar input { flex:1; min-width:200px }
.toolbar select { cursor:pointer }
table { width:100%; border-collapse:collapse; font-size:0.9rem }
th { text-align:left; padding:10px 12px; border-bottom:2px solid #334155; color:#94a3b8; font-weight:600; cursor:pointer; user-select:none; white-space:nowrap }
th:hover { color:#e2e8f0 }
td { padding:10px 12px; border-bottom:1px solid #1e293b; vertical-align:top }
tr.run-row { cursor:pointer; transition:background .15s }
tr.run-row:hover { background:#1e293b }
.status-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600 }
.status-completed { background:#064e3b; color:#4ade80 }
.status-failed { background:#7f1d1d; color:#f87171 }
.status-interrupted { background:#713f12; color:#facc15 }
.token-bar { display:flex; height:16px; border-radius:4px; overflow:hidden; min-width:60px }
.token-bar .input { background:#3b82f6; height:100% }
.token-bar .output { background:#a78bfa; height:100% }
.detail-panel { display:none; background:#1e293b; border-radius:8px; padding:16px; margin:4px 0 12px }
.detail-panel.open { display:block }
.detail-panel h3 { font-size:0.95rem; color:#94a3b8; margin-bottom:8px }
.detail-section { margin-bottom:12px }
.detail-section:last-child { margin-bottom:0 }
.detail-section .label { font-size:0.8rem; color:#64748b; margin-bottom:4px }
.detail-section .response { background:#0f172a; padding:12px; border-radius:6px; white-space:pre-wrap; word-break:break-word; max-height:300px; overflow-y:auto; font-size:0.85rem; line-height:1.5 }
.events { list-style:none }
.event-item { padding:6px 10px; margin:2px 0; border-radius:4px; font-size:0.8rem; display:flex; justify-content:space-between; align-items:center }
.event-item .etime { color:#64748b; font-family:monospace; font-size:0.75rem }
.event-item.llm_start { background:#1e3a5f }
.event-item.llm_end { background:#1e3a5f; border-left:3px solid #60a5fa }
.event-item.tool_start { background:#2e1065 }
.event-item.tool_end { background:#2e1065; border-left:3px solid #a78bfa }
.event-item.agent_start { background:#064e3b }
.event-item.agent_end { background:#064e3b; border-left:3px solid #4ade80 }
.event-item.error { background:#7f1d1d; border-left:3px solid #f87171 }
.event-item.tool_error { background:#7f1d1d; border-left:3px solid #f87171 }
.token-detail { display:flex; gap:16px; font-size:0.85rem }
.token-detail span { background:#0f172a; padding:4px 10px; border-radius:4px }
.text-mono { font-family:monospace; font-size:0.8rem; color:#64748b }
.cost { color:#fbbf24 }
.footer { text-align:center; padding:24px 0; color:#475569; font-size:0.8rem }
@media(max-width:640px) { .dashboard { grid-template-columns:repeat(2,1fr) } td:nth-child(3),th:nth-child(3) { display:none } }
</style>
</head>
<body>
<div class="header">
<h1>Eval: <span id="title"></span></h1>
<div class="sub" id="desc"></div>
<div class="sub">Agent: <span id="agent-id"></span> | <span id="total-runs"></span> runs</div>
</div>
<div id="app"></div>
<div class="footer">Generated by minimal-harness eval</div>
<script id="eval-data" type="application/json">
DATA_GOES_HERE
</script>
<script>
const D = JSON.parse(document.getElementById('eval-data').textContent);
function fmt(s) { return s!=null?s.toFixed(2):'-' }
function badge(s) { return '<span class="status-badge status-'+s+'">'+({completed:'completed',failed:'failed',interrupted:'interrupted'}[s]||s)+'</span>' }
function render() {
  let h='<div class="dashboard">'+
    '<div class="card green"><div class="value">'+D.completed+'</div><div class="label">Completed</div></div>'+
    '<div class="card red"><div class="value">'+D.failed+'</div><div class="label">Failed</div></div>'+
    '<div class="card yellow"><div class="value">'+D.interrupted+'</div><div class="label">Interrupted</div></div>'+
    '<div class="card blue"><div class="value">'+fmt(D.avgTime)+'s</div><div class="label">Avg Time</div></div>'+
    '<div class="card purple"><div class="value">'+D.totalTokens.toLocaleString()+'</div><div class="label">Total Tokens</div></div>'+
    (D.totalCost!=null?'<div class="card yellow"><div class="value">$'+D.totalCost.toFixed(4)+'</div><div class="label">Total Cost</div></div>':'')+
  '</div>';
  h+='<div class="toolbar">'+
    '<input type="text" id="search" placeholder="Search inputs..." oninput="render()">'+
    '<select id="filter" onchange="render()">'+
      '<option value="all">All</option>'+
      '<option value="completed">Completed</option>'+
      '<option value="failed">Failed</option>'+
      '<option value="interrupted">Interrupted</option>'+
    '</select>'+
  '</div>';
  var q=(document.getElementById('search')||{value:''}).value.toLowerCase();
  var f=(document.getElementById('filter')||{value:'all'}).value;
  var runs=D.runs.filter(function(r){return (f==='all'||r.status===f)&&r.input.toLowerCase().includes(q)});
  h+='<table><thead><tr>'+
    '<th onclick="sortBy(\\'input\\')">Input</th>'+
    '<th onclick="sortBy(\\'status\\')">Status</th>'+
    '<th onclick="sortBy(\\'timeTaken\\')">Time</th>'+
    '<th onclick="sortBy(\\'llmCalls\\')">LLM</th>'+
    '<th onclick="sortBy(\\'toolCalls\\')">Tools</th>'+
    '<th>Tokens</th>'+
    (D.totalCost!=null?'<th onclick="sortBy(\\'totalCost\\')">Cost</th>':'')+
  '</tr></thead><tbody>';
  for(var i=0;i<runs.length;i++) {
    var r=runs[i];
    var maxT=D.runs.reduce(function(a,b){return Math.max(a,b.totalTokens)},0);
    var inpPct=r.totalTokens>0?Math.round(r.inputTokens/r.totalTokens*100):0;
    var outPct=r.totalTokens>0?Math.round(r.outputTokens/r.totalTokens*100):0;
    var tokW=maxT>0?Math.round(r.totalTokens/maxT*60):0;
    h+='<tr class="run-row" data-run-id="'+esc(r.runId)+'" onclick="toggle(this.dataset.runId)">'+
      '<td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(r.input)+'</td>'+
      '<td>'+badge(r.status)+(r.exceeded?' <span style="color:#facc15;font-size:0.75rem">max-iter</span>':'')+'</td>'+
      '<td>'+(r.timeTaken!=null?r.timeTaken+'s':'-')+'</td>'+
      '<td>'+r.llmCalls+'</td>'+
      '<td>'+r.toolCalls+'</td>'+
      '<td><div class="token-bar" style="width:'+(tokW||20)+'px"><div class="input" style="width:'+inpPct+'%"></div><div class="output" style="width:'+outPct+'%"></div></div><span style="font-size:0.75rem;margin-left:6px;color:#94a3b8">'+r.totalTokens+'</span></td>'+
      (D.totalCost!=null?'<td class="cost">'+(r.totalCost!=null?'$'+r.totalCost.toFixed(6):'-')+'</td>':'')+
    '</tr>';
    h+='<tr id="detail-'+r.runId+'" class="detail-panel"><td colspan="'+(D.totalCost!=null?7:6)+'">'+
      '<div class="detail-section"><div class="label">Response</div><div class="response">'+esc(r.response||'(no response)')+'</div></div>'+
      (r.error?'<div class="detail-section"><div class="label">Error</div><div style="color:#f87171">'+esc(r.error)+'</div></div>':'')+
      '<div class="detail-section" style="margin-top:12px"><div class="label">Token Usage</div><div class="token-detail"><span>Input: '+r.inputTokens+'</span><span>Output: '+r.outputTokens+'</span><span>Total: '+r.totalTokens+'</span></div></div>'+
    '</td></tr>';
  }
  h+='</tbody></table>';
  if(runs.length===0)h+='<p style="text-align:center;padding:40px;color:#64748b">No runs match the filter.</p>';
  document.getElementById('app').innerHTML=h;
}
var sortDir={};
function sortBy(k) {
  var d=sortDir[k]=(sortDir[k]==='asc'?'desc':'asc');
  D.runs.sort(function(a,b) {
    var va=a[k]||'',vb=b[k]||'';
    if(typeof va==='string')return d==='asc'?va.localeCompare(vb):vb.localeCompare(va);
    return d==='asc'?va-vb:vb-va;
  });
  render();
}
function toggle(id) {
  var el=document.getElementById('detail-'+id);
  if(el)el.classList.toggle('open');
}
function esc(s) { if(!s)return'';return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') }
document.getElementById('title').textContent=D.taskName;
document.getElementById('desc').textContent=D.description;
document.getElementById('agent-id').textContent=D.agentId;
document.getElementById('total-runs').textContent=D.total;
render();
</script>
</body>
</html>"""
