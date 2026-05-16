from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .persistence import load_run_events
from .types import EvalSummary


def generate_html_report(summary: EvalSummary, output_path: str) -> None:
    data = _summary_to_report_data(summary)
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    data_json = data_json.replace("</", "<\\/")
    html = _MAIN_TEMPLATE.replace("DATA_GOES_HERE", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    details_dir = Path(summary.output_path) / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for run in summary.runs:
        events = load_run_events(summary.output_path, run.run_id)
        run_data = _run_to_report_data(run)
        run_data["events"] = events
        run_data_json = json.dumps(run_data, ensure_ascii=False, default=str).replace(
            "</", "<\\/"
        )
        detail_html = _DETAIL_TEMPLATE.replace("RUN_DATA_GOES_HERE", run_data_json)
        detail_path = details_dir / f"{run.run_id}.html"
        with open(detail_path, "w", encoding="utf-8") as f:
            f.write(detail_html)


def _summary_to_report_data(summary: EvalSummary) -> dict[str, Any]:
    runs_data = [_run_to_report_data(r) for r in summary.runs]
    times = [r.time_taken for r in summary.runs if r.time_taken is not None]
    tokens_list = [
        r.token_usage.total_tokens
        for r in summary.runs
        if r.token_usage and r.token_usage.total_tokens
    ]
    return {
        "taskName": summary.task_name,
        "description": summary.description,
        "agentId": summary.agent_metadata_id,
        "total": summary.total_runs,
        "completed": summary.completed,
        "failed": summary.failed,
        "interrupted": summary.interrupted,
        "passRate": round(summary.completed / summary.total_runs * 100, 1)
        if summary.total_runs
        else 0,
        "totalTime": round(summary.total_time, 2),
        "avgTime": round(summary.avg_time, 2),
        "minTime": round(min(times), 2) if times else 0,
        "maxTime": round(max(times), 2) if times else 0,
        "totalTokens": summary.total_tokens,
        "avgTokens": round(sum(tokens_list) / len(tokens_list)) if tokens_list else 0,
        "totalCost": round(summary.total_cost, 6)
        if summary.total_cost is not None
        else None,
        "runs": runs_data,
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


_MAIN_TEMPLATE = """<!DOCTYPE html>
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
.grid-2 { display:grid; grid-template-columns:auto 1fr; gap:20px; margin-bottom:24px }
.ring-wrap { display:flex; flex-direction:column; align-items:center; gap:8px }
.pass-ring { width:110px; height:110px; border-radius:50%; position:relative; display:flex; align-items:center; justify-content:center; flex-shrink:0 }
.pass-ring .inner { position:relative;z-index:1;width:82px;height:82px;background:#0f172a;border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center }
.pass-ring .inner .pct { font-size:1.5rem;font-weight:800;line-height:1 }
.pass-ring .inner .pct-label { font-size:0.65rem;color:#94a3b8;margin-top:2px }
.status-legend { display:flex; gap:16px; flex-wrap:wrap }
.status-legend-item { display:flex; align-items:center; gap:6px; font-size:0.85rem }
.status-legend-dot { width:10px;height:10px;border-radius:50% }
.dashboard { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:24px }
.card { background:#1e293b; border-radius:10px; padding:14px 16px; text-align:center }
.card .value { font-size:1.3rem; font-weight:700 }
.card .label { font-size:0.75rem; color:#94a3b8; margin-top:2px }
.card.green .value { color:#4ade80 }
.card.red .value { color:#f87171 }
.card.yellow .value { color:#facc15 }
.card.blue .value { color:#60a5fa }
.card.purple .value { color:#c084fc }
.card.orange .value { color:#fb923c }
.toolbar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap }
.toolbar input, .toolbar select { background:#1e293b; border:1px solid #334155; color:#e2e8f0; padding:8px 12px; border-radius:6px; font-size:0.9rem }
.toolbar input { flex:1; min-width:200px }
.toolbar select { cursor:pointer }
.table-wrap { overflow-x:auto }
table { width:100%; border-collapse:collapse; font-size:0.85rem; min-width:700px }
th { text-align:left; padding:10px 10px; border-bottom:2px solid #334155; color:#94a3b8; font-weight:600; cursor:pointer; user-select:none; white-space:nowrap; position:relative }
th:hover { color:#e2e8f0 }
th .sort-arrow { font-size:0.65rem; margin-left:3px; color:#64748b }
td { padding:10px 10px; border-bottom:1px solid #1e293b; vertical-align:middle }
tr.run-row { cursor:pointer; transition:background .15s }
tr.run-row:hover { background:#1e293b }
.input-cell { max-width:280px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap }
.status-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; white-space:nowrap }
.status-completed { background:#064e3b; color:#4ade80 }
.status-failed { background:#7f1d1d; color:#f87171 }
.status-interrupted { background:#713f12; color:#facc15 }
.token-bar { display:flex; height:14px; border-radius:3px; overflow:hidden; min-width:50px; background:#1e293b }
.token-bar .input { background:#3b82f6; height:100%; transition:width .3s }
.token-bar .output { background:#a78bfa; height:100%; transition:width .3s }
.token-cell { display:flex; align-items:center; gap:8px }
.detail-link { display:inline-flex;align-items:center;gap:4px;padding:4px 8px;background:#334155;color:#cbd5e1;border-radius:4px;font-size:0.7rem;text-decoration:none;transition:all .15s;white-space:nowrap }
.detail-link:hover { background:#475569; color:#e2e8f0 }
.detail-panel { display:none; background:#1e293b }
.detail-panel.open { display:table-row }
.detail-panel > td { padding:16px 20px }
.detail-panel-inner { max-width:800px }
.detail-section { margin-bottom:12px }
.detail-section:last-child { margin-bottom:0 }
.detail-section .dlabel { font-size:0.75rem; color:#64748b; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px }
.detail-section .dval { background:#0f172a; padding:10px 12px; border-radius:6px; white-space:pre-wrap; word-break:break-word; max-height:200px; overflow-y:auto; font-size:0.8rem; line-height:1.5 }
.detail-section .dval.error { color:#f87171 }
.token-detail { display:flex; gap:16px; font-size:0.82rem }
.token-detail span { background:#0f172a; padding:4px 10px; border-radius:4px; white-space:nowrap }
.cost { color:#fbbf24 }
.footer { text-align:center; padding:24px 0; color:#475569; font-size:0.8rem }
.status-summary-bar { display:flex; height:6px; border-radius:3px; overflow:hidden; margin-top:8px; width:100% }
.status-summary-bar .seg-completed { background:#4ade80 }
.status-summary-bar .seg-failed { background:#f87171 }
.status-summary-bar .seg-interrupted { background:#facc15 }
.count-badge { display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;border-radius:10px;font-size:0.7rem;font-weight:700;padding:0 6px }
.count-badge.green { background:#064e3b; color:#4ade80 }
.count-badge.red { background:#7f1d1d; color:#f87171 }
.count-badge.yellow { background:#713f12; color:#facc15 }
@media(max-width:640px) { .grid-2 { grid-template-columns:1fr } body { padding:12px } .dashboard { grid-template-columns:repeat(2,1fr) } }
</style>
</head>
<body>
<div class="header">
<h1>Eval: <span id="title"></span></h1>
<div class="sub" id="desc"></div>
<div class="sub">Agent: <span id="agent-id"></span> &middot; <span id="total-runs"></span> runs &middot; generated <span id="gen-time"></span></div>
</div>
<div id="app"></div>
<div class="footer">Generated by minimal-harness eval</div>
<script id="eval-data" type="application/json">
DATA_GOES_HERE
</script>
<script>
const D = JSON.parse(document.getElementById('eval-data').textContent);
function fmt(s) { return s!=null?s.toFixed(2):'-' }
function badge(s,e) {
  var b='<span class="status-badge status-'+s+'">'+({completed:'completed',failed:'failed',interrupted:'interrupted'}[s]||s)+'</span>';
  if(e)b+=' <span style="color:#facc15;font-size:0.7rem">max-iter</span>';
  return b;
}
function render() {
  var passDeg=D.completed/D.total*360;
  var h='<div class="grid-2">'+
    '<div class="ring-wrap">'+
      '<div class="pass-ring" style="background:conic-gradient(#4ade80 0deg '+passDeg+'deg, #f87171 '+passDeg+'deg '+(passDeg+(D.failed/D.total*360))+'deg, #facc15 '+(passDeg+(D.failed/D.total*360))+'deg 360deg)">'+
        '<div class="inner"><div class="pct">'+D.passRate+'%</div><div class="pct-label">pass</div></div>'+
      '</div>'+
      '<div class="status-legend">'+
        '<div class="status-legend-item"><span class="status-legend-dot" style="background:#4ade80"></span>'+D.completed+' completed</div>'+
        '<div class="status-legend-item"><span class="status-legend-dot" style="background:#f87171"></span>'+D.failed+' failed</div>'+
        '<div class="status-legend-item"><span class="status-legend-dot" style="background:#facc15"></span>'+D.interrupted+' interrupted</div>'+
      '</div>'+
    '</div>'+
    '<div class="dashboard">'+
      '<div class="card blue"><div class="value">'+fmt(D.avgTime)+'s</div><div class="label">Avg Time</div></div>'+
      '<div class="card orange"><div class="value">'+fmt(D.minTime)+'s</div><div class="label">Min Time</div></div>'+
      '<div class="card yellow"><div class="value">'+fmt(D.maxTime)+'s</div><div class="label">Max Time</div></div>'+
      '<div class="card purple"><div class="value">'+D.totalTokens.toLocaleString()+'</div><div class="label">Total Tokens</div></div>'+
      '<div class="card blue"><div class="value">'+D.avgTokens.toLocaleString()+'</div><div class="label">Avg Tokens</div></div>'+
      '<div class="card green"><div class="value">'+fmt(D.totalTime)+'s</div><div class="label">Wall Time</div></div>'+
      (D.totalCost!=null?'<div class="card yellow"><div class="value">$'+D.totalCost.toFixed(4)+'</div><div class="label">Total Cost</div></div>':'')+
    '</div>'+
  '</div>';
  h+='<div class="toolbar">'+
    '<input type="text" id="search" placeholder="Search inputs..." oninput="render()">'+
    '<select id="filter" onchange="render()">'+
      '<option value="all">All</option>'+
      '<option value="completed">Completed</option>'+
      '<option value="failed">Failed</option>'+
      '<option value="interrupted">Interrupted</option>'+
    '</select>'+
    '<span style="color:#64748b;font-size:0.85rem;display:flex;align-items:center" id="result-count"></span>'+
  '</div>';
  var q=(document.getElementById('search')||{value:''}).value.toLowerCase();
  var f=(document.getElementById('filter')||{value:'all'}).value;
  var runs=D.runs.filter(function(r){return (f==='all'||r.status===f)&&r.input.toLowerCase().includes(q)});
  h+='<div class="table-wrap"><table><thead><tr>'+
    '<th onclick="sortBy(\\'input\\')">Input<span class="sort-arrow"></span></th>'+
    '<th onclick="sortBy(\\'status\\')">Status<span class="sort-arrow"></span></th>'+
    '<th onclick="sortBy(\\'timeTaken\\')">Time<span class="sort-arrow"></span></th>'+
    '<th onclick="sortBy(\\'llmCalls\\')">LLM<span class="sort-arrow"></span></th>'+
    '<th onclick="sortBy(\\'toolCalls\\')">Tools<span class="sort-arrow"></span></th>'+
    '<th>Tokens</th>'+
    (D.totalCost!=null?'<th onclick="sortBy(\\'totalCost\\')">Cost<span class="sort-arrow"></span></th>':'')+
    '<th></th>'+
  '</tr></thead><tbody>';
  for(var i=0;i<runs.length;i++) {
    var r=runs[i];
    var maxT=D.runs.reduce(function(a,b){return Math.max(a,b.totalTokens)},0);
    var inpPct=r.totalTokens>0?Math.round(r.inputTokens/r.totalTokens*100):0;
    var outPct=r.totalTokens>0?Math.round(r.outputTokens/r.totalTokens*100):0;
    var tokW=maxT>0?Math.round(r.totalTokens/maxT*60):0;
    h+='<tr class="run-row" onclick="toggle(\\''+esc(r.runId)+'\\')">'+
      '<td class="input-cell" title="'+esc(r.input)+'">'+esc(r.input)+'</td>'+
      '<td>'+badge(r.status,r.exceeded)+'</td>'+
      '<td>'+(r.timeTaken!=null?r.timeTaken+'s':'-')+'</td>'+
      '<td>'+r.llmCalls+'</td>'+
      '<td>'+r.toolCalls+'</td>'+
      '<td class="token-cell"><div class="token-bar" style="width:'+(tokW||20)+'px"><div class="input" style="width:'+inpPct+'%"></div><div class="output" style="width:'+outPct+'%"></div></div><span style="font-size:0.7rem;color:#94a3b8">'+r.totalTokens+'</span></td>'+
      (D.totalCost!=null?'<td class="cost">'+(r.totalCost!=null?'$'+r.totalCost.toFixed(6):'-')+'</td>':'')+
      '<td><a href="details/'+esc(r.runId)+'.html" class="detail-link" onclick="event.stopPropagation()">Details &rarr;</a></td>'+
    '</tr>';
    h+='<tr id="detail-'+r.runId+'" class="detail-panel"><td colspan="'+(D.totalCost!=null?8:7)+'">'+
      '<div class="detail-panel-inner">'+
      '<div class="detail-section"><div class="dlabel">Response</div><div class="dval">'+esc(r.response||'(no response)')+'</div></div>'+
      (r.error?'<div class="detail-section"><div class="dlabel">Error</div><div class="dval error">'+esc(r.error)+'</div></div>':'')+
      '<div class="detail-section"><div class="dlabel">Token Usage</div><div class="token-detail"><span>Input: '+r.inputTokens+'</span><span>Output: '+r.outputTokens+'</span><span>Total: '+r.totalTokens+'</span></div></div>'+
      '</div>'+
    '</td></tr>';
  }
  h+='</tbody></table></div>';
  if(runs.length===0)h+='<p style="text-align:center;padding:40px;color:#64748b">No runs match the filter.</p>';
  document.getElementById('app').innerHTML=h;
  document.getElementById('result-count').textContent=runs.length+'/'+D.runs.length+' runs';
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
document.getElementById('gen-time').textContent=new Date().toLocaleString();
render();
</script>
</body>
</html>"""


_DETAIL_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run Detail</title>
<style>
* { margin:0; padding:0; box-sizing:border-box }
body { background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,sans-serif; padding:24px; line-height:1.6 }
a { color:#60a5fa; text-decoration:none }
a:hover { text-decoration:underline }
.back-link { display:inline-flex;align-items:center;gap:6px;font-size:0.85rem;color:#94a3b8;margin-bottom:16px;padding:6px 12px;background:#1e293b;border-radius:6px;transition:background .15s }
.back-link:hover { background:#334155;color:#e2e8f0;text-decoration:none }
.run-header { margin-bottom:24px }
.run-header h1 { font-size:1.4rem;font-weight:700;margin-bottom:8px;word-break:break-word }
.meta-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:12px }
.meta-item { background:#1e293b;border-radius:6px;padding:10px 12px }
.meta-item .mlabel { font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px }
.meta-item .mvalue { font-size:1.1rem;font-weight:600 }
.meta-item .mvalue.completed { color:#4ade80 }
.meta-item .mvalue.failed { color:#f87171 }
.meta-item .mvalue.interrupted { color:#facc15 }
.status-bar { display:flex;height:6px;border-radius:3px;overflow:hidden;gap:2px;margin-top:8px }
.status-bar .seg { height:100%;border-radius:2px }
.timeline { position:relative;padding-left:36px;margin-top:24px }
.timeline-line { position:absolute;left:16px;top:0;bottom:0;width:2px;background:#334155 }
.tl-section { margin-bottom:20px }
.tl-section.has-errors .tl-dot { box-shadow:0 0 0 4px rgba(248,113,113,.25) }
.section-label { font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;margin-left:-36px }
.tl-item { position:relative;margin-bottom:12px }
.tl-item:last-child { margin-bottom:0 }
.tl-dot { position:absolute;left:-20px;top:14px;width:14px;height:14px;border-radius:50%;z-index:1;border:2px solid;background:#0f172a }
.tl-dot.agent { border-color:#4ade80 }
.tl-dot.llm { border-color:#60a5fa }
.tl-dot.tool { border-color:#a78bfa }
.tl-dot.error { border-color:#f87171;background:#7f1d1d }
.tl-card { background:#1e293b;border-radius:8px;overflow:hidden;border-left:3px solid transparent }
.tl-card.agent { border-left-color:#4ade80 }
.tl-card.llm { border-left-color:#60a5fa }
.tl-card.tool { border-left-color:#a78bfa }
.tl-card.error { border-left-color:#f87171 }
.tl-card-header { display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid #0f172a }
.tl-event-type { font-size:0.8rem;font-weight:600 }
.tl-event-type.agent { color:#4ade80 }
.tl-event-type.llm { color:#60a5fa }
.tl-event-type.tool { color:#a78bfa }
.tl-event-type.error { color:#f87171 }
.tl-time { font-size:0.75rem;color:#64748b;font-family:monospace }
.tl-body { padding:12px 14px;font-size:0.85rem;line-height:1.5 }
.tl-body p { margin-bottom:8px }
.tl-body p:last-child { margin-bottom:0 }
.code-block { background:#0f172a;padding:8px 10px;border-radius:4px;font-family:monospace;font-size:0.78rem;white-space:pre-wrap;word-break:break-word;overflow-x:auto;margin:6px 0;line-height:1.4 }
.chat-msg { margin:6px 0;border-radius:6px;overflow:hidden }
.chat-msg .msg-role { font-size:0.7rem;font-weight:600;padding:4px 8px;text-transform:uppercase;letter-spacing:0.3px }
.chat-msg .msg-role.system { background:#1e293b;color:#94a3b8 }
.chat-msg .msg-role.user { background:#1e3a5f;color:#93c5fd }
.chat-msg .msg-role.assistant { background:#2e1065;color:#c4b5fd }
.chat-msg .msg-role.tool { background:#064e3b;color:#6ee7b7 }
.chat-msg .msg-body { background:#0f172a;padding:8px 10px;font-size:0.8rem;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow-y:auto;line-height:1.4 }
.chat-msg .msg-body.collapsed { max-height:80px;overflow:hidden;cursor:pointer;position:relative }
.chat-msg .msg-body.collapsed::after { content:'... (click to expand)';position:absolute;bottom:0;right:0;background:#0f172a;padding:0 6px;font-size:0.7rem;color:#64748b }
.tool-call-item { margin:6px 0;padding:8px 10px;background:#0f172a;border-radius:4px;border-left:2px solid #a78bfa }
.tool-call-item .tc-name { font-weight:600;color:#c4b5fd;font-size:0.82rem }
.tool-call-item .tc-args { font-family:monospace;font-size:0.78rem;color:#94a3b8;margin-top:4px;white-space:pre-wrap;word-break:break-word }
.usage-bar-wrap { margin:8px 0;padding:10px;background:#0f172a;border-radius:4px }
.usage-bar { display:flex;height:18px;border-radius:3px;overflow:hidden;margin-bottom:6px }
.usage-bar .ub-input { background:#3b82f6;transition:width .3s }
.usage-bar .ub-output { background:#a78bfa;transition:width .3s }
.usage-numbers { display:flex;gap:14px;font-size:0.78rem;flex-wrap:wrap }
.usage-numbers span { white-space:nowrap }
.usage-numbers .un-input { color:#93c5fd }
.usage-numbers .un-output { color:#c4b5fd }
.usage-numbers .un-total { color:#e2e8f0;font-weight:600 }
.usage-numbers .un-cost { color:#fbbf24 }
.tool-result { margin:6px 0;padding:8px 10px;background:#0f172a;border-radius:4px;border-left:2px solid #6ee7b7;font-family:monospace;font-size:0.78rem;white-space:pre-wrap;word-break:break-word;max-height:200px;overflow-y:auto }
.reasoning-block { margin:8px 0;padding:10px;background:#1e1b4b;border-radius:4px;border-left:3px solid #a78bfa }
.reasoning-block .r-label { font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px }
.reasoning-block .r-content { font-size:0.82rem;color:#c4b5fd;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow-y:auto;line-height:1.5;font-style:italic }
.error-block { margin:6px 0;padding:8px 10px;background:#7f1d1d;border-radius:4px;border-left:2px solid #f87171;font-family:monospace;font-size:0.8rem;white-space:pre-wrap;word-break:break-word;color:#fca5a5 }
.search-bar { margin-top:16px }
.search-bar input { width:100%;background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:8px 12px;border-radius:6px;font-size:0.85rem }
.search-bar input:focus { outline:none;border-color:#60a5fa }
.tl-highlight { animation:hl-fade 2s ease-out }
@keyframes hl-fade { 0% { background:#1e3a5f } 100% { background:transparent } }
.footer { text-align:center;padding:24px 0;color:#475569;font-size:0.8rem }
.collapsible { cursor:pointer;user-select:none }
.collapsible:hover { opacity:0.8 }
.collapsible-indicator { font-size:0.7rem;color:#64748b;margin-left:4px }
@media(max-width:640px) { body { padding:12px } .meta-grid { grid-template-columns:repeat(2,1fr) } .timeline { padding-left:28px } .tl-dot { left:-14px;width:12px;height:12px } }
</style>
</head>
<body>
<a class="back-link" href="../report.html">&larr; Back to Report</a>
<div class="run-header">
  <h1 id="run-input-display"></h1>
  <div class="meta-grid" id="meta-grid"></div>
  <div class="status-bar" id="status-bar"></div>
</div>
<div class="search-bar">
  <input type="text" id="tl-search" placeholder="Search in conversation..." oninput="filterTimeline()">
</div>
<div class="timeline" id="timeline">
  <div class="timeline-line"></div>
</div>
<div class="footer">Run detail &middot; minimal-harness eval</div>
<script id="run-data" type="application/json">
RUN_DATA_GOES_HERE
</script>
<script>
var runData = JSON.parse(document.getElementById('run-data').textContent);
var events = runData.events || [];
delete runData.events;

function esc(s) { if(!s)return'';return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') }

function fmtRel(t) { return (t>=0?'+':'')+t.toFixed(2)+'s' }

function trunc(s, n) { if(!s)return'';return s.length>n?s.slice(0,n)+'...':s }

document.getElementById('run-input-display').textContent = runData.input;

var metaHtml = '';
var metaItems = [
  {label:'Status',value:runData.status,cls:runData.status},
  {label:'Time',value:runData.timeTaken!=null?runData.timeTaken+'s':'-'},
  {label:'Total Tokens',value:runData.totalTokens.toLocaleString()},
  {label:'LLM Calls',value:runData.llmCalls},
  {label:'Tool Calls',value:runData.toolCalls},
];
if(runData.totalCost!=null) metaItems.push({label:'Total Cost',value:'$'+runData.totalCost.toFixed(6),cls:'cost'});
metaItems.forEach(function(m){
  metaHtml+='<div class="meta-item"><div class="mlabel">'+m.label+'</div><div class="mvalue'+(m.cls?' '+m.cls:'')+'">'+m.value+'</div></div>';
});
document.getElementById('meta-grid').innerHTML = metaHtml;

if(runData.error) {
  document.getElementById('status-bar').innerHTML = '<div style="margin-top:8px;padding:8px 12px;background:#7f1d1d;border-radius:4px;color:#fca5a5;font-size:0.85rem"><strong>Error:</strong> '+esc(runData.error)+'</div>';
}

function renderMessages(messages) {
  if(!messages||!messages.length) return '<div style="color:#64748b;font-style:italic">No messages</div>';
  var h='';
  for(var i=0;i<messages.length;i++) {
    var m=messages[i];
    var role=m.role||'unknown';
    var content=m.content;
    h+='<div class="chat-msg">';
    h+='<div class="msg-role '+role+'">'+esc(role)+'</div>';
    h+='<div class="msg-body';
    if(typeof content==='string' && content.length>500) h+=' collapsed';
    h+='" onclick="if(this.classList.contains(\\'collapsed\\')){this.classList.remove(\\'collapsed\\');this.scrollIntoView({behavior:\\'smooth\\',block:\\'nearest\\'})}"';
    h+='>';
    if(typeof content==='string') {
      h+=esc(content);
    } else if(Array.isArray(content)) {
      for(var j=0;j<content.length;j++) {
        var part=content[j];
        if(part.type==='text') h+=esc(part.text||'');
        else if(part.type==='image_url') h+='[Image]';
        else h+=esc(JSON.stringify(part));
      }
    } else if(content && typeof content==='object') {
      h+=esc(JSON.stringify(content));
    }
    h+='</div></div>';
  }
  return h;
}

function renderToolCalls(tcs) {
  if(!tcs||!tcs.length) return '';
  var h='<div style="margin-top:8px;font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.3px">Tool Calls</div>';
  for(var i=0;i<tcs.length;i++) {
    var tc=tcs[i];
    var fn=tc.function||tc;
    var fnName=fn.name||'unknown';
    var fnArgs='';
    try { fnArgs=typeof fn.arguments==='string'?JSON.stringify(JSON.parse(fn.arguments),null,2):JSON.stringify(fn.arguments,null,2); }
    catch(e) { fnArgs=String(fn.arguments||'') }
    h+='<div class="tool-call-item"><div class="tc-name">'+esc(fnName)+'</div><div class="tc-args">'+esc(fnArgs)+'</div></div>';
  }
  return h;
}

function renderUsage(usage) {
  if(!usage) return '';
  var inp=usage.prompt_tokens||usage.input_tokens||0;
  var out=usage.completion_tokens||usage.output_tokens||0;
  var total=usage.total_tokens||(inp+out);
  var inpPct=total>0?inp/total*100:0;
  var outPct=total>0?out/total*100:0;
  var costStr='';
  var cost = usage.total_cost;
  if(cost!=null) costStr='<span class="un-cost">$'+cost.toFixed(6)+'</span>';
  return '<div class="usage-bar-wrap">'+
    '<div class="usage-bar"><div class="ub-input" style="width:'+inpPct+'%"></div><div class="ub-output" style="width:'+outPct+'%"></div></div>'+
    '<div class="usage-numbers">'+
      '<span class="un-input">Input: '+inp.toLocaleString()+'</span>'+
      '<span class="un-output">Output: '+out.toLocaleString()+'</span>'+
      '<span class="un-total">Total: '+total.toLocaleString()+'</span>'+
      (costStr?costStr:'')+
    '</div></div>';
}

function buildTimeline() {
  if(!events||!events.length) {
    document.getElementById('timeline').innerHTML+='<div style="text-align:center;padding:40px;color:#64748b">No events recorded for this run.</div>';
    return;
  }
  var startTime=events[0].timestamp;
  var llmCounter=0;
  var toolCounter=0;
  var h='';
  for(var i=0;i<events.length;i++) {
    var evt=events[i];
    var type=evt.event_type;
    var ts=evt.timestamp;
    var data=evt.data||{};
    var relTime=ts-startTime;
    var dotClass,cardClass,typeClass,label;
    switch(type) {
      case 'agent_start': dotClass='agent';cardClass='agent';typeClass='agent';label='Agent Start'; break;
      case 'agent_end': dotClass='agent';cardClass='agent';typeClass='agent';label='Agent End'; break;
      case 'llm_start': llmCounter++;dotClass='llm';cardClass='llm';typeClass='llm';label='LLM Call #'+llmCounter; break;
      case 'llm_end': dotClass='llm';cardClass='llm';typeClass='llm';label='LLM Response #'+llmCounter; break;
      case 'tool_start': toolCounter++;dotClass='tool';cardClass='tool';typeClass='tool';label='Tool: '+(data.tool_call&&(data.tool_call.function||data.tool_call).name||'#'+toolCounter); break;
      case 'tool_end': dotClass='tool';cardClass='tool';typeClass='tool';label='Tool Result'; break;
      case 'tool_error': dotClass='error';cardClass='error';typeClass='error';label='Tool Error'; break;
      case 'error': dotClass='error';cardClass='error';typeClass='error';label='Error'; break;
      default: dotClass='';cardClass='';typeClass='';label=type; break;
    }
    h+='<div class="tl-item" data-type="'+type+'">';
    h+='<div class="tl-dot '+dotClass+'"></div>';
    h+='<div class="tl-card '+cardClass+'">';
    h+='<div class="tl-card-header"><span class="tl-event-type '+typeClass+'">'+esc(label)+'</span><span class="tl-time">'+fmtRel(relTime)+'</span></div>';
    h+='<div class="tl-body">';
    switch(type) {
      case 'agent_start':
        h+='<p><strong>User Input:</strong></p>';
        var ui=data.user_input||'(empty)';
        if(Array.isArray(ui)) {
          var txts=[];
          for(var _i=0;_i<ui.length;_i++) {
            var part=ui[_i];
            if(part.type==='text') txts.push(part.text||'');
            else if(typeof part==='string') txts.push(part);
            else txts.push(JSON.stringify(part));
          }
          h+='<div class="code-block">'+esc(txts.join('\\n'))+'</div>';
        } else if(typeof ui==='object') {
          h+='<div class="code-block">'+esc(JSON.stringify(ui,null,2))+'</div>';
        } else {
          h+='<div class="code-block">'+esc(String(ui))+'</div>';
        }
        break;
      case 'llm_start':
        h+='<p><strong>Messages</strong> ('+(data.messages?data.messages.length:0)+')</p>';
        h+=renderMessages(data.messages);
        if(data.tools&&data.tools.length) {
          h+='<div style="margin-top:8px;font-size:0.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.3px">Available Tools ('+data.tools.length+')</div>';
          h+='<div class="code-block">'+esc(data.tools.map(function(t){return t.function?t.function.name:t.name||'?'}).join(', '))+'</div>';
        }
        break;
      case 'llm_end':
        if(data.reasoning_content) {
          h+='<div class="reasoning-block"><div class="r-label">Reasoning</div><div class="r-content">'+esc(data.reasoning_content)+'</div></div>';
        }
        if(data.content) {
          h+='<p><strong>Response:</strong></p>';
          h+='<div class="code-block" style="max-height:400px;overflow-y:auto">'+esc(data.content)+'</div>';
        }
        if(data.tool_calls&&data.tool_calls.length) {
          h+=renderToolCalls(data.tool_calls);
        }
        if(data.usage) {
          h+=renderUsage(data.usage);
        }
        if(!data.content&&(!data.tool_calls||!data.tool_calls.length)&&!data.usage) {
          h+='<div style="color:#64748b;font-style:italic">No content</div>';
        }
        break;
      case 'tool_start':
        var tc=data.tool_call||{};
        var fn=tc.function||tc;
        h+='<p><strong>Arguments:</strong></p>';
        try { var args=typeof fn.arguments==='string'?JSON.stringify(JSON.parse(fn.arguments),null,2):JSON.stringify(fn.arguments,null,2); h+='<div class="code-block">'+esc(args)+'</div>'; }
        catch(e) { h+='<div class="code-block">'+esc(String(fn.arguments||'{}'))+'</div>'; }
        break;
      case 'tool_end':
        h+='<p><strong>Result:</strong></p>';
        h+='<div class="tool-result">'+esc(typeof data.result==='string'?data.result:JSON.stringify(data.result,null,2))+'</div>';
        break;
      case 'tool_error':
        h+='<div class="error-block">'+esc(data.error||'Unknown tool error')+'</div>';
        break;
      case 'error':
        h+='<div class="error-block">'+esc(data.error||'Unknown error')+'</div>';
        break;
      case 'agent_end':
        if(data.response) {
          h+='<p><strong>Final Response:</strong></p>';
          h+='<div class="code-block" style="max-height:300px;overflow-y:auto">'+esc(data.response)+'</div>';
        }
        if(data.time_taken!=null) h+='<p style="margin-top:8px"><span style="color:#94a3b8">Time taken:</span> '+data.time_taken.toFixed(2)+'s</p>';
        if(data.exceeded) h+='<p><span style="color:#facc15">Max iterations exceeded</span></p>';
        if(data.interrupted) h+='<p><span style="color:#facc15">Interrupted</span></p>';
        if(!data.response&&!data.time_taken) h+='<div style="color:#64748b;font-style:italic">Agent ended</div>';
        break;
      default:
        h+='<div class="code-block">'+esc(JSON.stringify(data,null,2))+'</div>';
    }
    h+='</div></div></div>';
  }
  document.getElementById('timeline').innerHTML = h;
}

function filterTimeline() {
  var q=(document.getElementById('tl-search').value||'').toLowerCase();
  var items=document.querySelectorAll('.tl-item');
  if(!q) { items.forEach(function(el){el.style.display=''}); return; }
  var found=0;
  items.forEach(function(el){
    var txt=el.textContent.toLowerCase();
    if(txt.includes(q)) { el.style.display='';found++; }
    else { el.style.display='none' }
  });
}

buildTimeline();
</script>
</body>
</html>"""
