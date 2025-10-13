const http = require('http');
const url = process.env.API_URL || 'http://127.0.0.1:8080';
const port = 3000;

const html = `<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>OPRA Chat</title>
<style>
body { font-family: sans-serif; max-width: 860px; margin: 2rem auto; }
#log { border: 1px solid #ddd; padding: 1rem; height: 400px; overflow: auto; white-space: pre-wrap; }
input, button { font-size: 1rem; }
/* Modal */
.modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: none; align-items: center; justify-content: center; }
.modal { background: #fff; width: 560px; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 16px 18px; }
.modal h2 { margin: 0 0 8px 0; font-size: 18px; }
.grid { display: grid; grid-template-columns: 160px 1fr; gap: 8px; font-size: 13px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; word-break: break-all; }
.actions { display:flex; justify-content: flex-end; margin-top: 12px; }
</style>
</head>
<body>
<h1>OPRA Chat (MVP)</h1>
<p>API: ${url}</p>
<div>
  <label>Index path: <input id="idx" value="/data/index/demo_index.jsonl" style="width: 480px" /></label>
</div>
<div>
  <input id="q" placeholder="Ask a question" style="width: 640px" />
  <button onclick="send()">Send</button>
  <button onclick="buildReport()">Build Report</button>
  <button onclick="openReceiptModal()">Receipts…</button>
</div>
<div id="log"></div>
<div id="receiptModal" class="modal-backdrop" onclick="if(event.target.id==='receiptModal'){closeReceiptModal();}">
  <div class="modal">
    <h2>Receipt</h2>
    <div class="grid" id="receiptGrid"></div>
    <div class="actions"><button onclick="closeReceiptModal()">Close</button></div>
  </div>
</div>
<script>
let lastResponse = null;
async function send() {
  const q = document.getElementById('q').value;
  const index_path = document.getElementById('idx').value;
    const payload = { index_path, q, backend: 'jsonl', k: 5, embed_model: 'bge-small-en-v1.5', synth_mode: 'extractive' };
    const r = await fetch('${url}/v1/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const j = await r.json();
    lastResponse = j;
  const el = document.getElementById('log');
  const cits = (j.citations || []).slice(0,3).map(function(c,i){
    const idx = (i+1);
    const sp = (c.source_path || '');
    const pg = (c.page_number == null ? '' : c.page_number);
    return idx + '. ' + sp + ' p' + pg;
  }).join('\n');
  el.textContent += "\n> " + q + "\nAnswer: " + (j.answer || '(abstain)') + "\nCitations:\n" + cits + "\n";
  el.scrollTop = el.scrollHeight;
}
  async function buildReport() {
    const index_path = document.getElementById('idx').value;
    const q = document.getElementById('q').value;
    const payload = { title: 'OPRA Report', index_path, q, backend: 'jsonl', k: 5, embed_model: 'bge-small-en-v1.5', fmt: 'docx' };
    const r = await fetch('${url}/v1/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const j = await r.json();
    const el = document.getElementById('log');
  el.textContent += "\nReport: " + j.path + " (bundle_hash=" + j.bundle_hash + ")\n";
    el.scrollTop = el.scrollHeight;
  }
  window.addEventListener('keydown', (e)=>{ if(e.key==='Enter') send(); });

  function openReceiptModal(){
    const modal = document.getElementById('receiptModal');
    const grid = document.getElementById('receiptGrid');
    grid.innerHTML='';
    const r = (lastResponse && (lastResponse.settle_receipt || lastResponse.receipt)) || null;
    // Prefer settle_receipt fields (e2e), else receipt
    const flat = {};
    if(r && typeof r==='object'){
      // Map known fields
      flat['ΔH'] = r.deltaH_total;
      flat['Final residual'] = r.final_residual;
      flat['CG iters'] = r.cg_iters;
      flat['Edge hash'] = r.edge_hash;
      // Optional fields possibly from ingest/query receipts
      const ing = (lastResponse && lastResponse.ingest_receipt) || {};
      flat['File SHA256'] = ing.file_sha256 || undefined;
      flat['Index SHA256'] = (ing && ing.index_sha256) || (lastResponse && lastResponse.receipt && lastResponse.receipt.index_sha256) || undefined;
      flat['Model SHA256'] = (lastResponse && lastResponse.receipt && lastResponse.receipt.query_model_sha256) || undefined;
      flat['Dim'] = (lastResponse && lastResponse.receipt && lastResponse.receipt.dim) || undefined;
      flat['ε'] = (lastResponse && lastResponse.receipt && lastResponse.receipt.epsilon);
      flat['τ'] = (lastResponse && lastResponse.receipt && lastResponse.receipt.tau);
    }
    const entries = Object.entries(flat).filter(([k,v])=> v !== undefined && v !== null);
    if(entries.length===0){ grid.innerHTML = '<div>No receipt available</div>'; }
    else {
      for(const [k,v] of entries){
        const kdiv = document.createElement('div'); kdiv.textContent = k;
        const vdiv = document.createElement('div'); vdiv.className='mono'; vdiv.textContent = (typeof v==='number' ? (Number.isInteger(v)? String(v) : v.toFixed(6)) : String(v));
        grid.appendChild(kdiv); grid.appendChild(vdiv);
      }
    }
    modal.style.display='flex';
  }
  function closeReceiptModal(){
    const modal = document.getElementById('receiptModal');
    modal.style.display='none';
  }
</script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(html);
});
server.listen(port, '127.0.0.1', () => console.log(`UI listening on http://127.0.0.1:${port}`));
