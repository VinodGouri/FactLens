const CSRF = document.querySelector('meta[name="csrf-token"]').content;
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function predictApi(statement) {
  const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json','X-CSRF-Token':CSRF}, body:JSON.stringify({statement})});
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    throw new Error('Server error (status ' + r.status + '). Please try again later.');
  }
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Request failed');
  return j;
}
function renderResult(r) {
  const cls = r.prediction === 'Fake' ? 'verdict-fake' : 'verdict-real';
  const ev = r.evidence.map(e => `<li class="small">${escapeHtml(e.text)} <span class="badge ${e.verdict==='Fake'?'verdict-fake':'verdict-real'}">${e.verdict}</span></li>`).join('');
  return `<div><span class="badge ${cls} fs-6">${r.prediction}</span> confidence <b>${(r.confidence*100).toFixed(1)}%</b> · P(fake)=${r.p_fake.toFixed(3)}
    <a class="small ms-2" href="/history/${r.id}">details</a>${ev ? `<div class="small text-muted mt-2">Retrieved fact-checked claims${r.uses_rag ? "" : " (reference only)"}:</div><ul class="mb-0">${ev}</ul>` : ''}</div>`;
}
