import json, subprocess


def run_dom():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="display:flex; flex-direction:column-reverse"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
function appendChat(role,text,latency=0){
  const divMsg = window.document.createElement('div');
  divMsg.className = role==='USER'?'user-msg':'bot-msg';
  divMsg.textContent = text;
  const meta = window.document.createElement('span');
  meta.className='meta';
  meta.textContent=new Date().toLocaleTimeString()+ ' · '+ latency+' ms';
  divMsg.appendChild(meta);
  box.appendChild(divMsg);
}
appendChat('BOT','hi',23);
console.log(JSON.stringify({html:box.innerHTML}));
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_timestamp_latency():
    res = run_dom()
    assert "ms" in res["html"]
