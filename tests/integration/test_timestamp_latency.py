import json, subprocess


def run_dom():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
if(!window.requestAnimationFrame){ window.requestAnimationFrame = cb => setTimeout(cb,16); }
function appendChat(role,text,latency=0){
  const divMsg = window.document.createElement('div');
  divMsg.className = role==='USER'?'user-msg':'bot-msg';
  divMsg.textContent = text;
  const meta = window.document.createElement('span');
  meta.className='meta';
  meta.textContent=new Date().toLocaleTimeString()+ ' · '+ latency+' ms';
  divMsg.appendChild(meta);
  box.prepend(divMsg);
  function scrollTop(){box.scrollTop=0;}
  scrollTop();
  window.requestAnimationFrame(scrollTop);
  window.setTimeout(scrollTop,30);
}
appendChat('BOT','hi',23);
console.log(JSON.stringify({html:box.innerHTML}));
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_timestamp_latency():
    res = run_dom()
    assert "ms" in res["html"]
