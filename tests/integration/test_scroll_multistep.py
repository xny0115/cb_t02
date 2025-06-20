import json, subprocess


def run_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto"></div>', { runScripts: 'outside-only' });
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
appendChat('USER','x');
setTimeout(()=>{console.log(JSON.stringify({top:box.scrollTop,height:box.scrollHeight,client:box.clientHeight}));},40);
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_scroll_multistep():
    res = run_script()
    assert res["top"] == 0
