import json, subprocess


def run_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto; display:flex; flex-direction:column-reverse"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
function appendChat(role,text,latency=0){
  const divMsg = window.document.createElement('div');
  divMsg.className = role==='USER'?'user-msg':'bot-msg';
  divMsg.textContent = text;
  const meta1 = window.document.createElement('span');
  meta1.className='meta';
  meta1.textContent=new Date().toLocaleTimeString()+ ' · '+ latency+' ms';
  divMsg.appendChild(meta1);
  box.appendChild(divMsg);
  const meta2 = window.document.createElement('span');
  meta2.className='meta';
  meta2.textContent=new Date().toLocaleTimeString()+ ' · '+ latency+' ms';
  divMsg.appendChild(meta2);
  box.appendChild(divMsg);
}
appendChat('USER','x');
setTimeout(()=>{console.log(JSON.stringify({top:box.scrollTop}));},40);
"""
    out = subprocess.check_output(["node", "-e", script])
    return json.loads(out.decode())


def test_scroll_multistep():
    res = run_script()
    assert res["top"] == 0
