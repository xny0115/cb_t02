import json, subprocess


def run_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto; display:flex; flex-direction:column-reverse"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
function appendChat(role,text){
  const div = window.document.createElement('div');
  div.className = role==='USER'?'user-msg':'bot-msg';
  div.textContent = text;
  box.appendChild(div);
}
for(let i=0;i<30;i++) appendChat('USER','z'+i);
console.log(JSON.stringify({ top: box.scrollTop }));
"""
    out = subprocess.check_output(['node','-e',script])
    return json.loads(out.decode())


def test_scroll_final():
    res = run_script()
    assert res['top'] == 0
