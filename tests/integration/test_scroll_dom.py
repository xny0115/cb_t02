import json
import subprocess


def run_dom_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const chatHistory = window.document.getElementById('chatHistory');
window.HTMLElement.prototype.scrollIntoView = function(){ chatHistory.scrollTop = chatHistory.scrollHeight; };
function appendChat(role, text){
  const div = window.document.createElement('div');
  div.className = role === 'USER' ? 'user-msg' : 'bot-msg';
  div.textContent = text;
  chatHistory.appendChild(div);
  div.scrollIntoView({ behavior: 'auto', block: 'end' });
}
for(let i=0;i<20;i++) appendChat('USER','m'+i);
console.log(JSON.stringify({top:chatHistory.scrollTop,height:chatHistory.scrollHeight,client:chatHistory.clientHeight}));
"""
    out = subprocess.check_output(['node', '-e', script])
    return json.loads(out.decode())


def test_scroll_dom():
    res = run_dom_script()
    assert res['top'] + res['client'] >= res['height'] - 1

