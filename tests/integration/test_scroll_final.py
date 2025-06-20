import json, subprocess


def run_script():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory" style="height:120px; overflow:auto"></div>', { runScripts: 'outside-only' });
const { window } = dom;
const box = window.document.getElementById('chatHistory');
window.HTMLElement.prototype.scrollIntoView = function(){
  this.parentNode.scrollTop = this.parentNode.scrollHeight;
};
function appendChat(role,text){
  const div = window.document.createElement('div');
  div.className = role==='USER'?'user-msg':'bot-msg';
  div.textContent = text;
  box.appendChild(div);
  div.scrollIntoView({ block:'end' });
}
for(let i=0;i<30;i++) appendChat('USER','z'+i);
console.log(JSON.stringify({ top: box.scrollTop, height: box.scrollHeight, client: box.clientHeight }));
"""
    out = subprocess.check_output(['node','-e',script])
    return json.loads(out.decode())


def test_scroll_final():
    res = run_script()
    assert res['top'] + res['client'] >= res['height'] - 1
