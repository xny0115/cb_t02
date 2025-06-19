import json
import subprocess


def send_two_messages():
    script = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<div id="chatHistory"></div>', { runScripts: 'outside-only' });
const { window } = dom;
window.requestAnimationFrame = fn => fn();
global.requestAnimationFrame = window.requestAnimationFrame;
const chatHistory = window.document.getElementById('chatHistory');
function appendChat(role, text){
  const div = window.document.createElement('div');
  div.className = role === 'USER' ? 'user-msg' : 'bot-msg';
  div.textContent = text;
  chatHistory.appendChild(div);
  requestAnimationFrame(()=>{ chatHistory.scrollTop = chatHistory.scrollHeight; });
}
Object.defineProperty(chatHistory, 'scrollHeight', { value: 100, writable: true });
appendChat('USER', 'hi');
Object.defineProperty(chatHistory, 'scrollHeight', { value: 200, writable: true });
appendChat('BOT', 'bye');
console.log(JSON.stringify({ top: chatHistory.scrollTop, height: chatHistory.scrollHeight }));
"""
    out = subprocess.check_output(['node', '-e', script])
    return json.loads(out.decode())


def test_auto_scroll():
    res = send_two_messages()
    assert res['top'] == res['height']
