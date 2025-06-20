import json
import subprocess


SCRIPT = """
const { JSDOM } = require('jsdom');
const dom = new JSDOM('<ul id="list" style="display:flex;flex-direction:column"></ul>', {runScripts:'outside-only'});
const list = dom.window.document.getElementById('list');
function add(msg){const li = dom.window.document.createElement('li');li.textContent=msg;list.prepend(li);}
for(let i=0;i<5;i++) add('m'+i);
console.log(JSON.stringify({first:list.firstChild.textContent}));
""";


def test_chat_order():
    out = subprocess.check_output(['node', '-e', SCRIPT])
    data = json.loads(out.decode())
    assert data['first'] == 'm4'
