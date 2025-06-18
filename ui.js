document.addEventListener('DOMContentLoaded', () => {
  const trainBtn = document.getElementById('trainBtn');
  const saveBtn = document.getElementById('saveSettingsBtn');
  const sendBtn = document.getElementById('sendBtn');
  const messageInput = document.getElementById('messageInput');
  const chatArea = document.getElementById('chatArea');
  const trainStatus = document.getElementById('trainStatus');
  const cpuStatus = document.getElementById('cpuStatus');
  const gpuStatus = document.getElementById('gpuStatus');
  const dropoutInput = document.getElementById('dropoutInput');
  const dropoutValue = document.getElementById('dropoutValue');

  function showStatus(el, msg) {
    el.textContent = msg;
  }

  function collect() {
    return {
      num_epochs: parseInt(document.getElementById('epochInput').value),
      batch_size: parseInt(document.getElementById('batchInput').value),
      learning_rate: parseFloat(document.getElementById('lrInput').value),
      dropout_ratio: parseFloat(dropoutInput.value)
    };
  }

  function addMessage(text, sender) {
    const div = document.createElement('div');
    div.classList.add('message', sender);
    div.textContent = text;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  fetch('/config').then(r => r.json()).then(r => {
    const cfg = r.data;
    document.getElementById('epochInput').value = cfg.num_epochs;
    document.getElementById('batchInput').value = cfg.batch_size;
    document.getElementById('lrInput').value = cfg.learning_rate;
    dropoutInput.value = cfg.dropout_ratio;
    dropoutValue.textContent = cfg.dropout_ratio.toFixed(2);
  });

  dropoutInput.addEventListener('input', e => {
    dropoutValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  trainBtn.addEventListener('click', () => {
    const cfg = collect();
    fetch('/config', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg)})
      .then(() => fetch('/train', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({data_path: '.'})}))
      .then(() => showStatus(trainStatus, 'training...'));
  });

  saveBtn.addEventListener('click', () => {
    const cfg = collect();
    fetch('/config', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg)})
      .then(() => showStatus(trainStatus, 'saved'));
  });

  sendBtn.addEventListener('click', () => {
    const msg = messageInput.value.trim();
    if (!msg) return;
    addMessage(msg, 'user');
    messageInput.value = '';
    fetch('/infer', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: msg})})
      .then(r => r.json())
      .then(r => addMessage(r.data.answer, 'bot'));
  });

  setInterval(() => {
    fetch('/status').then(r => r.json()).then(r => {
      showStatus(trainStatus, r.data.message);
    });
  }, 3000);
});
