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
  const tabs = document.querySelectorAll('.tab');
  const contents = document.querySelectorAll('.content-wrapper');

  function showStatus(el, msg) {
    el.textContent = msg;
  }

  function collect() {
    return {
      num_epochs: parseInt(document.getElementById('epochInput').value),
      batch_size: parseInt(document.getElementById('batchInput').value),
      learning_rate: parseFloat(document.getElementById('lrInput').value),
      dropout_ratio: parseFloat(dropoutInput.value),
      warmup_steps: parseInt(document.getElementById('warmupStepsInput').value),
      max_sequence_length: parseInt(document.getElementById('maxSeqLenInput').value),
      num_heads: parseInt(document.getElementById('numHeadsInput').value),
      num_encoder_layers: parseInt(document.getElementById('numEncoderLayersInput').value),
      num_decoder_layers: parseInt(document.getElementById('numDecoderLayersInput').value),
      model_dim: parseInt(document.getElementById('hiddenDimInput').value),
      ff_dim: parseInt(document.getElementById('ffDimInput').value),
      top_k: parseInt(document.getElementById('topkInput').value),
      temperature: parseFloat(document.getElementById('tempInput').value)
    };
  }

  function addMessage(text, sender) {
    const div = document.createElement('div');
    div.classList.add('message', sender);
    div.textContent = text;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  const api = window.pywebview ? window.pywebview.api : null;
  api.get_config().then(r => {
    const cfg = r.data;
    document.getElementById('epochInput').value = cfg.num_epochs;
    document.getElementById('batchInput').value = cfg.batch_size;
    document.getElementById('lrInput').value = cfg.learning_rate;
    document.getElementById('warmupStepsInput').value = cfg.warmup_steps;
    document.getElementById('maxSeqLenInput').value = cfg.max_sequence_length;
    document.getElementById('numHeadsInput').value = cfg.num_heads;
    document.getElementById('numEncoderLayersInput').value = cfg.num_encoder_layers;
    document.getElementById('numDecoderLayersInput').value = cfg.num_decoder_layers;
    document.getElementById('hiddenDimInput').value = cfg.model_dim;
    document.getElementById('ffDimInput').value = cfg.ff_dim;
    document.getElementById('topkInput').value = cfg.top_k;
    document.getElementById('tempInput').value = cfg.temperature;
    dropoutInput.value = cfg.dropout_ratio;
    dropoutValue.textContent = cfg.dropout_ratio.toFixed(2);
  });

  dropoutInput.addEventListener('input', e => {
    dropoutValue.textContent = parseFloat(e.target.value).toFixed(2);
  });

  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
  });

  messageInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const content = document.getElementById(tab.dataset.tab + 'Content');
      if (content) content.classList.add('active');
    });
  });

  trainBtn.addEventListener('click', () => {
    const cfg = collect();
    api.update_config(cfg)
      .then(() => api.start_train('.'))
      .then(() => showStatus(trainStatus, 'training...'));
  });

  saveBtn.addEventListener('click', () => {
    const cfg = collect();
    api.update_config(cfg).then(() => showStatus(trainStatus, 'saved'));
  });

  sendBtn.addEventListener('click', () => {
    const msg = messageInput.value.trim();
    if (!msg) return;
    addMessage(msg, 'user');
    messageInput.value = '';
    api.inference(msg).then(r => addMessage(r.data.answer, 'bot'));
  });

  setInterval(() => {
    api.get_status().then(r => {
      showStatus(trainStatus, r.data.message);
    });
  }, 3000);
});
