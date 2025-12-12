// tabs
document.querySelectorAll('.tablink').forEach(btn=>{
  btn.addEventListener('click', ()=> {
    document.querySelectorAll('.tablink').forEach(x=>x.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tabcontent').forEach(tc=>tc.style.display='none');
    document.getElementById(tab).style.display='block';
  });
});

// symptom form
document.getElementById('symptomForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const form = new FormData(e.target);
  const res = await fetch('/predict-symptoms', {method:'POST', body: form});
  if (!res.ok) { alert('Prediction failed'); return; }
  const data = await res.json();
  const lines = [];
  lines.push('Selected: ' + data.selected.join(', '));
  lines.push(`Predicted: ${data.predicted} (${(data.confidence*100).toFixed(2)}%)`);
  lines.push('');
  lines.push('Detailed probabilities:');
  data.probs.forEach(([d,p])=>{ lines.push(` - ${d}: ${(p*100).toFixed(2)}%`); });
  document.getElementById('symptomText').innerText = lines.join('\\n');
  document.getElementById('symptomResult').style.display='block';
});

// image form
document.getElementById('imageForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const form = new FormData(e.target);
  const res = await fetch('/predict-image', {method:'POST', body: form});
  if (!res.ok) { alert('Prediction failed'); return; }
  const data = await res.json();
  document.getElementById('imageText').innerText = `Predicted: ${data.predicted} — Confidence: ${(data.confidence*100).toFixed(2)}%`;
  document.getElementById('imageResult').style.display='block';
});

// audio form upload
document.getElementById('audioForm').addEventListener('submit', async (e)=>{
  e.preventDefault();
  const form = new FormData(e.target);
  const res = await fetch('/predict-audio', {method:'POST', body: form});
  if (!res.ok) { alert('Audio prediction failed'); return; }
  const data = await res.json();
  document.getElementById('audioText').innerText = `Predicted: ${data.predicted} — Confidence: ${(data.confidence*100).toFixed(2)}%`;
  document.getElementById('audioResult').style.display='block';
});

// microphone recording (3 seconds) using MediaRecorder
let mediaRecorder, audioChunks = [];
const recordBtn = document.getElementById('recordBtn');
const recStatus = document.getElementById('recStatus');
recordBtn.addEventListener('click', async ()=>{
  if (!navigator.mediaDevices) { alert('Microphone not supported'); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async ()=>{
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
      const form = new FormData();
      form.append('audio', file);
      recStatus.innerText = 'Uploading...';
      const res = await fetch('/predict-audio', { method: 'POST', body: form });
      if (!res.ok) { alert('Upload failed'); recStatus.innerText=''; return; }
      const data = await res.json();
      document.getElementById('audioText').innerText = `Predicted: ${data.predicted} — Confidence: ${(data.confidence*100).toFixed(2)}%`;
      document.getElementById('audioResult').style.display='block';
      recStatus.innerText='';
    };
    mediaRecorder.start();
    recStatus.innerText = 'Recording...';
    setTimeout(()=>{ mediaRecorder.stop(); recStatus.innerText='Processing...'; }, 3000);
  } catch (err) {
    alert('Could not access microphone: ' + err.message);
  }
});