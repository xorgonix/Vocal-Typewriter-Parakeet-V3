const statusEl = document.getElementById('connectionStatus');
const micSelect = document.getElementById('micSelect');
const recordBtn = document.getElementById('recordBtn');
const restartBtn = document.getElementById('restartBtn');
const logsBtn = document.getElementById('logsBtn');
const logsModal = document.getElementById('logsModal');
const closeLogsBtn = document.getElementById('closeLogsBtn');
const logsContainer = document.getElementById('logsContainer');

const wpmEl = document.getElementById('wpmValue');
const latencyEl = document.getElementById('latencyValue');
const waveCanvas = document.getElementById('waveCanvas');
const transcriptionBox = document.getElementById('transcriptionBox');
const copyBtn = document.getElementById('copyBtn');
const clearBtn = document.getElementById('clearBtn');

let ws;
let audioContext;
let processor;
let source;
let isRecording = false;
let waveData = [];

// Canvas Setup
const ctx = waveCanvas.getContext('2d');
let animationId;

function resizeCanvas() {
    waveCanvas.width = waveCanvas.parentElement.clientWidth;
    waveCanvas.height = waveCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave() {
    ctx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);
    ctx.fillStyle = '#00ff9d';
    
    const barWidth = 4;
    const gap = 2;
    const maxBars = Math.floor(waveCanvas.width / (barWidth + gap));
    
    // Keep last N data points
    if (waveData.length > maxBars) {
        waveData = waveData.slice(waveData.length - maxBars);
    }

    waveData.forEach((val, i) => {
        const height = val * waveCanvas.height;
        const x = i * (barWidth + gap);
        const y = (waveCanvas.height - height) / 2;
        
        // Neon glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00ff9d';
        
        ctx.fillRect(x, y, barWidth, height);
        
        ctx.shadowBlur = 0;
    });

    animationId = requestAnimationFrame(drawWave);
}
drawWave();

// WebSocket Connection
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; 
    
    ws = new WebSocket(`${protocol}//${host}/ws`);

    ws.onopen = () => {
        statusEl.textContent = 'Connected';
        statusEl.className = 'status-indicator connected';
    };

    ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        statusEl.className = 'status-indicator disconnected';
        setTimeout(connect, 3000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'wave') {
            waveData.push(data.value);
        } else if (data.type === 'transcript') {
            // Update metrics from transcript message
            if (data.speed) wpmEl.textContent = data.speed.toFixed(1) + 'x';
            if (data.latency) latencyEl.textContent = data.latency.toFixed(0);

            // Remove highlight from previous elements
            const previousLatest = transcriptionBox.querySelectorAll('.latest-text');
            previousLatest.forEach(el => el.classList.remove('latest-text'));

            const p = document.createElement('div');
            p.textContent = data.text;
            p.style.marginBottom = '1rem';
            
            // Add highlight class
            p.classList.add('latest-text');

            transcriptionBox.appendChild(p);
            
            // Force scroll to view the new element
            p.scrollIntoView({ behavior: "smooth", block: "end" });
        }
    };
}

// Audio Handling
async function getMicrophones() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const mics = devices.filter(d => d.kind === 'audioinput');
        micSelect.innerHTML = mics.map(m => `<option value="${m.deviceId}">${m.label || 'Default Microphone'}</option>`).join('');
    } catch (e) {
        console.error('Error listing mics:', e);
    }
}

async function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: micSelect.value ? { exact: micSelect.value } : undefined,
                channelCount: 1,
                sampleRate: 16000 
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = (e) => {
            if (!isRecording) return;
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0);
                ws.send(inputData.buffer);
            }
        };

        isRecording = true;
        recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
        recordBtn.className = 'icon-btn recording';

    } catch (e) {
        console.error('Error starting audio:', e);
        alert('Could not access microphone');
    }
}

function stopRecording() {
    isRecording = false;
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    recordBtn.className = 'icon-btn';
    
    if (source) source.disconnect();
    if (processor) processor.disconnect();
    if (audioContext) audioContext.close();
}

recordBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// Restart Button Logic
restartBtn.addEventListener('click', () => {
    // 1. Stop recording if active
    if (isRecording) {
        stopRecording();
    }
    
    // 2. Clear UI elements
    transcriptionBox.innerHTML = '';
    wpmEl.textContent = '0';
    latencyEl.textContent = '0';
    waveData = [];
    
    // 3. Reconnect WebSocket to reset server state if needed
    if (ws) {
        ws.close();
        // connect() will be called automatically by onclose handler
        // but we can force status update immediately
        statusEl.textContent = 'Restarting...';
    }
});

// Logs Button Logic
logsBtn.addEventListener('click', async () => {
    logsModal.classList.remove('hidden');
    
    try {
        const res = await fetch('/logs');
        const data = await res.json();
        
        logsContainer.innerHTML = '';
        
        const history = data.history.slice().reverse();
        
        history.forEach(row => {
            const time = new Date(row.timestamp * 1000).toLocaleTimeString();
            
            const entryDiv = document.createElement('div');
            entryDiv.className = 'log-entry';
            
            entryDiv.innerHTML = `
                <div class="log-stats">
                    <div class="stat-item">
                        <span class="stat-label">Time</span>
                        <span class="stat-value">${time}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Latency</span>
                        <span class="stat-value highlight">${row.latency_ms} ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Speed</span>
                        <span class="stat-value">${row.speed_x}x</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Duration</span>
                        <span class="stat-value">${row.duration_s}s</span>
                    </div>
                </div>
                <div class="log-divider"></div>
                <div class="log-text">${row.text}</div>
            `;
            
            logsContainer.appendChild(entryDiv);
        });
        
    } catch (e) {
        console.error('Failed to fetch logs', e);
        logsContainer.innerHTML = '<div class="log-entry" style="text-align:center; color:#ff4757;">Error loading logs</div>';
        
        // Fallback to emojis if Font Awesome fails or other errors occur (per user request)
        recordBtn.innerHTML = '🎙️';
        restartBtn.innerHTML = '🔄';
        logsBtn.innerHTML = '📄';
    }
});

closeLogsBtn.addEventListener('click', () => {
    logsModal.classList.add('hidden');
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === logsModal) {
        logsModal.classList.add('hidden');
    }
});

copyBtn.addEventListener('click', () => {
    const text = Array.from(transcriptionBox.children).map(div => div.textContent).join('\n');
    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = originalText, 2000);
    });
});

clearBtn.addEventListener('click', () => {
    transcriptionBox.innerHTML = '';
});

// Initialize
getMicrophones();
connect();
