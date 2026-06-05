// API base URL configuration (empty for relative paths since we serve via FastAPI)
const API_BASE = "";

// Global instances
let wavesurfer = null;
let featuresChart = null;
let attentionChart = null;
let mediaRecorder = null;
let audioChunks = [];
let recordTimerInterval = null;
let recordStartTime = null;

// Web Audio API Recorder instances
let audioContext = null;
let scriptProcessor = null;
let mediaStreamSource = null;
let recBuffers = [];
let recLength = 0;

// App State
let appState = {
    audioFile: null,      // Selected or recorded file blob
    audioPath: null,      // Path on server after upload
    sessionId: null,      // Session token
    duration: 0,
    segmentsCount: 0,
    stepResults: {},      // Cached step details
    currentExecutingStep: 0,
    dbInfo: null
};

// DOM References
const patientIdInput = document.getElementById("patientId");
const tabUpload = document.getElementById("tabUpload");
const tabRecord = document.getElementById("tabRecord");
const uploadArea = document.getElementById("uploadArea");
const recordArea = document.getElementById("recordArea");
const dropZone = document.getElementById("dropZone");
const audioFileInput = document.getElementById("audioFileInput");
const fileNameLabel = document.getElementById("fileNameLabel");
const btnStartRecord = document.getElementById("btnStartRecord");
const btnStopRecord = document.getElementById("btnStopRecord");
const recordPulse = document.getElementById("recordPulse");
const recordTimer = document.getElementById("recordTimer");
const btnRunFullPipeline = document.getElementById("btnRunFullPipeline");
const btnRunStepByStep = document.getElementById("btnRunStepByStep");
const btnReset = document.getElementById("btnReset");
const terminalLogs = document.getElementById("terminalLogs");
const ssiValue = document.getElementById("ssiValue");
const panssValue = document.getElementById("panssValue");
const severityBadge = document.getElementById("severityBadge");
const severityDesc = document.getElementById("severityDesc");
const historyBody = document.getElementById("historyBody");
const dbStatus = document.getElementById("dbStatus");
const dbName = document.getElementById("dbName");
const logModal = document.getElementById("logModal");
const closeModal = document.getElementById("closeModal");
const modalLogContent = document.getElementById("modalLogContent");

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener("DOMContentLoaded", () => {
    initWaveSurfer();
    initCharts();
    bindEvents();
    loadHistory();
    checkDatabaseStatus();
});

function initWaveSurfer() {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4f46e5',
        progressColor: '#c084fc',
        cursorColor: '#ffffff',
        height: 60,
        responsive: true,
        barWidth: 2,
        barGap: 3
    });
}

function initCharts() {
    // 1. Articulatory Features Radar Chart
    const ctxRadar = document.getElementById('featuresChart').getContext('2d');
    featuresChart = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Pause Ratio', 'Pitch Variation', 'Speech Rate (ZCR)', 'Articulation (Inv Energy)'],
            datasets: [{
                label: 'Acoustic Features',
                data: [0, 0, 0, 0],
                backgroundColor: 'rgba(139, 92, 246, 0.2)',
                borderColor: 'rgba(139, 92, 246, 0.8)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(139, 92, 246, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#9ca3af', font: { size: 10 } },
                    ticks: { display: false },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // 2. Attention Distribution Chart
    const ctxLine = document.getElementById('attentionChart').getContext('2d');
    attentionChart = new Chart(ctxLine, {
        type: 'line',
        data: {
            labels: Array.from({length: 30}, (_, i) => `S-${i+1}`),
            datasets: [{
                label: 'Attention Weights',
                data: Array(30).fill(0),
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderColor: '#3b82f6',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af', font: { size: 9 } },
                    min: 0
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// =============================================================================
// DOM EVENT BINDING
// =============================================================================
function bindEvents() {
    // Tabs
    tabUpload.addEventListener("click", () => {
        tabUpload.classList.add("active");
        tabRecord.classList.remove("active");
        uploadArea.classList.add("active");
        recordArea.classList.remove("active");
        stopRecordingIfActive();
    });

    tabRecord.addEventListener("click", () => {
        tabRecord.classList.add("active");
        tabUpload.classList.remove("active");
        recordArea.classList.add("active");
        uploadArea.classList.remove("active");
    });

    // File Drop Zone
    dropZone.addEventListener("click", () => audioFileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Recording controls
    btnStartRecord.addEventListener("click", startRecording);
    btnStopRecord.addEventListener("click", stopRecording);

    // Action Triggers
    btnRunFullPipeline.addEventListener("click", executeFullPipeline);
    btnRunStepByStep.addEventListener("click", executeStepByStep);
    btnReset.addEventListener("click", resetState);

    // Modal
    closeModal.addEventListener("click", () => logModal.classList.remove("active"));
    logModal.addEventListener("click", (e) => {
        if (e.target === logModal) logModal.classList.remove("active");
    });
}

// =============================================================================
// INPUT HANDLERS
// =============================================================================
function handleFileSelect(file) {
    if (!file.name.endsWith(".wav")) {
        log("Error: Only standard audio files in WAV format (.wav) are supported.", "error");
        return;
    }
    appState.audioFile = file;
    fileNameLabel.textContent = file.name;
    log(`Selected file: ${file.name} (${formatBytes(file.size)})`, "system");

    // Load into WaveSurfer
    const url = URL.createObjectURL(file);
    wavesurfer.load(url);

    // Enable buttons
    btnRunFullPipeline.disabled = false;
    btnRunStepByStep.disabled = false;
    btnReset.style.display = "block";
}

// =============================================================================
// VOICE RECORDER ENGINE (Pure JS WAV Recorder)
// =============================================================================
function bufferToWav(buffer, sampleRate) {
    const bufferLength = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + bufferLength * 2);
    const view = new DataView(arrayBuffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + bufferLength * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, bufferLength * 2, true);

    floatTo16BitPCM(view, 44, buffer);

    return arrayBuffer;
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

async function startRecording() {
    recBuffers = [];
    recLength = 0;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Initialize AudioContext at 16000Hz to automatically resample the audio
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        mediaStreamSource = audioContext.createMediaStreamSource(stream);
        
        // 4096 buffer size, 1 input channel, 1 output channel
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        
        scriptProcessor.onaudioprocess = (e) => {
            const channelData = e.inputBuffer.getChannelData(0);
            recBuffers.push(new Float32Array(channelData));
            recLength += channelData.length;
        };
        
        mediaStreamSource.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);
        
        btnStartRecord.disabled = true;
        btnStopRecord.disabled = false;
        recordPulse.classList.add("recording");
        log("Recording voice sample... Speak clearly into the microphone.", "system");

        recordStartTime = Date.now();
        recordTimerInterval = setInterval(() => {
            const elapsed = Date.now() - recordStartTime;
            const seconds = Math.floor((elapsed / 1000) % 60);
            const minutes = Math.floor((elapsed / (1000 * 60)) % 60);
            recordTimer.textContent = `${pad(minutes)}:${pad(seconds)}`;
        }, 1000);

    } catch (err) {
        log(`Microphone access failed: ${err.message}`, "error");
    }
}

function stopRecording() {
    if (scriptProcessor && mediaStreamSource) {
        scriptProcessor.disconnect();
        mediaStreamSource.disconnect();
        if (audioContext && audioContext.state !== "closed") {
            audioContext.close();
        }
    }
    
    btnStartRecord.disabled = false;
    btnStopRecord.disabled = true;
    recordPulse.classList.remove("recording");
    clearInterval(recordTimerInterval);
    recordTimer.textContent = "00:00";
    
    if (recLength > 0) {
        // Flatten buffers
        const resultBuffer = new Float32Array(recLength);
        let offset = 0;
        for (let i = 0; i < recBuffers.length; i++) {
            resultBuffer.set(recBuffers[i], offset);
            offset += recBuffers[i].length;
        }
        
        // Encode to WAV
        const wavBuffer = bufferToWav(resultBuffer, 16000);
        const audioBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        const randomId = Math.floor(Math.random() * 10000);
        const recordedFile = new File([audioBlob], `recorded_speech_${randomId}.wav`, { type: 'audio/wav' });
        handleFileSelect(recordedFile);
        log("Audio recording captured successfully.", "success");
    }
}

function stopRecordingIfActive() {
    stopRecording();
}

// =============================================================================
// PIPELINE EXECUTION: FULL END-TO-END
// =============================================================================
async function executeFullPipeline() {
    if (!appState.audioFile) return;

    resetPipelineUI();
    log("=== RUNNING FULL DIAGNOSTIC PIPELINE ===", "block");
    btnRunFullPipeline.disabled = true;
    btnRunStepByStep.disabled = true;

    // Pulse all cards
    const stepCards = document.querySelectorAll(".step-card");
    stepCards.forEach(card => card.classList.add("active"));

    const formData = new FormData();
    formData.append("file", appState.audioFile);
    formData.append("patient_id", patientIdInput.value || "Anonymous");

    try {
        const response = await fetch(`${API_BASE}/api/pipeline`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error(await response.text());

        const result = await response.json();
        
        // Success animation
        stepCards.forEach(card => {
            card.classList.remove("active");
            card.classList.add("completed");
        });

        // Log everything
        result.logs.forEach(line => {
            if (line.includes("[ERROR]")) log(line, "error");
            else if (line.includes("Started") || line.includes("Regression")) log(line, "block");
            else log(line, "success");
        });

        if (result.simulation_mode) {
            log("[WARNING] Server executed with dynamic simulations due to resource bounds.", "error");
        }

        // Display results
        displayResults(result);
        loadHistory();

    } catch (err) {
        log(`Pipeline Execution Failed: ${err.message}`, "error");
        stepCards.forEach(card => card.classList.remove("active"));
    } finally {
        btnRunFullPipeline.disabled = false;
        btnRunStepByStep.disabled = false;
    }
}

// =============================================================================
// PIPELINE EXECUTION: STEP-BY-STEP
// =============================================================================
async function executeStepByStep() {
    if (!appState.audioFile) return;

    btnRunFullPipeline.disabled = true;
    btnRunStepByStep.disabled = true;

    if (appState.currentExecutingStep === 0) {
        resetPipelineUI();
        log("=== STARTING STEP-BY-STEP ASSESSMENT ===", "block");
    }

    appState.currentExecutingStep++;
    const stepNum = appState.currentExecutingStep;
    const card = document.getElementById(getCardId(stepNum));
    
    if (card) {
        card.classList.add("active");
    }

    try {
        if (stepNum === 1) {
            // STEP 1: Upload & Register MongoDB
            log("Step 1: Contacting Data Storage & Management layer...", "system");
            const formData = new FormData();
            formData.append("file", appState.audioFile);
            formData.append("patient_id", patientIdInput.value || "Anonymous");

            const res = await fetch(`${API_BASE}/api/upload`, {
                method: "POST",
                body: formData
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            appState.sessionId = data.session_id;
            appState.audioPath = data.audio_path;
            
            log(`Session registered. ID: ${data.session_id}. Saved in secure file path: ${data.audio_path}`, "success");
            log(`Database state: ${data.db_info.db_type.toUpperCase()} active.`, "success");
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Run Step 2 (Preprocess)";
            btnRunStepByStep.disabled = false;

        } else if (stepNum === 2) {
            // STEP 2: Preprocess & Segment
            log("Step 2: Performing Normalization & fixed segmentation...", "system");
            const res = await fetch(`${API_BASE}/api/preprocess`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ audio_path: appState.audioPath, patient_id: patientIdInput.value })
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            appState.segmentsCount = data.segments_count;
            appState.duration = data.duration;
            
            log(`Preprocessing complete. Audio Duration: ${data.duration.toFixed(2)}s. Created ${data.segments_count} segments of size 3s.`, "success");
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Run Step 3 (Features)";
            btnRunStepByStep.disabled = false;

        } else if (stepNum === 3) {
            // STEP 3: Feature Extraction & Base SSI
            log("Step 3: Extracting 4D Articulatory Feature vectors & calculating SSI...", "system");
            const res = await fetch(`${API_BASE}/api/extract`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: appState.sessionId, audio_path: appState.audioPath, segments_count: appState.segmentsCount })
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            appState.stepResults.extracted_features = data.segment_features;
            appState.stepResults.ssi_base = data.ssi_base_score;
            
            log(`Features extracted. Avg Pause: ${data.features_summary.avg_pause_raw.toFixed(3)}, Avg Pitch Var: ${data.features_summary.avg_pitch_var.toFixed(1)}Hz.`, "success");
            log(`Computed base SSI score: ${data.ssi_base_score.toFixed(4)}`, "success");
            
            // Update radar chart with first segment features
            updateRadarChart(data.segment_features[0]);
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Run Step 4 (Latent representations)";
            btnRunStepByStep.disabled = false;

        } else if (stepNum === 4) {
            // STEP 4: Latent Learning
            log("Step 4: Feeding features to VQ-VAE & SDVAE. Fetching WavLM representations...", "system");
            const res = await fetch(`${API_BASE}/api/latent`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: appState.sessionId, features: appState.stepResults.extracted_features })
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            log(`VQ-VAE compressed states generated (size: 128D).`, "success");
            log(`SDVAE latent distribution generated (size: 128D).`, "success");
            log(`Contextual WavLM embeddings generated (dimensions: ${data.wavlm_shape.join("x")}).`, "success");
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Run Step 5 (Attention Fusion)";
            btnRunStepByStep.disabled = false;

        } else if (stepNum === 5) {
            // STEP 5: Attention Fusion
            log("Step 5: Fusing VQ-VAE/SDVAE outputs with WavLM sequence using Multi-Head Attention...", "system");
            const res = await fetch(`${API_BASE}/api/fuse`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: appState.sessionId, latent_reps: [] })
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            log(`Multi-Head Attention fused outputs across segments. Vector generated.`, "success");
            
            // Update Attention weights graph
            if (data.attention_sample && data.attention_sample.length) {
                // Flatten if multidimensional list
                const flatWeights = Array.isArray(data.attention_sample[0]) ? data.attention_sample[0][0] : data.attention_sample[0];
                updateAttentionChart(flatWeights);
            }
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Run Step 6 (MLP Regress)";
            btnRunStepByStep.disabled = false;

        } else if (stepNum === 6) {
            // STEP 6: Predict & Store
            log("Step 6: Feeding fused representations into MLP regression head...", "system");
            const res = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: appState.sessionId, fused_features: {} })
            });
            if (!res.ok) throw new Error(await res.text());
            
            const data = await res.json();
            log(`MLP Regression completed. SSI Severity predicted!`, "success");
            log(`Saved results into MongoDB. Evaluation completed.`, "success");
            
            displayResults(data);
            loadHistory();
            
            markStepComplete(card);
            btnRunStepByStep.textContent = "Pipeline Complete";
            btnRunStepByStep.disabled = true;
            btnRunFullPipeline.disabled = false;
        }

    } catch (err) {
        log(`Step ${stepNum} failed: ${err.message}`, "error");
        if (card) card.classList.remove("active");
        btnRunFullPipeline.disabled = false;
        btnRunStepByStep.disabled = false;
        appState.currentExecutingStep = 0; // reset
    }
}

function markStepComplete(card) {
    if (card) {
        card.classList.remove("active");
        card.classList.add("completed");
    }
}

function getCardId(step) {
    const ids = {
        1: "stepUpload",
        2: "stepPreprocess",
        3: "stepExtract",
        4: "stepLatent",
        5: "stepFuse",
        6: "stepPredict"
    };
    return ids[step];
}

// =============================================================================
// RESULTS DISPLAY PANEL
// =============================================================================
function displayResults(data) {
    ssiValue.textContent = data.ssi_score.toFixed(3);
    panssValue.textContent = Math.round(data.panss_score);
    
    // Set badge style classes
    severityBadge.className = "severity-badge";
    let desc = "";
    
    if (data.severity.includes("Healthy") || data.severity.includes("Normal")) {
        severityBadge.classList.add("healthy");
        severityBadge.textContent = "Healthy Control";
        desc = "Speech features match healthy reference controls. No evidence of severity indicators.";
    } else if (data.severity.includes("Mild")) {
        severityBadge.classList.add("mild");
        severityBadge.textContent = "Mild Severity";
        desc = "Slight articulation decreases and subtle pause patterns detected. Recommend monitoring.";
    } else if (data.severity.includes("Moderate")) {
        severityBadge.classList.add("moderate");
        severityBadge.textContent = "Moderate Severity";
        desc = "Prolonged pauses, high pitch variations, and flat speech pattern indicators detected.";
    } else {
        severityBadge.classList.add("severe");
        severityBadge.textContent = "Severe Severity";
        desc = "Significant clinical markers present. High silence ratios and severe vocal flatting detected.";
    }
    
    severityDesc.textContent = desc;
}

// =============================================================================
// CHARTS UPDATE LOGIC
// =============================================================================
function updateRadarChart(f) {
    // Normalise features manually for visualization
    const pause = Math.min(Math.max(f.pause_raw, 0), 1);
    const pitch = Math.tanh(f.pitch_var / 50.0);
    const zcr = Math.tanh(f.zcr * 5.0);
    const artic = Math.min(Math.max(f.articulation_raw, 0), 1);

    featuresChart.data.datasets[0].data = [pause, pitch, zcr, artic];
    featuresChart.update();
}

function updateAttentionChart(weights) {
    if (!weights || !weights.length) return;
    
    // Truncate/pad to 30 elements for clean visual
    let visualWeights = [...weights];
    if (visualWeights.length > 30) {
        // Average chunking
        const chunkSize = Math.ceil(visualWeights.length / 30);
        visualWeights = [];
        for (let i = 0; i < weights.length; i += chunkSize) {
            const chunk = weights.slice(i, i + chunkSize);
            visualWeights.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
        }
    }
    
    // Pad if shorter
    while (visualWeights.length < 30) {
        visualWeights.push(0);
    }
    
    // Normalise weights to sum to 1 visually
    const sum = visualWeights.reduce((a, b) => a + b, 0) || 1;
    const normWeights = visualWeights.map(w => w / sum);

    attentionChart.data.datasets[0].data = normWeights;
    attentionChart.update();
}

// =============================================================================
// HISTORICAL RECORDS RETRIEVAL
// =============================================================================
async function loadHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        if (!res.ok) return;
        const data = await res.json();
        renderHistoryTable(data.records);
    } catch (err) {
        console.error("History fetch error:", err);
    }
}

function renderHistoryTable(records) {
    if (!records || records.length === 0) {
        historyBody.innerHTML = `
            <tr>
                <td colspan="7" class="empty-table">No history found. Complete an evaluation to view records.</td>
            </tr>
        `;
        return;
    }

    historyBody.innerHTML = "";
    records.forEach(r => {
        const tr = document.createElement("tr");
        
        // Date formatting
        const dateStr = new Date(r.timestamp).toLocaleString();
        
        // Features extraction
        const featText = r.features ? 
            `ZCR: ${r.features.zcr ? r.features.zcr.toFixed(2) : '0.00'} | Pause: ${r.features.pause_raw ? r.features.pause_raw.toFixed(2) : '0.00'}` : 
            "No details";

        tr.innerHTML = `
            <td><strong>${r.patient_id}</strong></td>
            <td>${dateStr}</td>
            <td><code style="font-size:0.75rem; color:#9ca3af;">${featText}</code></td>
            <td><strong>${r.ssi_score.toFixed(3)}</strong></td>
            <td><span style="color:#a78bfa; font-weight:600;">${Math.round(r.prediction_score)}</span></td>
            <td><span class="severity-badge-sm ${getSeverityClass(r.severity)}">${r.severity}</span></td>
            <td><button class="btn-history-log" onclick="showRecordLog('${r.id}')"><i class="fa-solid fa-file-invoice"></i> Logs</button></td>
        `;
        historyBody.appendChild(tr);
    });
}

function getSeverityClass(sev) {
    if (sev.includes("Healthy") || sev.includes("Normal")) return "healthy-text";
    if (sev.includes("Mild")) return "mild-text";
    if (sev.includes("Moderate")) return "moderate-text";
    return "severe-text";
}

// Global modal log inspector hook
window.showRecordLog = async function(recordId) {
    try {
        const res = await fetch(`${API_BASE}/api/history`);
        if (!res.ok) return;
        const data = await res.json();
        const record = data.records.find(r => r.id === recordId);
        
        if (record) {
            modalLogContent.innerHTML = "";
            record.logs.forEach(l => {
                const line = document.createElement("div");
                line.className = "log-line";
                if (l.includes("[ERROR]")) line.classList.add("error");
                else if (l.includes("Started")) line.classList.add("block");
                else line.classList.add("success");
                line.textContent = l;
                modalLogContent.appendChild(line);
            });
            logModal.classList.add("active");
        }
    } catch (err) {
        console.error("Modal logs load error:", err);
    }
};

// =============================================================================
// DATABASE INTEGRATION HEALTHCHECK
// =============================================================================
async function checkDatabaseStatus() {
    try {
        const dbRes = await fetch(`${API_BASE}/api/history`);
        if (dbRes.ok) {
            dbStatus.style.background = "rgba(16, 185, 129, 0.1)";
            dbStatus.style.borderColor = "rgba(16, 185, 129, 0.3)";
            dbStatus.style.color = "#a7f3d0";
            dbName.textContent = "MONGO / SQLITE ACTIVE";
        }
    } catch (err) {
        dbName.textContent = "DISCONNECTED";
        dbStatus.style.background = "rgba(239, 68, 68, 0.1)";
        dbStatus.style.borderColor = "rgba(239, 68, 68, 0.3)";
        dbStatus.style.color = "#fecaca";
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
function log(msg, type = "info") {
    const div = document.createElement("div");
    div.className = `log-line ${type}`;
    div.textContent = msg;
    terminalLogs.appendChild(div);
    terminalLogs.scrollTop = terminalLogs.scrollHeight;
}

function resetPipelineUI() {
    const stepCards = document.querySelectorAll(".step-card");
    stepCards.forEach(card => {
        card.className = "step-card";
    });
    terminalLogs.innerHTML = "";
}

function resetState() {
    appState = {
        audioFile: null,
        audioPath: null,
        sessionId: null,
        duration: 0,
        segmentsCount: 0,
        stepResults: {},
        currentExecutingStep: 0,
        dbInfo: null
    };

    resetPipelineUI();
    wavesurfer.destroy();
    initWaveSurfer();
    
    // Clear graphs
    featuresChart.data.datasets[0].data = [0, 0, 0, 0];
    featuresChart.update();
    attentionChart.data.datasets[0].data = Array(30).fill(0);
    attentionChart.update();

    ssiValue.textContent = "0.00";
    panssValue.textContent = "--";
    severityBadge.className = "severity-badge";
    severityBadge.textContent = "Awaiting Input";
    severityDesc.textContent = "Submit a voice recording to estimate schizophrenia severity metrics.";
    
    fileNameLabel.textContent = "No file selected";
    audioFileInput.value = "";
    btnRunFullPipeline.disabled = true;
    btnRunStepByStep.disabled = true;
    btnRunStepByStep.textContent = "Run Step-by-Step";
    btnReset.style.display = "none";
    log("System reset complete. Awaiting new speech data recording...", "system");
}

function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

function pad(val) {
    return val.toString().padStart(2, "0");
}
