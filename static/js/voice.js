(() => {
    'use strict';

    const $ = (id) => document.getElementById(id);

    const modeMicBtn = $('modeMic');
    const modeUploadBtn = $('modeUpload');
    const micSection = $('micSection');
    const uploadSection = $('uploadSection');

    const recordBtn = $('recordBtn');
    const stopRecordBtn = $('stopRecordBtn');
    const recTimer = $('recTimer');
    const previewPlayer = $('previewPlayer');

    const fileInput = $('fileInput');
    const filePreviewPlayer = $('filePreviewPlayer');

    const patientIdInput = $('patientId');
    const exerciseSelect = $('exercise');
    const durationInput = $('duration');
    const exerciseWarning = $('exerciseWarning');

    const processBtn = $('processBtn');
    const statusDiv = $('status');

    const resultsBlock = $('resultsBlock');
    const waveformCanvas = $('waveformCanvas');
    const spectrogramImg = $('spectrogramImg');
    const featJitter = $('feat-jitter');
    const featShimmer = $('feat-shimmer');
    const featHnr = $('feat-hnr');
    const diagnosisLabel = $('diagnosisLabel');
    const diagnosisProb = $('diagnosisProb');
    const diagnosisMeta = $('diagnosisMeta');
    const savedInfo = $('savedInfo');

    let currentMode = 'mic';
    let mediaRecorder = null;
    let mediaStream = null;
    let recordedChunks = [];
    let recordingTimer = null;
    let recordingStartTs = 0;
    let recordingLimitSec = 0;
    let pendingBlob = null;

    modeMicBtn.addEventListener('click', () => setMode('mic'));
    modeUploadBtn.addEventListener('click', () => setMode('upload'));

    function setMode(mode) {
        currentMode = mode;
        modeMicBtn.classList.toggle('active', mode === 'mic');
        modeUploadBtn.classList.toggle('active', mode === 'upload');
        micSection.style.display = mode === 'mic' ? '' : 'none';
        uploadSection.style.display = mode === 'upload' ? '' : 'none';
        pendingBlob = null;
        updateProcessBtn();
    }

    exerciseSelect.addEventListener('change', syncExerciseMeta);

    function syncExerciseMeta() {
        const opt = exerciseSelect.options[exerciseSelect.selectedIndex];
        if (!opt) return;
        const isBeta = opt.dataset.beta === 'true';
        exerciseWarning.style.display = isBeta ? '' : 'none';
        const recommended = parseInt(opt.dataset.duration || '0', 10);
        if (recommended > 0) {
            durationInput.value = recommended;
        }
    }
    syncExerciseMeta();

    recordBtn.addEventListener('click', startRecording);
    stopRecordBtn.addEventListener('click', stopRecording);

    async function startRecording() {
        if (!navigator.mediaDevices?.getUserMedia) {
            setStatus('Браузер не поддерживает запись с микрофона', 'error');
            return;
        }
        recordingLimitSec = Math.max(2, parseInt(durationInput.value, 10) || 5);

        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
            });
        } catch (e) {
            setStatus('Не удалось получить доступ к микрофону: ' + e.message, 'error');
            return;
        }

        const mime = pickMime();
        try {
            mediaRecorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : {});
        } catch (e) {
            setStatus('MediaRecorder ошибка: ' + e.message, 'error');
            stopStream();
            return;
        }

        recordedChunks = [];
        mediaRecorder.ondataavailable = (ev) => {
            if (ev.data && ev.data.size > 0) recordedChunks.push(ev.data);
        };
        mediaRecorder.onstop = onRecordingStopped;

        mediaRecorder.start();
        recordingStartTs = Date.now();

        recordBtn.disabled = true;
        stopRecordBtn.disabled = false;
        setStatus('', '');
        tickRecording();
        recordingTimer = setInterval(tickRecording, 200);
    }

    function pickMime() {
        const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
        for (const m of candidates) {
            if (MediaRecorder.isTypeSupported(m)) return m;
        }
        return null;
    }

    function tickRecording() {
        const elapsed = (Date.now() - recordingStartTs) / 1000;
        const remaining = Math.max(0, recordingLimitSec - elapsed);
        recTimer.textContent = `⏱ ${remaining.toFixed(1)} с осталось`;
        if (remaining <= 0) stopRecording();
    }

    function stopRecording() {
        if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
        mediaRecorder.stop();
    }

    function onRecordingStopped() {
        clearInterval(recordingTimer);
        recordingTimer = null;
        recTimer.textContent = '';
        recordBtn.disabled = false;
        stopRecordBtn.disabled = true;
        stopStream();

        if (recordedChunks.length === 0) {
            setStatus('Ничего не записалось', 'error');
            return;
        }
        const mime = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(recordedChunks, { type: mime });
        pendingBlob = { blob, filename: `recording.${extFromMime(mime)}` };

        previewPlayer.src = URL.createObjectURL(blob);
        previewPlayer.style.display = '';
        setStatus(`Запись готова (${(blob.size / 1024).toFixed(1)} KB). Нажмите «Обработать запись».`, 'ok');
        updateProcessBtn();
    }

    function extFromMime(mime) {
        if (mime.includes('webm')) return 'webm';
        if (mime.includes('ogg')) return 'ogg';
        if (mime.includes('mp4') || mime.includes('m4a')) return 'm4a';
        return 'webm';
    }

    function stopStream() {
        if (mediaStream) {
            mediaStream.getTracks().forEach((t) => t.stop());
            mediaStream = null;
        }
    }

    fileInput.addEventListener('change', () => {
        const f = fileInput.files[0];
        if (!f) return;
        pendingBlob = { blob: f, filename: f.name };
        filePreviewPlayer.src = URL.createObjectURL(f);
        filePreviewPlayer.style.display = '';
        setStatus(`Выбран файл: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`, 'ok');
        updateProcessBtn();
    });

    function updateProcessBtn() {
        processBtn.disabled = !pendingBlob;
    }

    processBtn.addEventListener('click', processRecording);

    async function processRecording() {
        const patientId = patientIdInput.value.trim();
        if (!patientId) {
            setStatus('Укажите номер пациента', 'error');
            return;
        }
        if (!pendingBlob) {
            setStatus('Сначала запишите или загрузите аудио', 'error');
            return;
        }
        const exercise = exerciseSelect.value;
        const duration = parseInt(durationInput.value, 10) || null;

        processBtn.disabled = true;
        setStatus('Загрузка аудио…', 'progress');

        try {
            const fd = new FormData();
            fd.append('file', pendingBlob.blob, pendingBlob.filename);
            const resp = await fetch('/voice/upload', { method: 'POST', body: fd });
            if (!resp.ok) {
                throw new Error(`Upload failed: ${resp.status} ${await resp.text()}`);
            }
        } catch (e) {
            setStatus('Ошибка загрузки: ' + e.message, 'error');
            processBtn.disabled = false;
            return;
        }

        setStatus('Обработка записи (препроцессинг + признаки + модель)…', 'progress');
        try {
            const resp = await fetch('/voice/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ patient_id: patientId, exercise, duration }),
            });
            if (!resp.ok) {
                const t = await resp.text();
                throw new Error(`${resp.status} ${t}`);
            }
            const data = await resp.json();
            renderResults(data);
            setStatus('Готово', 'ok');
        } catch (e) {
            setStatus('Ошибка обработки: ' + e.message, 'error');
        } finally {
            processBtn.disabled = false;
        }
    }

    function renderResults(data) {
        resultsBlock.style.display = '';

        drawWaveform(data.audio?.waveform_downsampled || []);

        spectrogramImg.src = data.spectrogram_url + '?t=' + Date.now();

        featJitter.textContent = fmtFeature(data.features?.jitter);
        featShimmer.textContent = fmtFeature(data.features?.shimmer);
        featHnr.textContent = fmtFeature(data.features?.hnr);

        const pred = data.prediction || {};
        const isSick = pred.class === 1;
        diagnosisLabel.textContent = pred.label_ru || (isSick ? 'болен' : 'здоров');
        diagnosisLabel.classList.toggle('healthy', !isSick);
        diagnosisLabel.classList.toggle('sick', isSick);

        if (typeof pred.probability === 'number') {
            diagnosisProb.textContent = `Вероятность PD: ${(pred.probability * 100).toFixed(1)}%`;
        } else {
            diagnosisProb.textContent = '';
        }
        diagnosisMeta.textContent = `Порог: ${pred.threshold ?? 0.5}`;

        if (data.saved_to) {
            savedInfo.textContent = 'Результат сохранён: ' + data.saved_to;
        }

        const note = document.getElementById('diagnosisExerciseNote');
        if (note) {
            const supported = data.exercise_supported_by_model;
            const ex = data.exercise || '—';
            if (supported) {
                note.textContent = `Упражнение «${ex}» — модель обучена на нём.`;
            } else {
                note.textContent = `Упражнение «${ex}» — модель обучена только на «a». Предсказание может быть неточным.`;
            }
        }
    }

    function fmtFeature(f) {
        if (!f || f.value == null) return '—';
        return `${f.value.toFixed(3)} ${f.unit || ''}`.trim();
    }

    function drawWaveform(samples) {
        const ctx = waveformCanvas.getContext('2d');
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = '#dfe6ed';
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        if (!samples.length) return;

        const step = w / samples.length;
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < samples.length; i++) {
            const x = i * step;
            const y = h / 2 - samples[i] * (h / 2 - 4);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    function setStatus(msg, kind) {
        statusDiv.textContent = msg;
        statusDiv.style.color = {
            error: '#c0392b',
            ok: '#27ae60',
            progress: '#2980b9',
        }[kind] || '#333';
    }
})();