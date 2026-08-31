// static/js/mimic.js

// Индикатор "идёт работа" для кнопок обработки: обычный POST-запрос без
// промежуточного прогресса от сервера, поэтому честный процент не показать —
// но пульсация + секундомер явно показывают, что приложение не зависло.
(function injectProcessingSpinnerStyle() {
    const style = document.createElement("style");
    style.textContent = `
        @keyframes btn-processing-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.55; }
        }
        .btn-processing {
            animation: btn-processing-pulse 1.1s ease-in-out infinite;
            cursor: progress !important;
        }
    `;
    document.head.appendChild(style);
})();
let processingTimerId = null;

let ws;
let stream;
let isCameraOn = false;
let isRecording = false;
let neutralRecorded = false;
let timerInterval;
let seconds = 0;

// Элементы UI
let videoElement, canvas, ctx, videoFeed, placeholder, statusText, timerDisplay;
let camBtn, recBtn, procBtn;
let maskToggle, exerciseSelect, exerciseWarning;
let meshCanvas, analysisVizSection, resultsContainer, resultsContent;
let resultImageObj = new Image();

// Элементы загрузки
let btnUploadAction, btnProcessUpload, uploadStatus;

// === ГРУППЫ ТОЧЕК ДЛЯ ОТРИСОВКИ (СИНХРОНИЗИРОВАНО С PYTHON) ===
const MUSCLE_GROUPS = {
    smile: [
        // Smile 1 (Большая скуловая)
        [61, 186, 216, 207, 187, 123],
        [308, 410, 436, 427, 411, 352],
        // Smile 2 (Малая скуловая)
        [50, 203, 165, 39],
        [280, 423, 391, 269],
        // Smile 3 (Поднимающая верхнюю губу - ДОБАВЛЕНО)
        [50, 205, 206, 92, 40],
        [280, 425, 426, 322, 270]
    ],
    frown: [
        [55, 65, 52, 53, 46],
        [285, 295, 282, 283, 276]
    ],
    brows_up: [
        [107, 66, 105, 63, 70],
        [336, 296, 334, 293, 300]
    ],
    blink: [
        [159, 145, 133, 33],
        [386, 374, 362, 263]
    ],
    neutral: [
        [0, 17, 61, 291]
    ]
};

document.addEventListener("DOMContentLoaded", function() {
    videoElement = document.createElement('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    videoFeed = document.getElementById('videoFeed');
    placeholder = document.getElementById('videoPlaceholder');
    statusText = document.getElementById('statusText');
    timerDisplay = document.getElementById('timerDisplay');

    // Кнопки камеры
    camBtn = document.getElementById('cameraBtn');
    recBtn = document.getElementById('recordBtn');
    procBtn = document.getElementById('processBtn');

    // Кнопки загрузки
    btnUploadAction = document.getElementById('btnUploadAction');
    btnProcessUpload = document.getElementById('btnProcessUpload');
    uploadStatus = document.getElementById('uploadStatus');

    // Общие элементы результатов
    resultsContainer = document.getElementById('resultsContainer');
    resultsContent = document.getElementById('resultsContent');
    meshCanvas = document.getElementById('meshVisualizationCanvas');
    analysisVizSection = document.getElementById('analysisVizSection');

    maskToggle = document.getElementById('maskToggle');
    exerciseSelect = document.getElementById('exerciseSelect');
    exerciseWarning = document.getElementById('exerciseWarning');

    // Переместим блок результатов в секцию камеры по умолчанию
    const camSection = document.getElementById('cameraSection');
    const sharedWrapper = document.getElementById('sharedResultsWrapper');
    if(camSection && sharedWrapper) {
        camSection.appendChild(sharedWrapper);
    }
});

// === ОТРИСОВКА РЕЗУЛЬТАТА (ПАУТИНА) ===
function drawStaticResult(imgBase64, landmarks, exerciseType) {
    if (!meshCanvas) return;
    const mCtx = meshCanvas.getContext('2d');
    const w = meshCanvas.width;
    const h = meshCanvas.height;

    resultImageObj.onload = function() {
        mCtx.clearRect(0,0,w,h);
        mCtx.drawImage(resultImageObj, 0, 0, w, h);

        if (landmarks && landmarks.length > 0) {
            const scaleX = w / resultImageObj.naturalWidth;
            const scaleY = h / resultImageObj.naturalHeight;
            const scaled = landmarks.map(pt => [pt[0] * scaleX, pt[1] * scaleY]);
            drawCombinationsOnContext(mCtx, scaled, exerciseType);
        }
    };
    resultImageObj.src = imgBase64;
}

// Рисует ВСЕ линии комбинаций, соответствующие расчетам
function drawCombinationsOnContext(ctx, landmarks, type) {
    const groups = MUSCLE_GROUPS[type] || MUSCLE_GROUPS['neutral'];
    ctx.lineWidth = 1.0;

    const getPt = (idx) => {
        if (!landmarks[idx]) return {x:0, y:0};
        return { x: landmarks[idx][0], y: landmarks[idx][1] };
    };

    // Нормировочная база (глаза)
    if(landmarks[468] && landmarks[473]) {
         const pL = getPt(468);
         const pR = getPt(473);
         ctx.strokeStyle = "cyan";
         ctx.beginPath(); ctx.moveTo(pL.x, pL.y); ctx.lineTo(pR.x, pR.y); ctx.stroke();
    }

    groups.forEach(groupIndices => {
        // Точки
        groupIndices.forEach(idx => {
            const p = getPt(idx);
            ctx.fillStyle = "yellow";
            ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI*2); ctx.fill();
        });

        // ЛИНИИ (Combinations) - полный перебор пар
        ctx.strokeStyle = "rgba(0, 255, 0, 0.6)"; // Полупрозрачный зеленый

        for (let i = 0; i < groupIndices.length; i++) {
            for (let j = i + 1; j < groupIndices.length; j++) {
                const p1 = getPt(groupIndices[i]);
                const p2 = getPt(groupIndices[j]);

                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        }
    });
}

// === ВЫВОД РЕЗУЛЬТАТОВ ===
let distanceChartInstance = null;

function hideAllResultBlocks() {
    ['neutralReadyBlock', 'predictionBlock', 'chartBlock', 'qualityBadge', 'shapBlock', 'fusionBlock'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

let shapChartInstance = null;

function renderShap(explanation) {
    const block = document.getElementById('shapBlock');
    if (!block || !explanation) return;

    // bars: PD features (red, positive), Healthy features (green, negative)
    const pd = explanation.pushing_pd || [];
    const hh = explanation.pushing_healthy || [];
    // Собираем в единый список, сортируем по shap (от -max_healthy до +max_pd) для горизонтального графика
    const combined = [...hh, ...pd].sort((a, b) => a.shap - b.shap);
    const labels = combined.map(x => x.feature);
    const values = combined.map(x => x.shap);
    const colors = combined.map(x => x.shap > 0 ? '#e57373' : '#81c784');

    document.getElementById('shapChart').style.height = (combined.length * 55 + 60) + 'px';
    const ctx = document.getElementById('shapChart').getContext('2d');
    if (shapChartInstance) shapChartInstance.destroy();
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{
            label: 'SHAP вклад', data: values,
            backgroundColor: colors, borderColor: colors,
        }]},
        options: {
            indexAxis: 'y',
            responsive: true,
            scales: {
                x: { title: { display: true, text: '← Здоров    |    БП →' } },
                y: { ticks: { font: { size: 10 }, crossAlign: 'far' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // Детали по каждой топ-фиче: твоё значение vs обучающие распределения
    const box = document.getElementById('shapFeatureDetails');
    let html = '<table style="width:100%; border-collapse: collapse; font-size: 0.9em;">';
    html += '<thead style="background:#f1f1f1;"><tr>' +
            '<th style="padding:6px; border:1px solid #ddd; text-align:left;">Признак</th>' +
            '<th style="padding:6px; border:1px solid #ddd;">Значение</th>' +
            '<th style="padding:6px; border:1px solid #ddd;">SHAP</th>' +
            '</tr></thead><tbody>';
    combined.reverse().forEach(x => {
        const bg = x.shap > 0 ? '#ffebee' : '#e8f5e9';
        html += `<tr style="background:${bg};">
            <td style="padding:6px; border:1px solid #ddd;">${x.feature_label || x.feature}</td>
            <td style="padding:6px; border:1px solid #ddd; text-align:center; font-weight:bold;">${x.value.toFixed(3)}</td>
            <td style="padding:6px; border:1px solid #ddd; text-align:center;">${x.shap >= 0 ? '+' : ''}${x.shap.toFixed(3)}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    box.innerHTML = html;
    block.style.display = 'block';
}

function renderQualityBadge(quality) {
    const el = document.getElementById('qualityBadge');
    if (!el || !quality) return;
    const palette = {
        ok:        { bg: '#e8f5e9', border: '#66bb6a', color: '#1b5e20', icon: '✓' },
        warn:      { bg: '#fff8e1', border: '#ffb300', color: '#6d4c00', icon: '⚠' },
        bad:       { bg: '#ffebee', border: '#e57373', color: '#b71c1c', icon: '✕' },
        too_short: { bg: '#ffebee', border: '#e57373', color: '#b71c1c', icon: '✕' },
    };
    const p = palette[quality.level] || palette.ok;
    el.style.background = p.bg;
    el.style.borderLeft = '4px solid ' + p.border;
    el.style.color = p.color;
    el.innerHTML = `<strong>${p.icon} Качество записи:</strong> ${quality.message}`;
    el.style.display = 'block';
}

function renderConfidenceBadge(prediction) {
    const el = document.getElementById('confidenceBadge');
    if (!el || !prediction) return;
    const levels = {
        high:     { text: 'Уверенное предсказание',        color: '#1b5e20' },
        moderate: { text: 'Умеренная уверенность',          color: '#6d4c00' },
        low:      { text: 'На границе — результат неустойчив', color: '#b71c1c' },
    };
    const lvl = levels[prediction.confidence_level] || levels.moderate;
    const pct = (prediction.max_proba * 100).toFixed(1);
    el.style.color = lvl.color;
    el.innerHTML = `<em>${lvl.text} (max = ${pct}%)</em>`;
}

function renderNeutralReady(data, exerciseName) {
    resultsContainer.style.display = 'block';
    hideAllResultBlocks();
    document.getElementById('resultsSubtitle').innerText = 'Нейтраль готова';
    document.getElementById('neutralReadyBlock').style.display = 'block';
    renderQualityBadge(data.quality);

    if (data.best_frame) {
        analysisVizSection.style.display = 'block';
        drawStaticResult(data.best_frame, data.best_landmarks, exerciseName);
    }
}

function renderSmilePrediction(data, exerciseName) {
    resultsContainer.style.display = 'block';
    hideAllResultBlocks();
    document.getElementById('resultsSubtitle').innerText = 'Классификация выполнена';
    renderQualityBadge(data.quality);

    if (data.best_frame) {
        analysisVizSection.style.display = 'block';
        drawStaticResult(data.best_frame, data.best_landmarks, exerciseName);
    }

    const p = data.prediction;
    renderConfidenceBadge(p);
    document.getElementById('probHealthyVal').innerText = (p.proba_healthy * 100).toFixed(1) + '%';
    document.getElementById('probPdVal').innerText = (p.proba_pd * 100).toFixed(1) + '%';
    const lblBox = document.getElementById('predictionLabel');
    if (p.label === 'PD') {
        lblBox.innerText = "Предсказание: Болезнь Паркинсона";
        lblBox.style.background = '#ffebee';
        lblBox.style.color = '#b71c1c';
    } else {
        lblBox.innerText = 'Предсказание: Здоров';
        lblBox.style.background = '#e8f5e9';
        lblBox.style.color = '#1b5e20';
    }
    document.getElementById('predictionBlock').style.display = 'block';

    if (data.chart && data.chart.times && data.chart.series) {
        const times = data.chart.times;
        const seriesMap = data.chart.series;
        const colors = {
            // smile
            'dist_308_411': '#e74c3c',
            'dist_308_427': '#3498db',
            'dist_61_187':  '#2ecc71',
            'dist_61_207':  '#f39c12',
            // brows_up
            'dist_285_168': '#e74c3c',
            'dist_336_168': '#3498db',
            'dist_107_168': '#2ecc71',
            'dist_55_168':  '#f39c12',
            'dist_296_10':  '#9b59b6',
            'dist_66_10':   '#1abc9c',
            // frown
            'dist_285_6':   '#e74c3c',
            'dist_55_6':    '#3498db',
            'dist_52_168':  '#2ecc71',
            'dist_282_168': '#f39c12',
            'dist_295_4':   '#9b59b6',
        };
        const datasets = Object.keys(seriesMap).map(k => ({
            label: k,
            data: seriesMap[k],
            borderColor: colors[k] || '#666',
            backgroundColor: colors[k] || '#666',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2,
        }));
        const ctx = document.getElementById('distanceChart').getContext('2d');
        if (distanceChartInstance) { distanceChartInstance.destroy(); }
        distanceChartInstance = new Chart(ctx, {
            type: 'line',
            data: { labels: times.map(t => t.toFixed(2)), datasets: datasets },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { title: { display: true, text: 'Время (с)' }, ticks: { maxTicksLimit: 10 } },
                    y: { title: { display: true, text: 'Дистанция / нейтраль' } }
                },
                plugins: { legend: { position: 'top' } }
            }
        });
        document.getElementById('chartBlock').style.display = 'block';
    }

    if (data.explanation) {
        renderShap(data.explanation);
    }

    // Показываем блок fusion после любого упражнения с предсказанием
    const fb = document.getElementById('fusionBlock');
    if (fb) fb.style.display = 'block';
    // Всегда сбрасываем — не показываем результаты от предыдущей сессии
    document.getElementById('fusionResult').innerHTML =
        '<span style="color:#888; font-size:0.9em;">Обработайте улыбку, брови и нахмуривание, затем нажмите кнопку.</span>';
}

function renderResults(data, exerciseName) {
    if (data.kind === 'neutral_ready') {
        renderNeutralReady(data, exerciseName);
    } else if (data.kind && data.kind.endsWith('_prediction')) {
        renderSmilePrediction(data, exerciseName);
    } else {
        resultsContainer.style.display = 'block';
        hideAllResultBlocks();
    }
}

// === УПРАВЛЕНИЕ ЗАГРУЗКОЙ (НОВАЯ ФУНКЦИЯ) ===
async function uploadVideoFile() {
    const fileInput = document.getElementById('uploadFileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert("Пожалуйста, выберите файл!");
        return;
    }

    const exercise = document.getElementById('uploadExerciseSelect').value;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("exercise", exercise);

    btnUploadAction.disabled = true;
    uploadStatus.innerText = "⏳ Загрузка файла на сервер...";
    uploadStatus.style.color = "blue";

    try {
        const response = await fetch("/mimic/upload_video", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.status === "success") {
            uploadStatus.innerText = "✅ Файл загружен! Теперь нажмите 'Начать обработку'.";
            uploadStatus.style.color = "green";

            // Активируем кнопку обработки, разблокируем upload для повторной загрузки
            btnProcessUpload.disabled = false;
            btnProcessUpload.style.background = "#28a745";
            btnProcessUpload.style.color = "white";
            btnUploadAction.disabled = false;
        } else {
            uploadStatus.innerText = "❌ Ошибка загрузки: " + result.message;
            uploadStatus.style.color = "red";
            btnUploadAction.disabled = false;
        }
    } catch (err) {
        console.error(err);
        uploadStatus.innerText = "❌ Ошибка сети при загрузке";
        uploadStatus.style.color = "red";
        btnUploadAction.disabled = false;
    }
}


function resetUploadState() {
    if (btnUploadAction) btnUploadAction.disabled = false;
    if (btnProcessUpload) {
        btnProcessUpload.disabled = true;
        btnProcessUpload.style.background = '#e9ecef';
        btnProcessUpload.style.color = '#333';
    }
    if (uploadStatus) uploadStatus.innerText = '';
}

function onExerciseChange() {
    exerciseWarning.style.display = 'none';
    analysisVizSection.style.display = 'none';
    resultsContainer.style.display = 'none';
    resetUploadState();
}

function switchMode(mode) {
    const camSection = document.getElementById('cameraSection');
    const upSection = document.getElementById('uploadSection');
    const btnCam = document.getElementById('btnModeCamera');
    const btnUp = document.getElementById('btnModeUpload');

    // Блок с результатами (общий)
    const sharedWrapper = document.getElementById('sharedResultsWrapper');
    const uploadResultsContainer = document.getElementById('uploadResultsContainer');

    if (mode === 'camera') {
        camSection.style.display = 'block';
        upSection.style.display = 'none';
        btnCam.style.background = "#343a40"; btnCam.style.color = "white";
        btnUp.style.background = "#e9ecef"; btnUp.style.color = "#333";

        // Возвращаем результаты в секцию камеры
        if (sharedWrapper) camSection.appendChild(sharedWrapper);

    } else {
        camSection.style.display = 'none';
        upSection.style.display = 'block';
        if(isCameraOn) stopCamera();
        btnCam.style.background = "#e9ecef"; btnCam.style.color = "#333";
        btnUp.style.background = "#343a40"; btnUp.style.color = "white";

        // Перемещаем результаты в секцию загрузки, чтобы их было видно
        if (sharedWrapper && uploadResultsContainer) {
            uploadResultsContainer.appendChild(sharedWrapper);
        }
    }

    // Скрываем старые результаты при переключении
    analysisVizSection.style.display = 'none';
    resultsContainer.style.display = 'none';
}

async function toggleCamera() {
    if (!isCameraOn) {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            videoElement.srcObject = stream;
            videoElement.play();

            videoElement.onloadedmetadata = () => {
                canvas.width = 640;
                canvas.height = 480;
                isCameraOn = true;

                camBtn.innerText = "■ Остановить камеру";
                camBtn.style.background = "#dc3545";
                recBtn.disabled = false;
                recBtn.style.opacity = "1";

                videoFeed.style.display = 'block';
                placeholder.style.display = 'none';
                statusText.innerText = "Статус: Камера активна";
                analysisVizSection.style.display = 'none';
                resultsContainer.style.display = 'none';

                connectWebSocket();
            };
        } catch (err) {
            alert("Ошибка доступа к камере: " + err);
        }
    } else {
        stopCamera();
    }
}

function stopCamera() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (ws) ws.close();
    isCameraOn = false;
    isRecording = false;
    clearInterval(timerInterval);

    camBtn.innerText = "▶ Запустить камеру";
    camBtn.style.background = "#4dabf7";
    recBtn.disabled = true;
    recBtn.style.opacity = "0.6";
    recBtn.innerText = "● Начать запись";
    recBtn.style.background = "#9b59b6";
    procBtn.disabled = true;
    procBtn.style.background = "#e9ecef";
    procBtn.style.color = "#333";

    videoFeed.style.display = 'none';
    placeholder.style.display = 'block';
    statusText.innerText = "Статус: Ожидание";
    timerDisplay.innerText = "00:00";
}

function connectWebSocket() {
    let protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    let wsUrl = protocol + window.location.host + "/mimic/ws";
    ws = new WebSocket(wsUrl);

    ws.onopen = () => sendFrames();
    ws.onmessage = (event) => {
        let data = JSON.parse(event.data);
        if (data.image) videoFeed.src = data.image;
    };
}

const SEND_FPS = 20;
let sendTimer = null;

function sendFrames() {
    if (sendTimer) clearInterval(sendTimer);
    sendTimer = setInterval(() => {
        if (!isCameraOn || ws.readyState !== WebSocket.OPEN) {
            clearInterval(sendTimer); sendTimer = null; return;
        }
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        const data = canvas.toDataURL('image/jpeg', 0.5);
        ws.send(JSON.stringify({
            image: data,
            recording: isRecording,
            exercise: exerciseSelect.value,
            draw_mesh: maskToggle.checked
        }));
    }, 1000 / SEND_FPS);
}

function toggleRecording() {
    const currentExercise = exerciseSelect.value;

    if (!isRecording) {
        if (currentExercise !== 'neutral' && !neutralRecorded) {
            statusText.innerText = "⚠ Ошибка: Сначала запишите 'Нейтральное выражение'!";
            statusText.style.color = "red";
            exerciseWarning.style.display = 'inline';
            return;
        } else {
            exerciseWarning.style.display = 'none';
        }

        isRecording = true;
        recBtn.innerText = "■ Стоп запись";
        recBtn.style.background = "#e67e22";
        statusText.innerText = "Статус: ИДЕТ ЗАПИСЬ...";
        statusText.style.color = "red";
        procBtn.disabled = true;
        procBtn.style.background = "#e9ecef";
        procBtn.style.color = "#333";

        resultsContainer.style.display = 'none';
        analysisVizSection.style.display = 'none';

        seconds = 0;
        timerInterval = setInterval(() => {
            seconds++;
            let m = Math.floor(seconds / 60);
            let s = seconds % 60;
            timerDisplay.innerText = (m<10?"0"+m:m) + ":" + (s<10?"0"+s:s);
        }, 1000);

    } else {
        isRecording = false;
        clearInterval(timerInterval);

        if (currentExercise === 'neutral') {
            neutralRecorded = true;
        }

        recBtn.innerText = "↺ Перезаписать";
        recBtn.style.background = "#dc3545";
        statusText.innerText = "Статус: Запись завершена";
        statusText.style.color = "green";

        procBtn.disabled = false;
        procBtn.style.background = "#28a745";
        procBtn.style.color = "white";
    }
}

// === ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ (УНИВЕРСАЛЬНАЯ) ===
async function processData() {
    // Определяем текущий режим (Камера или Загрузка)
    const isUploadMode = document.getElementById('uploadSection').style.display !== 'none';
    const mode = isUploadMode ? "upload" : "camera";

    // Определяем, какое упражнение выбрано (разные селекты для разных режимов)
    let exercise;
    if (isUploadMode) {
        exercise = document.getElementById('uploadExerciseSelect').value;
    } else {
        exercise = document.getElementById('exerciseSelect').value;
    }

    // Определяем активные элементы UI
    const activeBtn = isUploadMode ? btnProcessUpload : procBtn;
    const activeStatus = isUploadMode ? uploadStatus : statusText;

    // Блокируем кнопку
    activeBtn.disabled = true;
    activeBtn.classList.add("btn-processing");
    activeStatus.innerText = "Статус: Обработка данных на сервере...";
    activeStatus.style.color = "black";

    // Скрываем старые результаты перед новым запросом
    analysisVizSection.style.display = 'none';
    resultsContainer.style.display = 'none';

    const processingStartedAt = Date.now();
    activeBtn.innerText = "⏳ Обработка... (0 с)";
    processingTimerId = setInterval(() => {
        const secs = Math.floor((Date.now() - processingStartedAt) / 1000);
        activeBtn.innerText = `⏳ Обработка... (${secs} с)`;
    }, 500);

    try {
        const response = await fetch("/mimic/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                exercise: exercise,
                mode: mode
            })
        });

        const result = await response.json();

        if (result.status === "success") {
            const totalSecs = Math.round((Date.now() - processingStartedAt) / 1000);
            activeStatus.innerText = `Статус: Готово! (за ${totalSecs} с)`;
            activeStatus.style.color = "green";
            renderResults(result.data, exercise);
        } else {
            const msg = (result.data && result.data.message) || result.message || "Ошибка обработки";
            alert("Ошибка: " + msg);
            activeStatus.innerText = "Статус: " + msg;
            activeStatus.style.color = "red";
        }

    } catch (err) {
        console.error(err);
        alert("Ошибка сети");
        activeStatus.innerText = "Статус: Ошибка сети";
        activeStatus.style.color = "red";
    } finally {
        clearInterval(processingTimerId);
        processingTimerId = null;
        activeBtn.disabled = false;
        activeBtn.classList.remove("btn-processing");
        activeBtn.innerText = isUploadMode ? "Начать обработку загруженного файла" : "Начать обработку";
        if (!isUploadMode && neutralRecorded && !isRecording) {
             activeBtn.style.background = "#28a745";
             activeBtn.style.color = "white";
        }
    }
}

async function runFusion() {
    const btn = document.getElementById('btnFusion');
    const box = document.getElementById('fusionResult');
    btn.disabled = true;
    btn.classList.add("btn-processing");
    box.innerHTML = '<span style="color:#888;">⏳ Вычисляю итоговый результат...</span>';

    try {
        const resp = await fetch('/mimic/fusion', { method: 'POST' });
        const json = await resp.json();
        if (json.status !== 'success') {
            box.innerHTML = '<span style="color:#c62828;">⚠ ' + (json.data?.message || json.message || 'Ошибка') + '</span>';
            btn.classList.remove("btn-processing");
            btn.disabled = false;
            return;
        }
        const d = json.data;
        const p = d.prediction;
        const isP = p.label === 'PD';
        const col = isP ? '#b71c1c' : '#1b5e20';
        const bg  = isP ? '#ffebee' : '#e8f5e9';
        const c   = d.components || {};
        box.innerHTML =
            '<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">' +
                '<div style="padding:12px; border-radius:8px; background:#e8f5e9; border:1px solid #a5d6a7; text-align:center;">' +
                    '<div style="color:#2e7d32; font-weight:bold;">Здоров</div>' +
                    '<div style="font-size:1.8em; font-weight:700; color:#1b5e20;">' + (p.proba_healthy*100).toFixed(1) + '%</div>' +
                '</div>' +
                '<div style="padding:12px; border-radius:8px; background:#ffebee; border:1px solid #ef9a9a; text-align:center;">' +
                    '<div style="color:#c62828; font-weight:bold;">Болезнь Паркинсона</div>' +
                    '<div style="font-size:1.8em; font-weight:700; color:#b71c1c;">' + (p.proba_pd*100).toFixed(1) + '%</div>' +
                '</div>' +
            '</div>' +
            '<div style="padding:10px; text-align:center; font-size:1.05em; font-weight:bold; border-radius:6px; background:' + bg + '; color:' + col + ';">' +
                'Итог: ' + (isP ? "Болезнь Паркинсона" : 'Здоров') +
            '</div>' +
            '<div style="margin-top:12px; font-size:0.9em; color:#555;">' +
                '<b>Вероятность заболевания P(PD) по упражнениям:</b> ' +
                'улыбка ' + (c.smile*100).toFixed(1) + '% &nbsp;|&nbsp; ' +
                'брови ' + (c.brow*100).toFixed(1) + '% &nbsp;|&nbsp; ' +
                'нахмуривание ' + (c.frown*100).toFixed(1) + '%' +
            '</div>';
    } catch(e) {
        box.innerHTML = '<span style="color:#c62828;">Ошибка запроса: ' + e + '</span>';
    }
    btn.classList.remove("btn-processing");
    btn.disabled = false;
}
