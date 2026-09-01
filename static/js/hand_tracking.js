// === УПРАВЛЕНИЕ CANVAS ===
const signalCanvas = document.getElementById("signalCanvas");
const featureCanvas = document.getElementById("featureCanvas");
const featureLegend = document.getElementById("featureLegend");

function hideSignalCanvas() {
    if (!signalCanvas) return;
    signalCanvas.style.display = "none";
    if (!featureCanvas) return;
    featureCanvas.style.display = "none";
    if (!featureLegend) return;
    featureLegend.style.display = "none";
    setStatus("");
}

// === ЛОГИКА ЗАГРУЗКИ ФАЙЛА ===
document.getElementById("uploadBtn").addEventListener("click", async () => {
    const input = document.getElementById("files");
    const file = input.files[0];

    if (!file) {
        alert("Выберите файл для загрузки");
        return;
    }

    hideSignalCanvas();

    let statusDiv = document.getElementById("upload-status");
    if (!statusDiv) {
        statusDiv = document.createElement("div");
        statusDiv.id = "upload-status";
        statusDiv.style.marginTop = "20px";
        statusDiv.style.fontFamily = "monospace";
        document.querySelector("#uploadSection").appendChild(statusDiv);
    }

    const updateStatus = (msg) => {
        console.log("[Upload]", msg);
        statusDiv.textContent = msg;
    };

    const data = new FormData();
    data.append("file", file);

    try {
        updateStatus(`⏳ Загружаем файл: ${file.name}...`);
        const res = await fetch("/upload", {
            method: "POST",
            body: data,
        });

        if (!res.ok) {
            updateStatus("❌ Ошибка при загрузке файла");
            return;
        }

        const json = await res.json();
        if (json.status === "success") {
            updateStatus(`✅ Файл успешно загружен: ${json.uploaded}`);
            input.value = "";
            setTimeout(() => (statusDiv.textContent = ""), 3000);
        } else {
            updateStatus(`❌ Ошибка: ${json.message}`);
        }
    } catch (err) {
        console.error("Ошибка сети:", err);
        updateStatus("❌ Ошибка сети при загрузке файла");
    }
});

// === МЕНЮ ===
const mainMenu = document.getElementById("mainMenu");
const backBtn = document.getElementById("backBtn");
const sections = document.querySelectorAll(".section");

function showMenu() {
    sections.forEach(s => s.classList.remove("active"));
    mainMenu.style.display = "grid";
    backBtn.style.display = "none";
    hideSignalCanvas();
}

function showSection(id) {
    mainMenu.style.display = "none";
    sections.forEach(s => s.classList.remove("active"));
    document.getElementById(id).classList.add("active");
    backBtn.style.display = "inline-block";
    hideSignalCanvas();
}

document.querySelectorAll("#mainMenu button").forEach(btn => {
    btn.addEventListener("click", () => showSection(btn.dataset.section));
});
backBtn.addEventListener("click", showMenu);

// === ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ===
const cameraSection = document.getElementById("cameraSection");
const uploadSection = document.getElementById("uploadSection");
const modeCamera = document.getElementById("modeCamera");
const modeUpload = document.getElementById("modeUpload");

function switchMode(mode) {
    if (mode === "camera") {
        cameraSection.style.display = "block";
        uploadSection.style.display = "none";
        modeCamera.classList.add("active");
        modeUpload.classList.remove("active");
    } else {
        cameraSection.style.display = "none";
        uploadSection.style.display = "block";
        modeUpload.classList.add("active");
        modeCamera.classList.remove("active");
    }
    hideSignalCanvas();
}
modeCamera.addEventListener("click", () => switchMode("camera"));
modeUpload.addEventListener("click", () => switchMode("upload"));

// === ЛОГИКА КАМЕРЫ ===
const localVideo = document.getElementById("localVideo");
const processedVideo = document.getElementById("processedVideo");
const debugCanvas = document.getElementById("debugCanvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const recordBtn = document.getElementById("recordBtn");
const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");

let ws;
let recording = false;
let cameraActive = false;
let frameAnimationId = null;
let currentSettings = { confidence: 0.6 };
// MediaPipe Hands на сервере обрабатывает кадр ~100+ мс — заметно дольше, чем
// интервал между кадрами с камеры (30 fps). Если слать кадры по таймеру, не
// дожидаясь ответа, очередь необработанных кадров растёт безостановочно —
// видео на глазах всё сильнее отстаёт от реальности. Поэтому шлём кадр только
// после того, как получили обратно предыдущий — сервер сам определяет темп.
let frameInFlight = false;

const captureCanvas = document.createElement("canvas");
const ctx = captureCanvas.getContext("2d");

confidenceSlider.addEventListener('input', (e) => {
    currentSettings.confidence = parseFloat(e.target.value);
    confidenceValue.textContent = currentSettings.confidence.toFixed(1);
});

async function startCamera() {
    if (cameraActive) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        localVideo.srcObject = stream;
        cameraActive = true;
        localVideo.onloadedmetadata = () => connectWebSocket();

        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
    } catch (err) {
        console.error("Ошибка доступа к камере:", err);
        startBtn.disabled = false;
    }
}

function stopCamera() {
    if (!cameraActive) return;
    if (frameAnimationId) cancelAnimationFrame(frameAnimationId);
    if (ws) ws.close();
    if (localVideo.srcObject) localVideo.srcObject.getTracks().forEach(track => track.stop());

    localVideo.srcObject = null;
    processedVideo.src = '';
    cameraActive = false;
    recording = false;
    recordBtn.textContent = "🔴 Начать запись";
    recordBtn.classList.remove('recording');

    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
    startBtn.disabled = false;
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(protocol + window.location.host + "/ws");
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        frameInFlight = false; // на случай переподключения посреди ожидания ответа
        startSendingFrames();
    };
    ws.onmessage = (event) => {
        // Проверяем, пришло ли бинарное изображение или текст
        if (typeof event.data === "string") {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === "recording_stopped") {
                    console.log("⏹ Автоостановка записи:", msg.reason);
                    recording = false;
                    recordBtn.textContent = "🔴 Начать запись";
                    recordBtn.classList.remove("recording");
                    setStatus("⏹ Запись завершена", "orange");
                }
            } catch (e) {
                console.warn("Нераспознанное сообщение:", event.data);
            }
            return;
        }

        // Если бинарные данные — это кадр
        const blob = new Blob([event.data], { type: "image/jpeg" });
        const oldUrl = processedVideo.src;
        processedVideo.src = URL.createObjectURL(blob);
        if (oldUrl && oldUrl.startsWith("blob:")) URL.revokeObjectURL(oldUrl);
        frameInFlight = false; // сервер ответил — можно слать следующий кадр
    };
    ws.onclose = () => cameraActive && setTimeout(connectWebSocket, 2000);
}

function startSendingFrames() {
    const MAX_FRAME_RATE = 30; // верхний потолок, реальный темп задаёт ack от сервера
    const MIN_FRAME_INTERVAL = 1000 / MAX_FRAME_RATE;
    let lastSendTime = 0;

    const sendFrame = (timestamp) => {
        if (!cameraActive) return;
        if (frameInFlight) return requestAnimationFrame(sendFrame); // ждём ответа сервера на предыдущий кадр
        if (!lastSendTime) lastSendTime = timestamp;
        if (timestamp - lastSendTime < MIN_FRAME_INTERVAL) return requestAnimationFrame(sendFrame);

        if (!localVideo.srcObject || localVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA)
            return requestAnimationFrame(sendFrame);

        const targetWidth = 320;
        const targetHeight = Math.floor(localVideo.videoHeight * (targetWidth / localVideo.videoWidth));
        captureCanvas.width = targetWidth;
        captureCanvas.height = targetHeight;
        ctx.drawImage(localVideo, 0, 0, targetWidth, targetHeight);
        captureCanvas.toBlob((blob) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                frameInFlight = true;
                ws.send(blob);
            }
            lastSendTime = timestamp;
            requestAnimationFrame(sendFrame);
        }, "image/jpeg", 0.5);
    };
    frameAnimationId = requestAnimationFrame(sendFrame);
}

startBtn.onclick = () => { startBtn.disabled = true; startCamera(); };
stopBtn.onclick = () => stopCamera();

recordBtn.onclick = async () => {
    if (!cameraActive) return;
    const patientIdElem = document.getElementById("patientId");
    const exerciseElem = document.getElementById("exercise");
    const handElem = document.getElementById("hand");
    const confidenceSliderElem = document.getElementById("confidenceSlider");


    const patientId = patientIdElem.value.trim();
    const exercise = exerciseElem.value;
    const hand = handElem.value;
    const confidence = parseFloat(confidenceSliderElem.value);

    console.log("Отправляемые данные:", { patientId, exercise, hand, confidence });

    if (!patientId) {
        setStatus("⚠️ Укажите номер пациента", "orange");
        return;
    }
    if (!exercise) {
        setStatus("⚠️ Выберите упражнение", "orange");
        return;
    }
    if (!hand) {
        setStatus("⚠️ Выберите руку", "orange");
        return;
    }

    if (!recording) {
        hideSignalCanvas();
        const resp = await fetch("/start_record", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patientId, exercise, confidence, hand })
        });
        const data = await resp.json();
        if (data.status === "started") {
            recording = true;
            recordBtn.textContent = "⏹ Остановить запись";
            recordBtn.classList.add('recording');
        }
    } else {
        const resp = await fetch("/stop_record", { method: "POST" });
        const data = await resp.json();
        recording = false;
        recordBtn.textContent = "🔴 Начать запись";
        recordBtn.classList.remove('recording');
    }
};

window.addEventListener('beforeunload', stopCamera);

// === ОБРАБОТКА ===
const startProcessingBtn = document.getElementById("startProcessingBtn");
let processingStatusDiv = null;
let processingTimerId = null;

// Индикатор "идёт работа": обработка запроса на сервере не отдаёт промежуточный
// прогресс (обычный POST, не WebSocket), поэтому честный процент не показать —
// но живая анимация + секундомер хотя бы явно показывают, что приложение не зависло.
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

if (startProcessingBtn) {
    startProcessingBtn.addEventListener("click", async () => {
        const patientIdElem = document.getElementById("patientId");
        const exerciseElem = document.getElementById("exercise");
        const handElem = document.getElementById("hand");
        const confidenceSliderElem = document.getElementById("confidenceSlider");


        const patientId = patientIdElem.value.trim();
        const exercise = exerciseElem.value;
        const hand = handElem.value;
        const confidence = parseFloat(confidenceSliderElem.value);

        console.log("Отправляемые данные:", { patientId, exercise, hand, confidence });

        if (!patientId) {
            setStatus("⚠️ Укажите номер пациента", "orange");
            return;
        }
        if (!exercise) {
            setStatus("⚠️ Выберите упражнение", "orange");
            return;
        }
        if (!hand) {
            setStatus("⚠️ Выберите руку", "orange");
            return;
        }

        if (!processingStatusDiv) createProcessingStatusDiv();
        startProcessingBtn.disabled = true;
        startProcessingBtn.classList.add("btn-processing");
        setStatus("");

        const processingStartedAt = Date.now();
        startProcessingBtn.textContent = "⏳ Обработка... (0 с)";
        processingTimerId = setInterval(() => {
            const secs = Math.floor((Date.now() - processingStartedAt) / 1000);
            startProcessingBtn.textContent = `⏳ Обработка... (${secs} с)`;
        }, 500);

        try {
            const response = await fetch("/raw_data_processing", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patientId, exercise, confidence, hand })
            });

            if (!response.ok) {
                let message = "Ошибка на сервере";
                try {
                    const errBody = await response.json();
                    if (errBody && errBody.message) message = errBody.message;
                } catch (e) { /* тело не JSON — оставляем дефолтное сообщение */ }
                setStatus("❌ " + message, "red");
                return;
            }

            const result = await response.json();
            const totalSecs = Math.round((Date.now() - processingStartedAt) / 1000);
            setStatus(`✅ Обработка завершена за ${totalSecs} с`, "green");
            setTimeout(() => {
                drawSignalGraph(result);
                if (result.features) {
                    drawFeatureBars(result.features_norm);
                    renderFeatureLegend(result.features);
                }
            }, 50);

        } catch (err) {
            setStatus("❌ Ошибка сети", "red");
        } finally {
            clearInterval(processingTimerId);
            processingTimerId = null;
            startProcessingBtn.disabled = false;
            startProcessingBtn.classList.remove("btn-processing");
            startProcessingBtn.textContent = "Начать обработку";
        }
    });
}

function drawSignalGraph(result) {
    // Показать канвас
    signalCanvas.style.display = "block";
    signalCanvas.width = signalCanvas.offsetWidth || 800;
    signalCanvas.height = signalCanvas.offsetHeight || 400;

    const ctx = signalCanvas.getContext("2d");
    if (!ctx) {
        console.error("❌ Не удалось получить 2D контекст!");
        return;
    }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, signalCanvas.width, signalCanvas.height);

    const { values, frames, times, max_X, min_X, max_Y, min_Y } = result;
    if (!values || !frames) return;


    // const minFrame = Math.min(...frames);
    // const minTime = Math.min(...times);
    // const minValue = Math.min(...values);
    const minFrame = 0;
    const minTime = 0;
    const minValue = 0;
    // const maxValue = Math.max(...values);

    const padding = 80;
    const maxValueRaw = Math.max(...values);
    const maxTimeRaw = Math.max(...times);
    // Функция округления вверх до ближайшего кратного 5
    function roundUpToStep(value, step = 5) {
        return Math.ceil(value / step) * step;
    }

    const maxValue = roundUpToStep(maxValueRaw, 5);
    const maxTime = roundUpToStep(maxTimeRaw, 1);

    // Количество делений по оси Y
    const yStepCount = maxValue / 5;
    const xStepCount = maxTime / 1;
    const scaleX = (signalCanvas.width - 2 * padding) / (maxTime - minTime);
    const scaleY = (signalCanvas.height - 2 * padding) / (maxValue - minValue || 1);

    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, signalCanvas.height - padding);
    ctx.lineTo(signalCanvas.width - padding, signalCanvas.height - padding);
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.font = "14px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    // Подпись оси X
    ctx.fillText("Время, с", signalCanvas.width / 2, signalCanvas.height - 30);

    // Подписи и цифры по оси X
    for (let i = 0; i <= xStepCount; i++) {
        let val = i;
        let px = padding + (val - minTime) * scaleX; // если используете time, иначе без минмма
        let py = signalCanvas.height - padding + 5; // чуть ниже линии оси X
        ctx.fillText(val.toFixed(0), px, py);
    }

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Подпись оси Y (с поворотом)
    ctx.translate(15, signalCanvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Амплитуда движения", 0, 0);
    ctx.restore();

    // Подписи и цифры по оси Y
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= yStepCount; i++) {
        let val = i * 5;
        let py = signalCanvas.height - padding - val * scaleY;
        ctx.fillText(val.toFixed(0), padding - 10, py);
    }
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.beginPath();
    times.forEach((x, i) => {
        const px = padding + (x - minTime) * scaleX;
        const py = signalCanvas.height - padding - (values[i] - minValue) * scaleY;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.fillStyle = "red";
    if (Array.isArray(max_X)) {
        max_X.forEach((idx, i) => {
            const px = padding + (times[idx] - minTime) * scaleX;
            const py = signalCanvas.height - padding - (max_Y[i] - minValue) * scaleY;
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    ctx.fillStyle = "green";
    if (Array.isArray(min_X)) {
        min_X.forEach((idx, i) => {
            const px = padding + (times[idx] - minTime) * scaleX;
            const py = signalCanvas.height - padding - (min_Y[i] - minValue) * scaleY;
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
}



function drawFeatureBars(features) {
    featureCanvas.style.display = "block";
    featureCanvas.width = featureCanvas.offsetWidth || 380;
    featureCanvas.height = featureCanvas.offsetWidth || 380;

    const ctx = featureCanvas.getContext("2d");
    ctx.clearRect(0, 0, featureCanvas.width, featureCanvas.height);

    const entries = Object.entries(features);
    const N = entries.length;

    const maxValueRaw = Math.max(...entries.map(([_, v]) => v));
    const maxValue = Math.ceil(maxValueRaw);

    const centerX = featureCanvas.width / 2;
    const centerY = featureCanvas.height / 2;
    const radius = Math.min(centerX, centerY) - 105;

    ctx.font = "bold 20px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 1; i <= maxValue; i++) {
        const r = (radius * i) / maxValue;
        ctx.beginPath();
        ctx.strokeStyle = "#ccc";
        ctx.lineWidth = 1;
        ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.fillStyle = "#000";
        ctx.fillText(i, centerX, centerY - r - 5);
    }

    entries.forEach(([key], i) => {
        const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);

        ctx.beginPath();
        ctx.strokeStyle = "#aaa";
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        ctx.stroke();

        ctx.fillStyle = "#000";
        ctx.textAlign = x < centerX ? "right" : (x > centerX ? "left" : "center");
        ctx.textBaseline = y < centerY ? "bottom" : "top";
        ctx.fillText(key, x, y);
    });

    // === Отрисовка полученных признаков ===
    ctx.beginPath();
    entries.forEach(([_, value], i) => {
        const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
        const r = (radius * value) / maxValue;
        const x = centerX + r * Math.cos(angle);
        const y = centerY + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });

    ctx.closePath();
    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "rgba(0,255,0,0.3)";
    ctx.fill();

    // Отрисовка значений нормы ===
    if (maxValue >= 1) {
        ctx.beginPath();
        entries.forEach(([_, __], i) => {
            const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
            const r = (radius * 1) / maxValue;
            const x = centerX + r * Math.cos(angle);
            const y = centerY + r * Math.sin(angle);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.strokeStyle = "red";
        ctx.lineWidth = 3;
        ctx.stroke();
    }

    // === Отрисовка показателей при болезни Паркинсона ===
    ctx.beginPath();
    entries.forEach(([key], i) => {
        const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
        const value = LEVEL_PD_NORMS[key] ?? 3;
        const r = (radius * value) / maxValue;
        const x = centerX + r * Math.cos(angle);
        const y = centerY + r * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.stroke();

    // === Отрисовка легенды ===
    const legendX = 20;
    const legendY = 20;
    const legendWidth = 200;
    const legendHeight = 90;

    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
    ctx.strokeStyle = "#ccc";
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    ctx.font = "15px Arial, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 15);
    ctx.lineTo(legendX + 40, legendY + 15);
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.fillText("\u041d\u043e\u0440\u043c\u0430", legendX + 50, legendY + 15);

    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 30);
    ctx.lineTo(legendX + 40, legendY + 30);
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.fillText("\u0411\u043e\u043b\u0435\u0437\u043d\u044c \u041f\u0430\u0440\u043a\u0438\u043d\u0441\u043e\u043d\u0430", legendX + 50, legendY + 30);

    ctx.strokeStyle = "green";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX + 10, legendY + 45);
    ctx.lineTo(legendX + 40, legendY + 45);
    ctx.stroke();
    ctx.fillStyle = "#000";
    ctx.fillText("\u041f\u043e\u043b\u0443\u0447\u0435\u043d\u043d\u044b\u0435 \u043f\u0440\u0438\u0437\u043d\u0430\u043a\u0438", legendX + 50, legendY + 45);
}


const FEATURE_DESCRIPTIONS = {
    NumA: "Количество пиков (амплитуд)",
    AvgFrq: "Средняя частота",
    VarFrq: "Дисперсия частоты",
    AvgVopen: "Средняя скорость открытия",
    AvgVclose: "Средняя скорость закрытия",
    AvgA: "Средняя амплитуда",
    VarA: "Дисперсия амплитуды",
    VarVopen: "Дисперсия скорости открытия",
    VarVclose: "Дисперсия скорости закрытия",
    DecA: "Коэф. затухания амплитуды",
    DecV: "Коэф. затухания скорости",
};
const LEVEL_PD_NORMS = {
    "NumA": 0.9,
    "AvgFrq": 0.8,
    "VarFrq": 2.3,
    "AvgVopen": 0.538,
    "AvgVclose": 0.538,
    "AvgA": 0.64,
    "VarA": 1.54,
    "VarVopen": 2,
    "VarVclose": 1.65,
    "DecA": 1,
    "DecV": 1,
};
const FEATURE_NORMA = {
    "NumA": 40,
    "AvgFrq": 3.62,
    "VarFrq": 10,
    "AvgVopen": 5,
    "AvgVclose": 5.24,
    "AvgA": 73.89,
    "VarA": 18,
    "VarVopen": 21,
    "VarVclose": 19.2,
    "DecA": 1,
    "DecV": 1,
}


function renderFeatureLegend(features) {
    featureLegend.style.display = "block";
    featureLegend.innerHTML = "";


    const table = document.createElement("table");
    table.style.borderCollapse = "collapse";
    table.style.width = "100%";
    table.style.maxWidth = "900px";
    table.style.margin = "20px auto";
    table.style.fontFamily = "monospace";
    table.style.boxShadow = "0 0 10px rgba(0,0,0,0.1)";
    table.style.borderRadius = "12px";
    table.style.overflow = "hidden";

    const header = document.createElement("tr");
    header.innerHTML = `
        <th style="padding: 10px; background: #f5f5f5; text-align:left;">Признак</th>
        <th style="padding: 10px; background: #f5f5f5;">🔴 Норма</th>
        <th style="padding: 10px; background: #f5f5f5;">🔵 Паркинсон</th>
        <th style="padding: 10px; background: #f5f5f5;">🟢 Пациент</th>
    `;
    table.appendChild(header);

    Object.keys(features).forEach(key => {
        const desc = FEATURE_DESCRIPTIONS[key] || "—";
        const normValue = FEATURE_NORMA[key]; // значение нормы
        const parkinsonValue = (FEATURE_NORMA[key] * LEVEL_PD_NORMS[key]).toFixed(2) ?? "—";
        const patientValue = features[key].toFixed(2);

        const row = document.createElement("tr");
        row.style.borderBottom = "1px solid #ddd";
        row.innerHTML = `
            <td style="padding: 8px 12px; text-align:left;">
                <strong>${key}</strong><br>
                <small style="color:gray;">${desc}</small>
            </td>
            <td style="text-align:center; color:red;">${normValue}</td>
            <td style="text-align:center; color:blue;">${parkinsonValue}</td>
            <td style="text-align:center; color:green; font-weight:bold;">${patientValue}</td>
        `;
        table.appendChild(row);
    });

    featureLegend.appendChild(table);
}


function createProcessingStatusDiv() {
    processingStatusDiv = document.createElement("div");
    processingStatusDiv.id = "processing-status";
    processingStatusDiv.style.marginTop = "15px";
    processingStatusDiv.style.fontFamily = "monospace";
    startProcessingBtn.parentElement.appendChild(processingStatusDiv);
}

function setStatus(msg, color = "black") {
    if (!processingStatusDiv) createProcessingStatusDiv();
    processingStatusDiv.textContent = msg;
    processingStatusDiv.style.color = color;
    console.log("[Processing]", msg);
}