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

    hideSignalCanvas();
}

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(protocol + window.location.host + "/ws");

    ws.onopen = () => startSendingFrames();
    ws.onmessage = (event) => {
        if (event.data && event.data.startsWith("data:image")) {
            processedVideo.src = event.data;
        }
    };
    ws.onclose = () => cameraActive && setTimeout(connectWebSocket, 2000);
}

function startSendingFrames() {
    const FRAME_RATE = 30;
    const FRAME_INTERVAL = 1000 / FRAME_RATE;
    let lastSendTime = 0;

    const sendFrame = (timestamp) => {
        if (!cameraActive) return;
        if (!lastSendTime) lastSendTime = timestamp;
        if (timestamp - lastSendTime < FRAME_INTERVAL) return requestAnimationFrame(sendFrame);

        if (!localVideo.srcObject || localVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA)
            return requestAnimationFrame(sendFrame);

        captureCanvas.width = localVideo.videoWidth;
        captureCanvas.height = localVideo.videoHeight;
        ctx.drawImage(localVideo, 0, 0, captureCanvas.width, captureCanvas.height);

        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.5);
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(dataUrl.split(",")[1]);
        lastSendTime = timestamp;
        frameAnimationId = requestAnimationFrame(sendFrame);
    };
    frameAnimationId = requestAnimationFrame(sendFrame);
}

startBtn.onclick = () => { startBtn.disabled = true; startCamera(); };
stopBtn.onclick = () => stopCamera();

recordBtn.onclick = async () => {
    if (!cameraActive) return;
    const patientId = document.getElementById("patientId").value.trim();
    const exercise = document.getElementById("exercise").value;
    if (!patientId || !exercise) return;

    if (!recording) {
        hideSignalCanvas();
        const resp = await fetch("/start_record", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patientId, exercise })
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

if (startProcessingBtn) {
    startProcessingBtn.addEventListener("click", async () => {
        const patientId = document.getElementById("patientId").value.trim();
        const exercise = document.getElementById("exercise").value;
        const confidence = parseFloat(document.getElementById("confidenceSlider").value);

        if (!patientId || !exercise) {
            setStatus("⚠️ Укажите номер пациента и выберите упражнение", "orange");
            return;
        }

        if (!processingStatusDiv) createProcessingStatusDiv();
        startProcessingBtn.disabled = true;
        setStatus("");
        startProcessingBtn.textContent = "⏳ Обработка...";

        try {
            const response = await fetch("/raw_data_processing", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patientId, exercise, confidence })
            });

            if (!response.ok) {
                setStatus("❌ Ошибка на сервере", "red");
                return;
            }

            const result = await response.json();
            setStatus("✅ Обработка завершена", "green");
            setTimeout(() => {
                drawSignalGraph(result);
                if (result.features) {
                    drawFeatureBars(result.features);
                    renderFeatureLegend(result.features);
                }
            }, 50);

        } catch (err) {
            setStatus("❌ Ошибка сети", "red");
        } finally {
            startProcessingBtn.disabled = false;
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
    featureCanvas.width = featureCanvas.offsetWidth || 800;
    featureCanvas.height = 300;

    const ctx = featureCanvas.getContext("2d");
    ctx.clearRect(0, 0, featureCanvas.width, featureCanvas.height);

    const entries = Object.entries(features);

    const maxValueRaw = Math.max(...entries.map(([_, v]) => v));
    function roundUpToStep(value, step = 5) {
        return Math.ceil(value / step) * step;
    }

    const maxValue = roundUpToStep(maxValueRaw, 1);
    // Отступы для осей и полей
    const paddingLeft = 50;   // место слева для оси Y
    const paddingRight = 30;  // место справа, чтобы бары не выходили из видимой области
    const paddingBottom = 40; // место снизу для подписей оси X
    const paddingTop = 10;    // небольшой верхний отступ

    const chartWidth = featureCanvas.width - paddingLeft - paddingRight;
    const chartHeight = featureCanvas.height - paddingBottom - paddingTop;

    ctx.font = "12px monospace";
    ctx.textAlign = "center";

    // Рисуем ось Y
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, paddingTop);
    ctx.lineTo(paddingLeft, paddingTop + chartHeight);
    ctx.stroke();

    // Рисуем деления и подписи по оси Y
    const yStepCount = 5;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= yStepCount; i++) {
        const val = (maxValue / yStepCount) * i;
        const y = paddingTop + chartHeight - (chartHeight / yStepCount) * i;

        // Рисуем деление
        ctx.beginPath();
        ctx.moveTo(paddingLeft - 5, y);
        ctx.lineTo(paddingLeft, y);
        ctx.stroke();

        // Пишем значение
        ctx.fillText(val.toFixed(2), paddingLeft - 10, y);
    }

    // Рисуем столбцы
    const barWidth = chartWidth / entries.length - 10;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    entries.forEach(([key, value], i) => {
        const x = paddingLeft + i * (barWidth + 10) + 5;
        const barHeight = (value / maxValue) * chartHeight;
        const y = paddingTop + chartHeight - barHeight;

        ctx.fillStyle = "#007bff";
        ctx.fillRect(x, y, barWidth, barHeight);

        ctx.fillStyle = "#333";
        ctx.fillText(key, x + barWidth / 2, paddingTop + chartHeight + 20);
    });

    // Рисуем линию нормы на уровне 1, если maxValue >= 1
    if (maxValue >= 1) {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        const normLineY = paddingTop + chartHeight - (chartHeight * 1) / maxValue;
        ctx.beginPath();
        ctx.moveTo(paddingLeft, normLineY);
        ctx.lineTo(featureCanvas.width - paddingRight, normLineY);
        ctx.stroke();

        ctx.fillStyle = "red";
        ctx.textAlign = "right";
        ctx.textBaseline = "bottom";
        ctx.fillText("Норма", featureCanvas.width - paddingRight + 5, normLineY - 5);
    }
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
    DecLin: "Линейный тренд"
};

function renderFeatureLegend(features) {
    featureLegend.style.display = "block";
    featureLegend.innerHTML = "";

    const container = document.createElement("div");
    // горизонтальный flex-контейнер, центрированное содержимое
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.alignItems = "center"; // или "flex-start" для выравнивания по левой стороне
    container.style.gap = "10px";
    container.style.fontFamily = "monospace";

    Object.keys(features).forEach(key => {
        const item = document.createElement("div");
        item.textContent = `${key} - ${FEATURE_DESCRIPTIONS[key] || "—"}`;
        item.style.fontWeight = "bold";
        item.style.minWidth = "180px";
        item.style.textAlign = "center";
        container.appendChild(item);
    });

    featureLegend.appendChild(container);
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