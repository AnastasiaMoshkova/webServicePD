// === УПРАВЛЕНИЕ CANVAS ===
const facialSignalCanvas = document.getElementById("facialSignalCanvas");
const facialFeatureCanvas = document.getElementById("facialFeatureCanvas");
const facialFeatureLegend = document.getElementById("facialFeatureLegend");

function hideFacialCanvas() {
    if (facialSignalCanvas) facialSignalCanvas.style.display = "none";
    if (facialFeatureCanvas) facialFeatureCanvas.style.display = "none";
    if (facialFeatureLegend) facialFeatureLegend.style.display = "none";
    setFacialStatus("");
}

// === MEDIAPIPE FACE LANDMARKER ===
let faceLandmarker = null;
let facialRecording = false;
let facialCameraActive = false;
let recordedData = [];
let lastVideoTime = -1;

// MediaPipe возвращает 52 blendshapes (аналог Action Units)
const BLENDSHAPE_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
    'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen',
    'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower',
    'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft',
    'noseSneerRight', '_neutral'
];

// Инициализация MediaPipe
async function initializeFaceLandmarker() {
    const vision = await window.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    
    faceLandmarker = await window.FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });
    
    console.log("MediaPipe Face Landmarker инициализирован");
}

// === ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ===
const facialCameraSection = document.getElementById("facialCameraSection");
const facialUploadSection = document.getElementById("facialUploadSection");
const facialModeCamera = document.getElementById("facialModeCamera");
const facialModeUpload = document.getElementById("facialModeUpload");

function switchFacialMode(mode) {
    if (mode === "camera") {
        facialCameraSection.style.display = "block";
        facialUploadSection.style.display = "none";
        facialModeCamera.classList.add("active");
        facialModeUpload.classList.remove("active");
    } else {
        facialCameraSection.style.display = "none";
        facialUploadSection.style.display = "block";
        facialModeUpload.classList.add("active");
        facialModeCamera.classList.remove("active");
    }
    hideFacialCanvas();
}

facialModeCamera?.addEventListener("click", () => switchFacialMode("camera"));
facialModeUpload?.addEventListener("click", () => switchFacialMode("upload"));

// === КАМЕРА ===
const facialVideo = document.getElementById("facialVideo");
const facialCanvas = document.getElementById("facialCanvas");
const facialStartBtn = document.getElementById("facialStartBtn");
const facialStopBtn = document.getElementById("facialStopBtn");
const facialRecordBtn = document.getElementById("facialRecordBtn");

const canvasCtx = facialCanvas.getContext("2d");

async function startFacialCamera() {
    if (facialCameraActive) return;
    
    try {
        if (!faceLandmarker) {
            setFacialStatus("⏳ Загрузка модели MediaPipe...", "blue");
            await initializeFaceLandmarker();
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        facialVideo.srcObject = stream;
        facialCameraActive = true;
        
        facialVideo.addEventListener("loadeddata", () => {
            facialCanvas.width = facialVideo.videoWidth;
            facialCanvas.height = facialVideo.videoHeight;
            predictWebcam();
        });
        
        facialStartBtn.style.display = 'none';
        facialStopBtn.style.display = 'inline-block';
        setFacialStatus("✅ Камера запущена", "green");
        
    } catch (err) {
        console.error("Ошибка доступа к камере:", err);
        setFacialStatus("❌ Ошибка доступа к камере", "red");
    }
}

function stopFacialCamera() {
    if (!facialCameraActive) return;
    
    facialCameraActive = false;
    facialRecording = false;
    
    if (facialVideo.srcObject) {
        facialVideo.srcObject.getTracks().forEach(track => track.stop());
    }
    
    facialVideo.srcObject = null;
    canvasCtx.clearRect(0, 0, facialCanvas.width, facialCanvas.height);
    
    facialRecordBtn.textContent = "🔴 Начать запись";
    facialRecordBtn.classList.remove('recording');
    facialStartBtn.style.display = 'inline-block';
    facialStopBtn.style.display = 'none';
    setFacialStatus("", "black");
}

// === ОТРИСОВКА ТОЧЕК (КРАСНЫМ ЦВЕТОМ) ===
function drawFacialLandmarks(landmarks) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Рисуем все 478 точек красным цветом
    for (const landmark of landmarks) {
        const x = landmark.x * facialCanvas.width;
        const y = landmark.y * facialCanvas.height;
        
        canvasCtx.fillStyle = 'red';
        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 2, 0, 2 * Math.PI);
        canvasCtx.fill();
    }
}

// === ОБРАБОТКА КАДРОВ В РЕАЛЬНОМ ВРЕМЕНИ ===
async function predictWebcam() {
    if (!facialCameraActive) return;
    
    canvasCtx.drawImage(facialVideo, 0, 0, facialCanvas.width, facialCanvas.height);
    
    if (faceLandmarker && facialVideo.currentTime !== lastVideoTime) {
        lastVideoTime = facialVideo.currentTime;
        
        const results = await faceLandmarker.detectForVideo(facialVideo, performance.now());
        
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            drawFacialLandmarks(results.faceLandmarks[0]);
            
            // Если идет запись, сохраняем blendshapes (AU)
            if (facialRecording && results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                const blendshapes = {};
                results.faceBlendshapes[0].categories.forEach(category => {
                    blendshapes[category.categoryName] = category.score;
                });
                
                recordedData.push({
                    timestamp: facialVideo.currentTime,
                    blendshapes: blendshapes,
                    landmarks: results.faceLandmarks[0]
                });
            }
        }
    }
    
    requestAnimationFrame(predictWebcam);
}

// === ЗАПИСЬ ===
facialRecordBtn.onclick = async () => {
    if (!facialCameraActive) {
        setFacialStatus("⚠️ Сначала запустите камеру", "orange");
        return;
    }
    
    const patientIdElem = document.getElementById("facialPatientId");
    const expressionElem = document.getElementById("facialExpression");
    const patientId = patientIdElem.value.trim();
    const expression = expressionElem.value;
    
    if (!patientId) {
        setFacialStatus("⚠️ Укажите номер пациента", "orange");
        return;
    }
    
    if (!expression) {
        setFacialStatus("⚠️ Выберите выражение/упражнение", "orange");
        return;
    }
    
    if (!facialRecording) {
        recordedData = [];
        facialRecording = true;
        facialRecordBtn.textContent = "⏹ Остановить запись";
        facialRecordBtn.classList.add('recording');
        setFacialStatus("🔴 Идет запись...", "red");
    } else {
        facialRecording = false;
        facialRecordBtn.textContent = "🔴 Начать запись";
        facialRecordBtn.classList.remove('recording');
        
        if (recordedData.length > 0) {
            try {
                const response = await fetch("/save_facial_recording", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        patientId,
                        expression,
                        data: recordedData
                    })
                });
                
                const result = await response.json();
                if (result.status === "success") {
                    setFacialStatus(`✅ Сохранено ${recordedData.length} кадров`, "green");
                }
            } catch (err) {
                console.error("Ошибка сохранения:", err);
                setFacialStatus("❌ Ошибка сохранения", "red");
            }
        }
    }
};

// === ОБРАБОТКА ДАННЫХ И РАСЧЕТ СТАТИСТИКИ ===
const startFacialProcessingBtn = document.getElementById("startFacialProcessingBtn");

if (startFacialProcessingBtn) {
    startFacialProcessingBtn.addEventListener("click", async () => {
        const patientIdElem = document.getElementById("facialPatientId");
        const expressionElem = document.getElementById("facialExpression");
        const patientId = patientIdElem.value.trim();
        const expression = expressionElem.value;
        
        if (!patientId || !expression) {
            setFacialStatus("⚠️ Заполните все поля", "orange");
            return;
        }
        
        startFacialProcessingBtn.disabled = true;
        startFacialProcessingBtn.textContent = "⏳ Обработка...";
        
        try {
            const response = await fetch("/facial_data_processing", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patientId, expression })
            });
            
            const result = await response.json();
            
            if (result.status === "success") {
                setFacialStatus("✅ Обработка завершена", "green");
                
                if (result.statistics) {
                    drawStatisticsTable(result.statistics);
                }
            } else {
                setFacialStatus(`❌ ${result.message}`, "red");
            }
        } catch (err) {
            console.error("Ошибка обработки:", err);
            setFacialStatus("❌ Ошибка обработки", "red");
        } finally {
            startFacialProcessingBtn.disabled = false;
            startFacialProcessingBtn.textContent = "Начать обработку";
        }
    });
}

// === ВЫЧИСЛЕНИЕ СТАТИСТИКИ (MEDIAN, STD, MIN, MAX) ===
function calculateStatistics(values) {
    if (!values || values.length === 0) {
        return { median: 0, std: 0, min: 0, max: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const n = sorted.length;
    
    // Median
    const median = n % 2 === 0 
        ? (sorted[n/2 - 1] + sorted[n/2]) / 2 
        : sorted[Math.floor(n/2)];
    
    // Mean (для расчета STD)
    const mean = values.reduce((a, b) => a + b, 0) / n;
    
    // Standard Deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const std = Math.sqrt(variance);
    
    // Min and Max
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return { median, std, min, max };
}

// === РАСЧЕТ ПРОИЗВОДНЫХ ===
function calculateDerivatives(timeSeries) {
    if (timeSeries.length < 2) return [];
    
    const derivatives = [];
    for (let i = 1; i < timeSeries.length; i++) {
        const dt = timeSeries[i].timestamp - timeSeries[i-1].timestamp;
        if (dt === 0) continue;
        
        const deriv = {};
        for (const key in timeSeries[i].blendshapes) {
            const dValue = timeSeries[i].blendshapes[key] - timeSeries[i-1].blendshapes[key];
            deriv[key] = dValue / dt;
        }
        derivatives.push({ timestamp: timeSeries[i].timestamp, derivatives: deriv });
    }
    
    return derivatives;
}

// === ТАБЛИЦА СТАТИСТИКИ ===
function drawStatisticsTable(statistics) {
    facialFeatureLegend.style.display = "block";
    facialFeatureLegend.innerHTML = "";
    
    const container = document.createElement("div");
    container.style.fontFamily = "monospace";
    container.style.overflowX = "auto";
    
    const title = document.createElement("h3");
    title.textContent = "Статистика признаков (Blendshapes / Action Units)";
    title.style.textAlign = "center";
    container.appendChild(title);
    
    // Таблица для основных признаков
    const auSection = document.createElement("div");
    const auTitle = document.createElement("h4");
    auTitle.textContent = "Action Units (Blendshapes)";
    auTitle.style.marginTop = "20px";
    auSection.appendChild(auTitle);
    
    const auTable = createStatTable(statistics.blendshapes);
    auSection.appendChild(auTable);
    container.appendChild(auSection);
    
    // Таблица для производных
    if (statistics.derivatives) {
        const derivSection = document.createElement("div");
        const derivTitle = document.createElement("h4");
        derivTitle.textContent = "Производные Action Units";
        derivTitle.style.marginTop = "30px";
        derivSection.appendChild(derivTitle);
        
        const derivTable = createStatTable(statistics.derivatives);
        derivSection.appendChild(derivTable);
        container.appendChild(derivSection);
    }
    
    facialFeatureLegend.appendChild(container);
}

function createStatTable(data) {
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.style.marginTop = "10px";
    table.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
    
    // Заголовок
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    
    ["Признак", "Median", "Std", "Min", "Max"].forEach(headerText => {
        const th = document.createElement("th");
        th.textContent = headerText;
        th.style.border = "1px solid #ddd";
        th.style.padding = "12px";
        th.style.backgroundColor = "#3498db";
        th.style.color = "white";
        th.style.textAlign = "left";
        th.style.position = "sticky";
        th.style.top = "0";
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Тело таблицы
    const tbody = document.createElement("tbody");
    
    Object.entries(data).forEach(([featureName, stats], index) => {
        const row = document.createElement("tr");
        if (index % 2 === 0) {
            row.style.backgroundColor = "#f9f9f9";
        }
        
        // Название признака
        const cellName = document.createElement("td");
        cellName.textContent = featureName;
        cellName.style.border = "1px solid #ddd";
        cellName.style.padding = "10px";
        cellName.style.fontWeight = "bold";
        row.appendChild(cellName);
        
        // Median
        const cellMedian = document.createElement("td");
        cellMedian.textContent = stats.median.toFixed(6);
        cellMedian.style.border = "1px solid #ddd";
        cellMedian.style.padding = "10px";
        row.appendChild(cellMedian);
        
        // Std
        const cellStd = document.createElement("td");
        cellStd.textContent = stats.std.toFixed(6);
        cellStd.style.border = "1px solid #ddd";
        cellStd.style.padding = "10px";
        row.appendChild(cellStd);
        
        // Min
        const cellMin = document.createElement("td");
        cellMin.textContent = stats.min.toFixed(6);
        cellMin.style.border = "1px solid #ddd";
        cellMin.style.padding = "10px";
        row.appendChild(cellMin);
        
        // Max
        const cellMax = document.createElement("td");
        cellMax.textContent = stats.max.toFixed(6);
        cellMax.style.border = "1px solid #ddd";
        cellMax.style.padding = "10px";
        row.appendChild(cellMax);
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    return table;
}

// === СТАТУС ===
let facialProcessingStatusDiv = null;

function setFacialStatus(msg, color = "black") {
    if (!facialProcessingStatusDiv) {
        facialProcessingStatusDiv = document.createElement("div");
        facialProcessingStatusDiv.style.marginTop = "15px";
        facialProcessingStatusDiv.style.fontFamily = "monospace";
        facialProcessingStatusDiv.style.textAlign = "center";
        facialProcessingStatusDiv.style.fontWeight = "bold";
        facialProcessingStatusDiv.style.fontSize = "16px";
        
        if (startFacialProcessingBtn?.parentElement) {
            startFacialProcessingBtn.parentElement.appendChild(facialProcessingStatusDiv);
        } else if (facialRecordBtn?.parentElement) {
            facialRecordBtn.parentElement.appendChild(facialProcessingStatusDiv);
        }
    }
    facialProcessingStatusDiv.textContent = msg;
    facialProcessingStatusDiv.style.color = color;
}

// === КНОПКИ ===
facialStartBtn.onclick = startFacialCamera;
facialStopBtn.onclick = stopFacialCamera;
window.addEventListener('beforeunload', stopFacialCamera);

// === ЗАГРУЗКА ФАЙЛОВ ===
document.getElementById("facialUploadBtn")?.addEventListener("click", async () => {
    const input = document.getElementById("facialFiles");
    const file = input.files[0];
    
    if (!file) {
        alert("Выберите файл для загрузки");
        return;
    }
    
    const data = new FormData();
    data.append("file", file);
    
    try {
        setFacialStatus("⏳ Загрузка файла...", "blue");
        const res = await fetch("/upload_facial", {
            method: "POST",
            body: data,
        });
        
        const json = await res.json();
        if (json.status === "success") {
            setFacialStatus(`✅ Файл успешно загружен: ${json.uploaded}`, "green");
            setTimeout(() => setFacialStatus("", "black"), 3000);
        } else {
            setFacialStatus(`❌ ${json.message}`, "red");
        }
    } catch (err) {
        console.error("Ошибка загрузки:", err);
        setFacialStatus("❌ Ошибка загрузки", "red");
    }
});

// Добавьте в конец файла для совместимости с навигацией вашего проекта

// === НАВИГАЦИЯ МЕЖДУ СЕКЦИЯМИ ===
document.querySelectorAll('.menu button[data-section]').forEach(button => {
    button.addEventListener('click', () => {
        const sectionId = button.getAttribute('data-section');
        
        // Скрываем все секции
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Показываем выбранную секцию
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
        }
        
        // Если открыли секцию мимики, инициализируем MediaPipe
        if (sectionId === 'facialSection' && !faceLandmarker) {
            initializeFaceLandmarker().catch(err => {
                console.error("Ошибка инициализации MediaPipe:", err);
            });
        }
    });
});

// === АВТОМАТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ ===
// Если страница загружена и секция мимики активна
if (document.getElementById('facialSection')?.classList.contains('active')) {
    initializeFaceLandmarker().catch(err => {
        console.error("Ошибка инициализации MediaPipe:", err);
    });
}

