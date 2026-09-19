// static/js/gait.js

// Индикатор "идёт обработка" для кнопок
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

let gaitSignalChartInstance = null;
let gaitShapChartInstance = null;

document.addEventListener("DOMContentLoaded", function () {
    const btnUpload = document.getElementById("btnGaitUploadAction");
    const btnProcess = document.getElementById("btnGaitProcessUpload");

    if (btnUpload) {
        btnUpload.addEventListener("click", uploadGaitFile);
    }
    if (btnProcess) {
        btnProcess.addEventListener("click", processGaitData);
    }
});

// === 1. ЗАГРУЗКА ФАЙЛА ПОХОДКИ НА СЕРВЕР ===
async function uploadGaitFile() {
    const fileInput = document.getElementById("gaitFileInput");
    const uploadStatus = document.getElementById("gaitUploadStatus");
    const btnUpload = document.getElementById("btnGaitUploadAction");
    const btnProcess = document.getElementById("btnGaitProcessUpload");

    const file = fileInput ? fileInput.files[0] : null;
    if (!file) {
        alert("Пожалуйста, выберите CSV или Excel файл с сигналами!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    btnUpload.disabled = true;
    uploadStatus.innerText = "⏳ Загрузка файла данных...";
    uploadStatus.style.color = "#0056b3";

    try {
        const response = await fetch("/gait/upload_file", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.status === "success") {
            uploadStatus.innerText = "✅ Файл успешно загружен! Нажмите 'Запустить анализ'.";
            uploadStatus.style.color = "green";

            btnProcess.disabled = false;
            btnProcess.style.background = "#28a745";
            btnProcess.style.color = "white";
        } else {
            uploadStatus.innerText = "❌ Ошибка загрузки: " + (result.message || "Неизвестная ошибка");
            uploadStatus.style.color = "red";
        }
    } catch (err) {
        console.error(err);
        uploadStatus.innerText = "❌ Сетевая ошибка при загрузке файла";
        uploadStatus.style.color = "red";
    } finally {
        btnUpload.disabled = false;
    }
}

// === 2. ЗАПУСК ОБРАБОТКИ СИГНАЛОВ ПОХОДКИ ===
async function processGaitData() {
    const btnProcess = document.getElementById("btnGaitProcessUpload");
    const uploadStatus = document.getElementById("gaitUploadStatus");
    const resultsContainer = document.getElementById("gaitResultsContainer");

    btnProcess.disabled = true;
    btnProcess.classList.add("btn-processing");
    uploadStatus.innerText = "⏳ Выполняется цифровой анализ сигналов...";
    uploadStatus.style.color = "black";

    if (resultsContainer) {
        resultsContainer.style.display = "none";
    }

    try {
        const response = await fetch("/gait/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });

        const result = await response.json();

        if (result.status === "success") {
            uploadStatus.innerText = "✅ Анализ походки успешно завершен!";
            uploadStatus.style.color = "green";
            renderGaitResults(result.data);
        } else {
            const msg = (result.data && result.data.message) || result.message || "Ошибка обработки";
            uploadStatus.innerText = "❌ " + msg;
            uploadStatus.style.color = "red";
        }
    } catch (err) {
        console.error(err);
        uploadStatus.innerText = "❌ Ошибка при передаче данных обработки";
        uploadStatus.style.color = "red";
    } finally {
        btnProcess.disabled = false;
        btnProcess.classList.remove("btn-processing");
    }
}

// === 3. ОТРЕСОВКА РЕЗУЛЬТАТОВ И ГРАФИКОВ ===
function renderGaitResults(data) {
    const resultsContainer = document.getElementById("gaitResultsContainer");
    if (!resultsContainer) return;

    resultsContainer.style.display = "block";

    // А) Отображение качества и статуса
    if (data.quality) {
        renderGaitQuality(data.quality);
    }

    // Б) Отображение вероятностей классификации
    if (data.prediction) {
        const p = data.prediction;
        document.getElementById("gaitProbHealthy").innerText = (p.proba_healthy * 100).toFixed(1) + "%";
        document.getElementById("gaitProbPd").innerText = (p.proba_pd * 100).toFixed(1) + "%";

        const labelBox = document.getElementById("gaitPredictionLabel");
        if (p.label === "PD") {
            labelBox.innerText = "Предсказание: Болезнь Паркинсона";
            labelBox.style.background = "#ffebee";
            labelBox.style.color = "#b71c1c";
        } else {
            labelBox.innerText = "Предсказание: Здоров";
            labelBox.style.background = "#e8f5e9";
            labelBox.style.color = "#1b5e20";
        }
    }

    // В) График временного ряда сигналов походки
    if (data.chart && data.chart.times && data.chart.series) {
        renderGaitSignalChart(data.chart.times, data.chart.series);
    }

    // Г) SHAP-интерпретация признаков
    if (data.explanation) {
        renderGaitShap(data.explanation);
    }
}

function renderGaitQuality(quality) {
    const el = document.getElementById("gaitQualityBadge");
    if (!el) return;

    const isOk = quality.level === "ok";
    el.style.background = isOk ? "#e8f5e9" : "#fff8e1";
    el.style.borderLeft = isOk ? "4px solid #66bb6a" : "4px solid #ffb300";
    el.style.color = isOk ? "#1b5e20" : "#6d4c00";
    el.innerHTML = `<strong>${isOk ? "✓" : "⚠"} Качество сигнала:</strong> ${quality.message}`;
}

function renderGaitSignalChart(times, seriesMap) {
    const chartBlock = document.getElementById("gaitChartBlock");
    const canvas = document.getElementById("gaitSignalChart");
    if (!chartBlock || !canvas) return;

    chartBlock.style.display = "block";
    const ctx = canvas.getContext("2d");

    const colorPalette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"];
    const datasets = Object.keys(seriesMap).map((key, idx) => ({
        label: key,
        data: seriesMap[key],
        borderColor: colorPalette[idx % colorPalette.length],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1
    }));

    if (gaitSignalChartInstance) {
        gaitSignalChartInstance.destroy();
    }

    gaitSignalChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: times.map(t => typeof t === "number" ? t.toFixed(2) : t),
            datasets: datasets
        },
        options: {
            responsive: true,
            interaction: { mode: "index", intersect: false },
            scales: {
                x: { title: { display: true, text: "Время (с)" } },
                y: { title: { display: true, text: "Амплитуда сигнала / Ускорение" } }
            }
        }
    });
}

function renderGaitShap(explanation) {
    const block = document.getElementById("gaitShapBlock");
    const canvas = document.getElementById("gaitShapChart");
    if (!block || !canvas) return;

    const pd = explanation.pushing_pd || [];
    const hh = explanation.pushing_healthy || [];
    const combined = [...hh, ...pd].sort((a, b) => a.shap - b.shap);

    const labels = combined.map(x => x.feature_label || x.feature);
    const values = combined.map(x => x.shap);
    const colors = combined.map(x => x.shap > 0 ? "#e57373" : "#81c784");

    canvas.style.height = (combined.length * 40 + 50) + "px";
    const ctx = canvas.getContext("2d");

    if (gaitShapChartInstance) {
        gaitShapChartInstance.destroy();
    }

    gaitShapChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Вклад SHAP",
                data: values,
                backgroundColor: colors,
                borderColor: colors
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            scales: {
                x: { title: { display: true, text: "← Здоров | Болезнь Паркинсона →" } }
            },
            plugins: { legend: { display: false } }
        }
    });

    block.style.display = "block";
}