document.addEventListener("DOMContentLoaded", () => {


    const expenseForm = document.getElementById("expenseForm");
    const askForm = document.getElementById("askForm");

    const expenseMsg = document.getElementById("expenseMsg");
    const prediction = document.getElementById("prediction");
    const responseBox = document.getElementById("response");

    // =========================
    // Add Expense
    // =========================

    if (expenseForm) {
        expenseForm.addEventListener("submit", async (e) => {

            e.preventDefault();

            const formData = new FormData(expenseForm);

            const res = await fetch("/add_expense", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            expenseMsg.innerText =
                data.message || "Expense added successfully.";

            expenseForm.reset();

            loadHistory();
            loadCharts();
        });
    }

    // =========================
    // AI Assistant
    // =========================

    if (askForm) {
        askForm.addEventListener("submit", async (e) => {

            e.preventDefault();

            responseBox.innerText = "Thinking...";

            const formData = new FormData(askForm);

            const res = await fetch("/ask", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            responseBox.innerText =
                data.answer || "No response.";
        });
    }

    // =========================
    // Prediction
    // =========================

    const predictBtn = document.getElementById("predictBtn");

    if (predictBtn) {
        predictBtn.addEventListener("click", async () => {

            const res = await fetch("/predict_future");
            const data = await res.json();

            prediction.innerText =
                data.prediction || "No prediction available.";
        });
    }

    // =========================
    // CSV
    // =========================

    const csvBtn = document.getElementById("csvBtn");

    if (csvBtn) {
        csvBtn.addEventListener("click", () => {
            window.location.href = "/download_csv";
        });
    }

    // =========================
    // PDF
    // =========================

    const pdfBtn = document.getElementById("pdfBtn");

    if (pdfBtn) {
        pdfBtn.addEventListener("click", () => {
            window.location.href = "/download_pdf";
        });
    }

    // =========================
    // History
    // =========================

    async function loadHistory() {

        try {

            const res = await fetch("/history");
            const data = await res.json();

            const history = document.getElementById("history");

            if (!history) return;

            history.innerHTML = "";

            data.history.forEach(item => {

                history.innerHTML += `
            <div>
                  ${item.date} — ${item.category} ₹${item.amount}
            </div>
            `;
            });

        } catch (err) {
            console.error(err);
        }
    }

    // =========================
    // Charts + Summary Cards
    // =========================

    async function loadCharts() {

        try {

            const res = await fetch("/get_data");
            const data = await res.json();

            // =====================
            // Summary Cards
            // =====================

            const totalSpent =
                data.totals.reduce((a, b) => a + b, 0);

            const totalCategories =
                data.categories.length;

            let topCategory = "-";

            if (data.categories.length > 0) {

                let maxIndex = 0;

                data.totals.forEach((value, index) => {

                    if (value > data.totals[maxIndex]) {
                        maxIndex = index;
                    }

                });

                topCategory =
                    data.categories[maxIndex];
            }

            const averageSpend =
                totalCategories > 0
                    ? totalSpent / totalCategories
                    : 0;

            document.getElementById("totalSpent").innerText =
                `₹${totalSpent.toFixed(2)}`;

            document.getElementById("totalCategories").innerText =
                totalCategories;

            document.getElementById("topCategory").innerText =
                topCategory;

            document.getElementById("averageSpend").innerText =
                `₹${averageSpend.toFixed(2)}`;

            // =====================
            // Pie Chart
            // =====================

            const pieCanvas =
                document.getElementById("categoryChart");

            if (window.categoryChartInstance) {
                window.categoryChartInstance.destroy();
            }

            window.categoryChartInstance =
                new Chart(pieCanvas, {
                    type: "pie",
                    data: {
                        labels: data.categories,
                        datasets: [{
                            data: data.totals
                        }]
                    }
                });

            // =====================
            // Trend Chart
            // =====================

            const trendCanvas =
                document.getElementById("trendChart");

            if (window.trendChartInstance) {
                window.trendChartInstance.destroy();
            }

            window.trendChartInstance =
                new Chart(trendCanvas, {
                    type: "line",
                    data: {
                        labels: data.dates,
                        datasets: [{
                            label: "Daily Spending",
                            data: data.daily_totals,
                            tension: 0.3
                        }]
                    }
                });

        } catch (err) {

            console.error(err);

        }
    }

    loadHistory();
    loadCharts();


});
