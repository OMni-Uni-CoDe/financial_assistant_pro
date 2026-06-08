document.addEventListener("DOMContentLoaded", () => {

    const expenseForm = document.getElementById("expenseForm");
    const askForm = document.getElementById("askForm");

    const expenseMsg = document.getElementById("expenseMsg");
    const prediction = document.getElementById("prediction");
    const responseBox = document.getElementById("response");

    // Add expense
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
                data.message || "Expense added successfully";

            loadHistory();
            loadCharts();
        });
    }

    // AI
    if (askForm) {
        askForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            const formData = new FormData(askForm);

            const res = await fetch("/ask", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            responseBox.innerText =
                data.answer || "No response";
        });
    }

    // Prediction
    const predictBtn = document.getElementById("predictBtn");

    if (predictBtn) {
        predictBtn.addEventListener("click", async () => {

            const res = await fetch("/predict_future");

            const data = await res.json();

            prediction.innerText = data.prediction;
        });
    }

    // CSV
    const csvBtn = document.getElementById("csvBtn");

    if (csvBtn) {
        csvBtn.addEventListener("click", () => {
            window.location.href = "/download_csv";
        });
    }

    // PDF
    const pdfBtn = document.getElementById("pdfBtn");

    if (pdfBtn) {
        pdfBtn.addEventListener("click", () => {
            window.location.href = "/download_pdf";
        });
    }

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
                        ${item.date}
                        —
                        ${item.category}
                        ₹${item.amount}
                    </div>
                `;
            });

        } catch (err) {
            console.error(err);
        }
    }

    async function loadCharts() {

        try {

            const res = await fetch("/get_data");
            const data = await res.json();

            const canvas =
                document.getElementById("categoryChart");

            if (!canvas) return;

            if (window.categoryChartInstance) {
                window.categoryChartInstance.destroy();
            }

            window.categoryChartInstance =
                new Chart(canvas, {
                    type: "pie",
                    data: {
                        labels: data.categories,
                        datasets: [{
                            data: data.totals
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