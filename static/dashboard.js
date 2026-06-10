document.addEventListener("DOMContentLoaded", () => {

    const expenseForm = document.getElementById("expenseForm");
    const askForm = document.getElementById("askForm");

    const expenseMsg = document.getElementById("expenseMsg");
    const prediction = document.getElementById("prediction");
    const responseBox = document.getElementById("response");

    const categoryFilter =
        document.getElementById("categoryFilter");

    const periodFilter =
        document.getElementById("periodFilter");

    // =========================
    // Current Filters
    // =========================

    function getFilters() {

        return {
            category:
                categoryFilter?.value || "all",

            period:
                periodFilter?.value || "all"
        };

    }

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

    const predictBtn =
        document.getElementById("predictBtn");

    if (predictBtn) {

        predictBtn.addEventListener("click", async () => {

            const res =
                await fetch("/predict_future");

            const data =
                await res.json();

            prediction.innerText =
                data.prediction ||
                "No prediction available.";

        });

    }

    // =========================
    // CSV
    // =========================

    const csvBtn =
        document.getElementById("csvBtn");

    if (csvBtn) {

        csvBtn.addEventListener("click", () => {

            window.location.href =
                "/download_csv";

        });

    }

    // =========================
    // PDF
    // =========================

    const pdfBtn =
        document.getElementById("pdfBtn");

    if (pdfBtn) {

        pdfBtn.addEventListener("click", () => {

            window.location.href =
                "/download_pdf";

        });

    }

    // =========================
    // Delete Expense
    // =========================

    async function deleteExpense(id) {

        if (!confirm("Delete this expense?")) {
            return;
        }

        try {

            const res =
                await fetch(
                    `/delete_expense/${id}`,
                    {
                        method: "POST"
                    }
                );

            const data =
                await res.json();

            if (data.success) {

                loadHistory();
                loadCharts();

            } else {

                alert(data.message);

            }

        } catch (err) {

            console.error(err);

        }

    }

    // =========================
    // History
    // =========================

    async function loadHistory() {

        try {

            const filters =
                getFilters();

            const res =
                await fetch(
                    `/history?category=${filters.category}&period=${filters.period}`
                );

            const data =
                await res.json();

            const history =
                document.getElementById("history");

            if (!history) return;

            history.innerHTML = "";

            data.history.forEach(item => {

                const row =
                    document.createElement("div");

                row.style.display = "flex";
                row.style.justifyContent = "space-between";
                row.style.alignItems = "center";

                row.innerHTML = `
                <span>
                    ${item.date} — ${item.category} ₹${item.amount}
                </span>

                <button
                    class="delete-btn"
                    data-id="${item.id}">
                    🗑
                </button>
            `;

                history.appendChild(row);

            });

            document
                .querySelectorAll(".delete-btn")
                .forEach(btn => {

                    btn.addEventListener(
                        "click",
                        () => deleteExpense(
                            btn.dataset.id
                        )
                    );

                });

        } catch (err) {

            console.error(err);

        }

    }

    // =========================
    // Charts + Stats
    // =========================

    async function loadCharts() {

        try {

            const filters =
                getFilters();

            const res =
                await fetch(
                    `/get_data?category=${filters.category}&period=${filters.period}`
                );

            const data =
                await res.json();

            const totalSpent =
                data.totals.reduce(
                    (a, b) => a + b,
                    0
                );

            const totalCategories =
                data.categories.length;

            let topCategory = "-";

            if (data.categories.length > 0) {

                let maxIndex = 0;

                data.totals.forEach(
                    (value, index) => {

                        if (
                            value >
                            data.totals[maxIndex]
                        ) {
                            maxIndex = index;
                        }

                    }
                );

                topCategory =
                    data.categories[maxIndex];

            }

            const averageSpend =
                totalCategories > 0
                    ? totalSpent /
                    totalCategories
                    : 0;

            document.getElementById(
                "totalSpent"
            ).innerText =
                `₹${totalSpent.toFixed(2)}`;

            document.getElementById(
                "totalCategories"
            ).innerText =
                totalCategories;

            document.getElementById(
                "topCategory"
            ).innerText =
                topCategory;

            document.getElementById(
                "averageSpend"
            ).innerText =
                `₹${averageSpend.toFixed(2)}`;

            const pieCanvas =
                document.getElementById(
                    "categoryChart"
                );

            if (
                window.categoryChartInstance
            ) {
                window.categoryChartInstance
                    .destroy();
            }

            window.categoryChartInstance =
                new Chart(
                    pieCanvas,
                    {
                        type: "pie",
                        data: {
                            labels:
                                data.categories,
                            datasets: [{
                                data:
                                    data.totals
                            }]
                        }
                    }
                );

            const trendCanvas =
                document.getElementById(
                    "trendChart"
                );

            if (
                window.trendChartInstance
            ) {
                window.trendChartInstance
                    .destroy();
            }

            window.trendChartInstance =
                new Chart(
                    trendCanvas,
                    {
                        type: "line",
                        data: {
                            labels:
                                data.dates,
                            datasets: [{
                                label:
                                    "Daily Spending",
                                data:
                                    data.daily_totals,
                                tension: 0.3
                            }]
                        }
                    }
                );

        } catch (err) {

            console.error(err);

        }

    }

    // =========================
    // Filter Events
    // =========================

    if (categoryFilter) {

        categoryFilter.addEventListener(
            "change",
            () => {

                loadHistory();
                loadCharts();

            }
        );

    }

    if (periodFilter) {

        periodFilter.addEventListener(
            "change",
            () => {

                loadHistory();
                loadCharts();

            }
        );

    }

    // =========================
    // Mobile Sidebar
    // =========================

    const menuToggle =
        document.getElementById(
            "menuToggle"
        );

    const sidebar =
        document.querySelector(
            ".sidebar"
        );

    const sidebarOverlay =
        document.getElementById(
            "sidebarOverlay"
        );

    if (
        menuToggle &&
        sidebar &&
        sidebarOverlay
    ) {

        menuToggle.addEventListener(
            "click",
            () => {

                sidebar.classList.toggle(
                    "open"
                );

                sidebarOverlay.classList.toggle(
                    "show"
                );

            }
        );

        sidebarOverlay.addEventListener(
            "click",
            () => {

                sidebar.classList.remove(
                    "open"
                );

                sidebarOverlay.classList.remove(
                    "show"
                );

            }
        );

    }

    loadHistory();
    loadCharts();

});