document.addEventListener("DOMContentLoaded", () => {

    const categoryMap = {

        "Housing": [
            "Rent",
            "Mortgage",
            "Maintenance",
            "Furniture",
            "Home Decor",
            "Property Tax",
            "Security",
            "Household Supplies"
        ],

        "Food & Dining": [
            "Groceries",
            "Restaurant",
            "Cafe",
            "Snacks",
            "Food Delivery",
            "Bakery",
            "Fast Food",
            "Dining Out"
        ],

        "Transportation": [
            "Fuel",
            "Public Transport",
            "Taxi",
            "Ride Sharing",
            "Vehicle Maintenance",
            "Parking",
            "Tolls",
            "Vehicle Insurance"
        ],

        "Utilities": [
            "Electricity",
            "Water",
            "Gas",
            "Internet",
            "Mobile Recharge",
            "Cable TV",
            "Cloud Storage"
        ],

        "Healthcare": [
            "Doctor",
            "Medicine",
            "Hospital",
            "Dental",
            "Vision",
            "Health Insurance",
            "Gym",
            "Fitness",
            "Mental Health"
        ],

        "Education": [
            "School Fees",
            "College Fees",
            "Books",
            "Courses",
            "Certifications",
            "Exam Fees",
            "Stationery"
        ],

        "Shopping": [
            "Clothing",
            "Shoes",
            "Accessories",
            "Electronics",
            "Gadgets",
            "Home Appliances",
            "Beauty Products"
        ],

        "Entertainment": [
            "Movies",
            "Games",
            "Streaming",
            "Music",
            "Events",
            "Sports",
            "Hobbies"
        ],

        "Travel": [
            "Flights",
            "Hotels",
            "Transport",
            "Food",
            "Tourism",
            "Visa",
            "Travel Insurance"
        ],

        "Finance": [
            "EMI",
            "Loan Payment",
            "Credit Card",
            "Bank Charges",
            "Taxes",
            "Investment",
            "Trading"
        ],

        "Savings": [
            "Emergency Fund",
            "Retirement",
            "Goal Contribution",
            "Fixed Deposit",
            "Mutual Fund",
            "Stocks"
        ],

        "Family": [
            "Children",
            "Parents",
            "Spouse",
            "Family Events",
            "Gifts"
        ],

        "Pets": [
            "Pet Food",
            "Veterinary",
            "Pet Accessories",
            "Pet Insurance"
        ],

        "Work": [
            "Office Supplies",
            "Software",
            "Business Travel",
            "Freelancing",
            "Professional Services"
        ],

        "Subscriptions": [
            "Netflix",
            "Spotify",
            "YouTube Premium",
            "ChatGPT",
            "Software Subscription",
            "Membership"
        ],

        "Miscellaneous": [
            "Charity",
            "Donations",
            "Unexpected Expenses",
            "Other"
        ]

    };

    function updateSubcategories() {

        const category =
            document.getElementById(
                "category"
            )?.value;

        const subcategorySelect =
            document.getElementById(
                "subcategory"
            );

        if (
            !category ||
            !subcategorySelect
        ) {
            return;
        }

        subcategorySelect.innerHTML = "";

        categoryMap[category]
            .forEach(item => {

                const option =
                    document.createElement(
                        "option"
                    );

                option.value = item;
                option.textContent = item;

                subcategorySelect.appendChild(
                    option
                );

            });

    }

    const expenseForm = document.getElementById("expenseForm");
    const askForm = document.getElementById("askForm");

    const expenseMsg = document.getElementById("expenseMsg");
    const prediction = document.getElementById("prediction");
    const responseBox = document.getElementById("response");
    const budgetForm =
        document.getElementById("budgetForm");

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
            loadTopSubcategory();
            loadSubcategoryBreakdown();

        });

    }

    if (budgetForm) {

        budgetForm.addEventListener(
            "submit",
            async (e) => {

                e.preventDefault();

                const formData =
                    new FormData(budgetForm);

                const res =
                    await fetch(
                        "/set_budget",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                const data =
                    await res.json();

                alert(data.message);

                loadBudget();

            }
        );

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

                        <strong>
                            ${item.category}
                        </strong>

                        ${item.subcategory
                        ? `→ ${item.subcategory}`
                        : ""}

                        <br>

                        ${item.date}

                        <br>

                        ₹${item.amount}

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
            loadBudget();

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

    // =============
    // Load Budget
    // =============
    async function loadBudget() {

        try {

            const res =
                await fetch("/get_budget");

            const data =
                await res.json();

            document.getElementById(
                "budgetAmount"
            ).innerText =
                `₹${data.budget.toFixed(2)}`;

            document.getElementById(
                "remainingBudget"
            ).innerText =
                `₹${data.remaining.toFixed(2)}`;

            document.getElementById(
                "budgetPercentage"
            ).innerText =
                `${data.percentage}%`;

            const fill =
                document.getElementById(
                    "budgetFill"
                );

            fill.style.width =
                `${Math.min(data.percentage, 100)}%`;

            const alertBox =
                document.getElementById(
                    "budgetAlert"
                );

            if (data.percentage >= 100) {

                alertBox.innerText =
                    "🚨 Budget exceeded!";

            } else if (
                data.percentage >= 80
            ) {

                alertBox.innerText =
                    "⚠ Approaching budget limit";

            } else {

                alertBox.innerText = "";

            }

        } catch (err) {

            console.error(err);

        }

    }

    // ================
    // Load Insights
    // ================

    async function loadInsights() {

        try {

            const response =
                await fetch(
                    "/get_insights"
                );

            const data =
                await response.json();

            const container =
                document.getElementById(
                    "insightsContainer"
                );

            container.innerHTML = "";

            data.insights.forEach(
                insight => {

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className =
                        "insight-item";

                    card.innerHTML = `
                    <strong>
                        ${insight.title}
                    </strong>

                    <br><br>

                    ${insight.message}
                `;

                    container.appendChild(
                        card
                    );

                }
            );

        }

        catch (err) {

            console.error(
                "Insights error:",
                err
            );

        }

    }

    // ================
    // Load Forecast
    // ================

    async function loadForecast() {

        try {

            const response =
                await fetch("/get_forecast");

            const data =
                await response.json();

            const container =
                document.getElementById(
                    "forecastContainer"
                );

            container.innerHTML =
                data.forecast.replace(
                    /\n/g,
                    "<br>"
                );

        }

        catch (err) {

            console.error(err);

        }

    }

    // =========================
    // AI Recommendations
    // =========================

    async function loadRecommendations() {

        try {

            const response =
                await fetch(
                    "/get_recommendations"
                );

            const data =
                await response.json();

            const container =
                document.getElementById(
                    "recommendationsContainer"
                );

            container.innerHTML = "";

            data.recommendations.forEach(
                rec => {

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className =
                        "recommendation-item";

                    card.innerHTML = `
                    <strong>
                        ${rec.title}
                    </strong>

                    <br><br>

                    ${rec.message}
                `;

                    container.appendChild(
                        card
                    );

                }
            );

        }

        catch (err) {

            console.error(
                "Recommendations error:",
                err
            );

        }

    }

    // =========================
    // Financial Health Score
    // =========================

    async function loadHealthScore() {

        try {

            const response =
                await fetch(
                    "/get_health_score"
                );

            const data =
                await response.json();

            const scoreElement =
                document.getElementById(
                    "healthScore"
                );

            const ratingElement =
                document.getElementById(
                    "healthRating"
                );

            scoreElement.innerText =
                `${data.score}/100`;

            ratingElement.innerText =
                data.rating;

            // ======================
            // Dynamic Colors
            // ======================

            if (data.score >= 85) {

                scoreElement.style.color =
                    "#22c55e";

                ratingElement.innerText =
                    "🟢 Excellent";

            }

            else if (data.score >= 70) {

                scoreElement.style.color =
                    "#3b82f6";

                ratingElement.innerText =
                    "🔵 Good";

            }

            else if (data.score >= 50) {

                scoreElement.style.color =
                    "#facc15";

                ratingElement.innerText =
                    "🟡 Fair";

            }

            else {

                scoreElement.style.color =
                    "#ef4444";

                ratingElement.innerText =
                    "🔴 Needs Attention";

            }

        }

        catch (err) {

            console.error(
                "Health score error:",
                err
            );

        }

    }

    // =========================
    // Top Subcategory Analytics
    // =========================

    async function loadTopSubcategory() {

        try {

            const response =
                await fetch(
                    "/get_top_subcategory"
                );

            const data =
                await response.json();

            document.getElementById(
                "topSubcategory"
            ).innerText =
                data.subcategory;

            document.getElementById(
                "topSubcategoryInfo"
            ).innerText =
                `${data.category} • ₹${data.amount}
                (${data.percentage}% of spending)`;

        }

        catch (err) {

            console.error(
                "Top subcategory error:",
                err
            );

        }

    }


    async function loadSubcategoryBreakdown() {

        try {

            const response =
                await fetch(
                    "/get_subcategory_breakdown"
                );

            const data =
                await response.json();

            const container =
                document.getElementById(
                    "subcategoryBreakdownContainer"
                );

            if (!container) return;

            container.innerHTML = "";

            for (const category in data) {

                const total =
                    data[category]
                        .reduce(
                            (sum, item) =>
                                sum + item.amount,
                            0
                        );

                const wrapper =
                    document.createElement(
                        "div"
                    );

                wrapper.className =
                    "drilldown-card";

                const header =
                    document.createElement(
                        "div"
                    );

                header.className =
                    "drilldown-header";

                header.innerHTML =
                    `
                <span>
                    ▶ ${category}
                </span>

                <span>
                    ₹${total}
                </span>
                `;

                const body =
                    document.createElement(
                        "div"
                    );

                body.className =
                    "drilldown-body";

                body.style.display =
                    "none";

                data[category].forEach(
                    item => {

                        body.innerHTML += `
                        <div class="drilldown-row">
                            <span>
                                ${item.subcategory}
                            </span>

                            <span>
                                ₹${item.amount}
                            </span>
                        </div>
                    `;

                    }
                );

                header.addEventListener(
                    "click",
                    () => {

                        const open =
                            body.style.display
                            === "block";

                        body.style.display =
                            open
                                ? "none"
                                : "block";

                        header.querySelector(
                            "span"
                        ).innerHTML =
                            `${open ? "▶" : "▼"} ${category}`;

                    }
                );

                wrapper.appendChild(
                    header
                );

                wrapper.appendChild(
                    body
                );

                container.appendChild(
                    wrapper
                );

            }

        }

        catch (err) {

            console.error(
                "Subcategory breakdown error:",
                err
            );

        }

    }


    // =========================
    // Monthly Comparison
    // =========================

    window.loadMonthlyComparison =
        async function () {

            try {

                const response =
                    await fetch(
                        "/get_monthly_comparison"
                    );

                const data =
                    await response.json();

                if (data.comparison) {

                    document.getElementById(
                        "comparisonValue"
                    ).innerText =
                        "--";

                    document.getElementById(
                        "comparisonTrend"
                    ).innerText =
                        data.comparison;

                    return;

                }

                document.getElementById(
                    "comparisonValue"
                ).innerText =
                    `₹${data.current_month}`;

                document.getElementById(
                    "comparisonTrend"
                ).innerText =
                    data.trend;

            }

            catch (err) {

                console.error(
                    "Comparison error:",
                    err
                );

            }

        };

    // =========================
    // Savings Goal
    // =========================

    window.saveGoal = async function () {

        try {

            const formData = new FormData();

            formData.append(
                "goal_name",
                document.getElementById(
                    "goalName"
                ).value
            );

            formData.append(
                "target_amount",
                document.getElementById(
                    "goalTarget"
                ).value
            );

            formData.append(
                "current_amount",
                document.getElementById(
                    "goalCurrent"
                ).value
            );

            const response =
                await fetch(
                    "/set_goal",
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            alert(
                data.message ||
                "Goal saved."
            );

            loadGoal();

        }

        catch (err) {

            console.error(
                "Goal save error:",
                err
            );

            alert(
                "Unable to save goal."
            );

        }

    };


    window.loadGoal = async function () {

        const response =
            await fetch(
                "/get_goal"
            );

        const data =
            await response.json();

        document.getElementById(
            "goalDisplayName"
        ).innerText =
            data.goal_name || "-";

        document.getElementById(
            "goalCurrentDisplay"
        ).innerText =
            data.current_amount;

        document.getElementById(
            "goalTargetDisplay"
        ).innerText =
            data.target_amount;

        document.getElementById(
            "goalPercentage"
        ).innerText =
            `${data.percentage}%`;

        document.getElementById(
            "goalProgressBar"
        ).style.width =
            `${Math.min(data.percentage, 100)}%`;

        document.getElementById(
            "goalMilestone"
        ).innerText =
            data.milestone || "";

        document.getElementById(
            "goalEta"
        ).innerText =
            data.eta || "";
    };

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

    updateSubcategories();

    document
        .getElementById("category")
        ?.addEventListener(
            "change",
            updateSubcategories
        );

    loadHistory();
    loadCharts();
    loadBudget();
    loadInsights();
    loadForecast();
    loadRecommendations();
    loadHealthScore();
    loadTopSubcategory();
    loadSubcategoryBreakdown();
    loadGoal();
    loadMonthlyComparison();
});