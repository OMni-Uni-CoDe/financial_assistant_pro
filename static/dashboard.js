document.addEventListener("DOMContentLoaded", () => {

    const csrfToken =
        document.querySelector(
            'meta[name="csrf-token"]'
        )?.content;

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

    const askForm = document.getElementById("askForm");
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
    // Add Budget
    // =========================

    if (budgetForm) {

        budgetForm.addEventListener(
            "submit",
            async (e) => {

                e.preventDefault();

                const formData =
                    new FormData(budgetForm);

                const res =
                    await fetch(
                        "/budget",
                        {
                            method: "POST",

                            headers: {
                                "X-CSRFToken": csrfToken
                            },

                            body: formData
                        }
                    );

                window.location.reload();

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
                "/download_monthly_csv";

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
                "/download_monthly_pdf";

        });

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

                loadCharts();

            }
        );

    }

    if (periodFilter) {

        periodFilter.addEventListener(
            "change",
            () => {

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

            if (
                !document.getElementById("budgetAmount")
            ) {
                return;
            }

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

            const dashboardBudget =
                document.getElementById(
                    "budgetRemaining"
                );

            if (dashboardBudget) {
                dashboardBudget.innerText =
                    `₹${data.remaining}`;
            }

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

    // =============
    // LOAD GOALS
    // =============

    async function loadGoalProgress() {
        try {

            const response =
                await fetch(
                    "/get_goal_progress"
                );

            const data =
                await response.json();

            const card =
                document.getElementById(
                    "goalProgress"
                );

            if (card) {
                card.innerText =
                    data.progress + "%";
            }

        }

        catch (error) {
            console.error(
                "Goal progress error:",
                error
            );
        }
    }


    async function loadGoalDetails() {

        try {

            const response =
                await fetch("/get_goal_details");

            const data =
                await response.json();

            const name =
                document.getElementById(
                    "goalDisplayName"
                );

            if (!name) return;

            name.innerText =
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
                data.progress + "%";

            document.getElementById(
                "goalProgressBar"
            ).style.width =
                Math.min(
                    data.progress,
                    100
                ) + "%";

            const eta =
                document.getElementById(
                    "goalEta"
                );

            if (eta) {

                eta.innerText =
                    `₹${data.remaining.toFixed(2)}
                    remaining to reach goal`;

            }

            const milestone =
                document.getElementById(
                    "goalMilestone"
                );

            if (data.progress >= 100)
                milestone.innerText =
                    "🏆 Goal Completed!";

            else if (data.progress >= 75)
                milestone.innerText =
                    "🔥 Almost there!";

            else if (data.progress >= 50)
                milestone.innerText =
                    "🚀 Halfway achieved!";

            else if (data.progress >= 25)
                milestone.innerText =
                    "✅ Great start!";

            else
                milestone.innerText =
                    "🎯 Keep saving!";


        }

        catch (error) {

            console.error(
                "Goal details error:",
                error
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

            if (!container) return;

            container.innerHTML = "";

            data.recommendations.forEach(
                rec => {

                    const card =
                        document.createElement(
                            "div"
                        );

                    card.className = "recommendation-card";

                    let badge = "INFO";

                    if (rec.title.includes("⚠"))
                        badge = "HIGH PRIORITY";

                    else if (rec.title.includes("✅"))
                        badge = "GOOD";

                    else if (rec.title.includes("🎯"))
                        badge = "GOAL";

                    card.innerHTML = `

                    <div class="recommendation-badge">
                        ${badge}
                    </div>

                    <h3>${rec.title}</h3>

                    <p>${rec.message}</p>

                    `;

                    container.appendChild(
                        card
                    );

                }
            );

            const total =
                data.recommendations.length;

            const highPriority =
                data.recommendations.filter(
                    r => r.title.includes("⚠")
                ).length;

            if (
                document.getElementById(
                    "recommendationCount"
                )
            ) {
                document.getElementById(
                    "recommendationCount"
                ).innerText = total;
            }

            if (
                document.getElementById(
                    "highPriorityCount"
                )
            ) {
                document.getElementById(
                    "highPriorityCount"
                ).innerText = highPriority;
            }

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

            if (
                !scoreElement ||
                !ratingElement
            ) {
                return;
            }

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

            const topSubcategory =
                document.getElementById(
                    "topSubcategory"
                );

            if (topSubcategory) {

                topSubcategory.innerText =
                    data.subcategory;

            }

            const topSubcategoryInfo =
                document.getElementById(
                    "topSubcategoryInfo"
                );

            if (topSubcategoryInfo) {

                topSubcategoryInfo.innerText =
                    `${data.category} • ₹${data.amount}
                    (${data.percentage}% of spending)`;

            }

        }

        catch (err) {

            console.error(
                "Top subcategory error:",
                err
            );

        }

    }


    async function loadTrendChart() {
        const response =
            await fetch("/get_data");

        const data =
            await response.json();

        if (data.daily_totals.length > 0) {

            const maxValue = Math.max(...data.daily_totals);
            const minValue = Math.min(...data.daily_totals);

            const avgValue =
                data.daily_totals.reduce(
                    (a, b) => a + b,
                    0
                ) / data.daily_totals.length;

            const maxIndex =
                data.daily_totals.indexOf(maxValue);

            const minIndex =
                data.daily_totals.indexOf(minValue);

            const highestMonth =
                document.getElementById(
                    "highestMonth"
                );

            const lowestMonth =
                document.getElementById(
                    "lowestMonth"
                );

            const averageMonth =
                document.getElementById(
                    "averageMonth"
                );

            if (highestMonth) {
                highestMonth.innerText =
                    `${data.dates[maxIndex]}
             (₹${maxValue.toFixed(0)})`;
            }

            if (lowestMonth) {
                lowestMonth.innerText =
                    `${data.dates[minIndex]}
             (₹${minValue.toFixed(0)})`;
            }

            if (averageMonth) {
                averageMonth.innerText =
                    `₹${avgValue.toFixed(0)}`;
            }
        }

        const ctx =
            document
                .getElementById("trendChart");

        if (!ctx) return;

        new Chart(ctx, {
            type: "line",
            data: {
                labels: data.dates,
                datasets: [{
                    label: "Daily Spending",
                    data: data.daily_totals
                }]
            }
        });
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

                const canvas =
                    document.createElement(
                        "canvas"
                    );

                canvas.height = 180;

                body.appendChild(canvas);

                const labels = [];
                const values = [];

                data[category].forEach(item => {

                    labels.push(
                        item.subcategory
                    );

                    values.push(
                        item.amount
                    );

                    const row =
                        document.createElement(
                            "div"
                        );

                    row.className =
                        "drilldown-row";

                    row.innerHTML = `
                        <span>
                            ${item.subcategory}
                        </span>

                        <span>
                            ₹${item.amount}
                        </span>
                    `;

                    body.appendChild(row);

                });

                new Chart(canvas, {

                    type: "bar",

                    data: {

                        labels: labels,

                        datasets: [{

                            label: category,

                            data: values

                        }]

                    },

                    options: {

                        responsive: true,

                        plugins: {

                            legend: {
                                display: false
                            }

                        }

                    }

                });

                header.addEventListener(
                    "click",
                    () => {

                        const open =
                            body.style.display === "block";

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


    async function loadAnalyticsBreakdown() {

        const response =
            await fetch(
                "/get_subcategory_breakdown"
            );

        const data =
            await response.json();

        const container =
            document.getElementById(
                "analyticsBreakdown"
            );

        if (!container) return;

        container.innerHTML = "";

        Object.keys(data).forEach(category => {

            let total = 0;

            data[category].forEach(item => {
                total += item.amount;
            });

            let html = `
            <div class="breakdown-card">

                <h4>${category}</h4>

                <p>
                    Total Spending:
                    ₹${total.toFixed(2)}
                </p>

                <ul>
        `;

            data[category].forEach(item => {

                html += `
                <li>
                    ${item.subcategory}
                    :
                    ₹${item.amount}
                </li>
            `;

            });

            html += `
                </ul>

            </div>
        `;

            container.innerHTML += html;

        });

    }


    async function loadHistory() {
        const response =
            await fetch("/history");

        const data =
            await response.json();

        const container =
            document.getElementById(
                "expenseHistory"
            );

        if (!container) return;

        container.innerHTML = "";

        data.history.forEach(
            item => {

                container.innerHTML += `
            <div class="history-item">

                ${item.date}

                -
                ${item.category}

                -

                ₹${item.amount}

            </div>
            `;
            }
        );
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

    // Dashboard

    if (
        document.getElementById("totalSpent")
    ) {

        loadCharts();
        loadInsights();
        loadBudget();
        loadGoalProgress();
        loadGoalDetails();
        loadRecommendations();
        loadHealthScore();
        loadTopSubcategory();
        loadMonthlyComparison();
        loadHistory();
    }

    // Budget Page

    if (
        document.getElementById("budgetAmount")
    ) {

        loadBudget();

    }

    // Forecast Page

    if (
        document.getElementById("forecastContainer")
    ) {

        loadForecast();

    }

    // Goals Page

    if (
        document.getElementById("goalDisplayName")
    ) {

        loadGoalDetails();

    }

    // Recommendations Page

    if (
        document.getElementById(
            "recommendationsContainer"
        )
    ) {

        loadRecommendations();

    }

    // Monthly Trend Page

    if (
        document.getElementById(
            "trendChart"
        )
    ) {

        loadTrendChart();

    }

    // Category Breakdown Page

    if (
        document.getElementById(
            "subcategoryBreakdownContainer"
        ) &&

        !document.getElementById(
            "totalSpent"
        )
    ) {

        loadSubcategoryBreakdown();

    }

    // Analytics Breakdown Page

    if (document.getElementById("analyticsBreakdown")) {
        loadAnalyticsBreakdown();
    }

});