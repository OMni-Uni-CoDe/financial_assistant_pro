# app.py
import os
import re
import io
from markupsafe import Markup
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, send_file, flash, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, UserMixin, current_user)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
from openai import OpenAI
from fpdf import FPDF
from sklearn.linear_model import LinearRegression



# ---------- App / Config ----------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-fallback-secret")

db_url = os.environ.get("DATABASE_URL")

if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///local.db"


# Mail settings for email confirmation
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER", "")
app.config["MAIL_PORT"] = int(os.environ.get("MAIL_PORT", 587))
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS", "true").lower() in ("true", "1", "yes")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER", app.config["MAIL_USERNAME"])

# Optional: short token expiry minutes for confirmation links
CONFIRM_TOKEN_EXP_MIN = int(os.environ.get("CONFIRM_TOKEN_EXP_MIN", 60))    

# ---------- Extensions ----------
db = SQLAlchemy(app)
csrf = CSRFProtect(app)
app.config["WTF_CSRF_ENABLED"] = True
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
limiter = Limiter(app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# OpenAI
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# ---------- Models ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=True)
    confirmed = db.Column(db.Boolean, default=False)
    confirm_token = db.Column(db.String(255), nullable=True)
    confirm_sent_at = db.Column(db.DateTime, nullable=True)

    budget = db.Column(db.Float, default=0)

    expenses = db.relationship("Expense", backref="user", lazy=True)


class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow)
    category = db.Column(db.String(120))
    amount = db.Column(db.Float)


class SavingsGoal(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    goal_name = db.Column(
        db.String(150),
        nullable=False
    )

    target_amount = db.Column(
        db.Float,
        nullable=False
    )

    current_amount = db.Column(
        db.Float,
        default=0
    )


@app.route("/init_db")
def init_db():
    try:
        with app.app_context():
            db.create_all()
        return "Database initialized successfully!"
    except Exception as e:
        return str(e), 500

@app.route("/upgrade_budget")
def upgrade_budget():

    try:

        with db.engine.connect() as conn:

            conn.exec_driver_sql(
                'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS budget FLOAT DEFAULT 0'
            )

            conn.commit()

        return "Budget column added successfully."

    except Exception as e:

        return str(e), 500


# ---------- Helpers ----------
USERNAME_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{3,}$")
PASSWORD_RE = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z\d]).{8,}$")

def is_valid_username(u: str) -> bool:
    return bool(USERNAME_RE.match(u))

def is_strong_password(p: str) -> bool:
    return bool(PASSWORD_RE.match(p))

def generate_confirmation_token(username: str) -> str:
    # simple token: username + timestamp hashed - keep lightweight
    import hashlib, time
    payload = f"{username}-{int(time.time())}"
    return hashlib.sha256(payload.encode()).hexdigest()

def token_is_valid(user: User) -> bool:
    if not user.confirm_sent_at: return False
    return datetime.utcnow() - user.confirm_sent_at <= timedelta(minutes=CONFIRM_TOKEN_EXP_MIN)

# simple decorator to require confirmed email
def confirmed_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user.confirmed:
            flash("Please confirm your email to access this page.", "warning")
            return redirect(url_for("unconfirmed"))
        return f(*a, **kw)
    return wrapper

# ---------- Forms ----------
class DummyForm(FlaskForm):
    dummy = StringField("dummy")  # used if you want CSRF on simple endpoints

# ---------- Login ----------
@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ---------- Routes: auth ----------
@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("10 per hour")
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        email = request.form.get("email", "").strip() or None

        if not is_valid_username(username):
            return render_template("signup.html", error="Username must contain letters and numbers (min 3).")
        if not is_strong_password(password):
            return render_template("signup.html", error="Password must be 8+ chars with upper, lower, number, special.")

        if User.query.filter_by(username=username).first():
            return render_template("signup.html", error="Username already exists.")
        if email and User.query.filter_by(email=email).first():
            return render_template("signup.html", error="Email already used.")

        hashed = generate_password_hash(password)
        user = User(username=username, password=hashed, email=email)
        # create confirmation token if email provided
        if email:
            token = generate_confirmation_token(username)
            user.confirm_token = token
            user.confirm_sent_at = datetime.utcnow()
            user.confirmed = False
        else:
            user.confirmed = True  # no email -> mark confirmed (option)

        db.session.add(user)
        db.session.commit()

        # send confirmation email if email set
        if email:
            send_confirmation_email(user)
            return render_template("signup.html", message="Account created. Check your email for confirmation.")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/check_username")
def check_username():
    username = request.args.get("username", "").strip()
    if not username:
        return jsonify({"available": False, "reason": "empty"})
    exists = User.query.filter_by(username=username).first() is not None
    return jsonify({"available": not exists})

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("30 per hour")
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            return render_template("login.html", error="Invalid credentials.")
        login_user(user)
        return redirect(url_for("splash"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------- Email confirmation ----------
def send_confirmation_email(user: User):
    if not user.email or not app.config["MAIL_USERNAME"]:
        return
    token = user.confirm_token or generate_confirmation_token(user.username)
    user.confirm_token = token
    user.confirm_sent_at = datetime.utcnow()
    db.session.commit()
    confirm_url = url_for("confirm_email", token=token, _external=True)
    html = render_template("emails/confirm.html", confirm_url=confirm_url, username=user.username)
    msg = Message("Confirm your Financial Assistant account", recipients=[user.email], html=html)
    mail.send(msg)

@app.route("/confirm/<token>")
def confirm_email(token):
    user = User.query.filter_by(confirm_token=token).first_or_404()
    if not token_is_valid(user):
        return "Confirmation link expired. Request a new confirmation.", 400
    user.confirmed = True
    user.confirm_token = None
    user.confirm_sent_at = None
    db.session.commit()
    return render_template("confirm_done.html")

@app.route("/unconfirmed")
@login_required
def unconfirmed():
    if current_user.confirmed:
        return redirect(url_for("dashboard"))
    return render_template("unconfirmed.html")

@app.route("/resend_confirmation")
@login_required
def resend_confirmation():
    send_confirmation_email(current_user)
    flash("Confirmation email resent.", "info")
    return redirect(url_for("unconfirmed"))

@app.route("/history")
@login_required
def history():

    category = request.args.get("category", "all")
    period = request.args.get("period", "all")

    query = Expense.query.filter_by(
        user_id=current_user.id
    )

    today = datetime.utcnow().date()

    if period == "7days":
        query = query.filter(
            Expense.date >= today - timedelta(days=7)
        )

    elif period == "30days":
        query = query.filter(
            Expense.date >= today - timedelta(days=30)
        )

    elif period == "month":
        query = query.filter(
            Expense.date >= today.replace(day=1)
        )

    if category != "all":
        query = query.filter(
            Expense.category == category
        )

    expenses = (
        query
        .order_by(Expense.date.desc())
        .limit(20)
        .all()
    )

    return jsonify({
        "history": [
            {
                "id": e.id,
                "date": str(e.date),
                "category": e.category,
                "amount": e.amount
            }
            for e in expenses
        ]
    })

# ---------- Splash (quote) ----------
@app.route("/splash")
@login_required
def splash():
    import random

    # folder path in static
    bg_folder = os.path.join(app.static_folder, "backgrounds")

    # read all JPG/JPEG/PNG files
    files = [
        f for f in os.listdir(bg_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        # fallback color or image
        bg_url = url_for("static", filename="backgrounds/default.jpg")
    else:
        chosen = random.choice(files)
        bg_url = url_for("static", filename=f"backgrounds/{chosen}")

    # random motivational quotes
    quotes = [
        {"text": "An investment in knowledge pays the best interest. — Benjamin Franklin"},
        {"text": "Money is a terrible master but an excellent servant. — P.T. Barnum"},
        {"text": "Do not save what is left after spending; spend what is left after saving. — Warren Buffett"},
        {"text": "Beware of little expenses; a small leak will sink a great ship. — Benjamin Franklin"},
        {"text": "A budget tells us what we can't afford, but it doesn't keep us from buying it. — William Feather"}
    ]

    quote = random.choice(quotes)

    return render_template(
        "splash.html",
        quote=quote,
        delay_ms=5000,
        bg=bg_url
    )

# ---------- Dashboard ----------
@app.route("/")
@login_required
def home_redirect():
    return redirect(url_for("splash"))

@app.route("/dashboard")
@login_required
@confirmed_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

# ---------- Expense endpoints ----------
@app.route("/add_expense", methods=["POST"])
@login_required
@limiter.limit("60 per hour")
def add_expense():
    category = request.form.get("category", "Other")
    try:
        amount = float(request.form.get("amount", 0))
    except ValueError:
        return jsonify({"message": "Invalid amount."}), 400
    entry = Expense(user_id=current_user.id, category=category, amount=amount, date=datetime.utcnow().date())
    db.session.add(entry)
    db.session.commit()
    return jsonify({"message": "Expense added successfully."})

# --------- Delete an expense ---------
@app.route("/delete_expense/<int:expense_id>", methods=["POST"])
@csrf.exempt
@login_required
def delete_expense(expense_id):

    expense = Expense.query.filter_by(
        id=expense_id,
        user_id=current_user.id
    ).first()

    if not expense:
        return jsonify({
            "success": False,
            "message": "Expense not found."
        }), 404

    db.session.delete(expense)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Expense deleted successfully."
    })

# --------- Catagory for sorting expenses ---------
@app.route("/get_data")
@login_required
def get_data():

    category = request.args.get("category", "all")
    period = request.args.get("period", "all")

    query = Expense.query.filter_by(
        user_id=current_user.id
    )

    today = datetime.utcnow().date()

    if period == "7days":
        query = query.filter(
            Expense.date >= today - timedelta(days=7)
        )

    elif period == "30days":
        query = query.filter(
            Expense.date >= today - timedelta(days=30)
        )

    elif period == "month":
        query = query.filter(
            Expense.date >= today.replace(day=1)
        )

    if category != "all":
        query = query.filter(
            Expense.category == category
        )

    expenses = query.order_by(
        Expense.date.asc()
    ).all()

    if not expenses:
        return jsonify({
            "categories": [],
            "totals": [],
            "dates": [],
            "daily_totals": []
        })

    df = pd.DataFrame([
        {
            "date": e.date,
            "category": e.category,
            "amount": e.amount
        }
        for e in expenses
    ])

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .to_dict()
    )

    daily = (
        df.groupby("date")["amount"]
        .sum()
        .to_dict()
    )

    return jsonify({
        "categories": list(category_totals.keys()),
        "totals": list(category_totals.values()),
        "dates": [str(d) for d in daily.keys()],
        "daily_totals": list(daily.values())
    })

# --------- Budget creation ---------
@app.route("/get_budget")
@login_required
def get_budget():

    total_spent = db.session.query(
        db.func.sum(Expense.amount)
    ).filter_by(
        user_id=current_user.id
    ).scalar() or 0

    budget = current_user.budget or 0

    remaining = budget - total_spent

    percentage = 0

    if budget > 0:
        percentage = min(
            round((total_spent / budget) * 100, 2),
            100
        )

    return jsonify({
        "budget": budget,
        "spent": total_spent,
        "remaining": remaining,
        "percentage": percentage
    })

# --------- Expense insights ---------
@app.route("/get_insights")
@login_required
def get_insights():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "insights": [
                "No expenses available yet."
            ]
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    insights = []

    top_category = category_totals.index[0]

    top_amount = category_totals.iloc[0]

    percentage = (
        top_amount / total_spent
    ) * 100

    insights.append(
        f"{top_category} accounts for {percentage:.1f}% of your spending."
    )

    insights.append(
        f"Your largest expense category is {top_category}."
    )

    if current_user.budget and current_user.budget > 0:

        usage = (
            total_spent / current_user.budget
        ) * 100

        insights.append(
            f"You have used {usage:.1f}% of your budget."
        )

        if usage >= 80:

            insights.append(
                "You are approaching your budget limit."
            )

    average_expense = (
        df["amount"].mean()
    )

    insights.append(
        f"Average expense amount is ₹{average_expense:.2f}."
    )

    return jsonify({
        "insights": insights
    })

# --------- End of month spending forecast ---------
@app.route("/get_forecast")
@login_required
def get_forecast():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "forecast": "No spending data available."
        })

    total_spent = sum(
        e.amount for e in expenses
    )

    today = datetime.utcnow().date()

    days_passed = max(today.day, 1)

    projected = (
        total_spent / days_passed
    ) * 30

    message = (
        f"Projected month-end spending: "
        f"₹{projected:.2f}"
    )

    budget = current_user.budget or 0

    if budget > 0:

        difference = projected - budget

        if difference > 0:

            message += (
                f"\n⚠ You may exceed your budget "
                f"by ₹{difference:.2f}"
            )

        else:

            message += (
                f"\n✅ You are likely to remain "
                f"within budget."
            )

    return jsonify({
        "forecast": message
    })

# --------- AI Recommendations ---------
@app.route("/get_recommendations")
@login_required
def get_recommendations():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "recommendations": [
                "Add some expenses to receive recommendations."
            ]
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    recommendations = []

    # ==================================
    # Recommendation 1:
    # Saving Opportunity
    # ==================================

    top_category = category_totals.index[0]
    top_amount = category_totals.iloc[0]

    annual_saving = (
        top_amount * 0.10 * 12
    )

    recommendations.append(
        f"Reducing {top_category} expenses by 10% could save approximately ₹{annual_saving:.0f} annually."
    )

    # ==================================
    # Recommendation 2:
    # Daily Budget Target
    # ==================================

    if current_user.budget and current_user.budget > 0:

        budget = current_user.budget

        remaining_budget = (
            budget - total_spent
        )

        today = datetime.utcnow().date()

        days_remaining = max(
            30 - today.day,
            1
        )

        daily_limit = (
            remaining_budget /
            days_remaining
        )

        recommendations.append(
            f"To remain within budget, keep spending below approximately ₹{daily_limit:.0f} per day."
        )

    # ==================================
    # Recommendation 3:
    # Forecast Warning
    # ==================================

    today = datetime.utcnow().date()

    days_passed = max(
        today.day,
        1
    )

    projected_spending = (
        total_spent / days_passed
    ) * 30

    if current_user.budget and current_user.budget > 0:

        excess = (
            projected_spending -
            current_user.budget
        )

        if excess > 0:

            recommendations.append(
                f"At your current pace, you may exceed your budget by approximately ₹{excess:.0f} this month."
            )

    # ==================================
    # Recommendation 4:
    # Spending Concentration
    # ==================================

    if len(category_totals) >= 2:

        top_two_total = (
            category_totals.iloc[0]
            + category_totals.iloc[1]
        )

        top_two_percent = (
            top_two_total /
            total_spent
        ) * 100

        recommendations.append(
            f"The top two categories account for {top_two_percent:.1f}% of your spending. Review recurring expenses for possible savings."
        )

    return jsonify({
        "recommendations": recommendations
    })

# --------- Financial Health Score ---------
@app.route("/get_health_score")
@login_required
def get_health_score():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "score": 100,
            "rating": "Excellent",
            "message": "No spending recorded yet."
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    score = 0

    # ==========================
    # Budget Usage (40 pts)
    # ==========================

    budget = current_user.budget or 0

    if budget > 0:

        usage = (
            total_spent / budget
        ) * 100

        if usage <= 50:
            score += 40

        elif usage <= 80:
            score += 30

        elif usage <= 100:
            score += 15

    else:
        score += 20

    # ==========================
    # Forecast Risk (30 pts)
    # ==========================

    today = datetime.utcnow().date()

    projected = (
        total_spent /
        max(today.day, 1)
    ) * 30

    if budget > 0:

        if projected <= budget:
            score += 30

        elif projected <= budget * 1.1:
            score += 15

    else:
        score += 15

    # ==========================
    # Spending Concentration
    # ==========================

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    top_share = (
        category_totals.iloc[0]
        / total_spent
    ) * 100

    if top_share < 40:
        score += 30

    elif top_share < 60:
        score += 20

    else:
        score += 10

    # ==========================
    # Rating
    # ==========================

    if score >= 85:
        rating = "Excellent"

    elif score >= 70:
        rating = "Good"

    elif score >= 50:
        rating = "Fair"

    else:
        rating = "Needs Attention"

    return jsonify({
        "score": round(score),
        "rating": rating,
        "message":
            f"Financial health is currently rated as {rating}."
    })

# --------- Budget creation ---------
@app.route("/set_budget", methods=["POST"])
@login_required
def set_budget():

    try:

        amount = float(
            request.form.get("budget", 0)
        )

        current_user.budget = amount

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Budget updated."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 400

# ---------- CSV / PDF export ----------
@app.route("/download_csv")
@login_required
def download_csv():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    df = pd.DataFrame([{"Date": e.date, "Category": e.category, "Amount": e.amount} for e in expenses])
    out = io.BytesIO()
    df.to_csv(out, index=False)
    out.seek(0)
    return send_file(out, mimetype="text/csv", as_attachment=True, download_name="expenses.csv")

@app.route("/download_pdf")
@login_required
def download_pdf():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Expense Report", ln=True, align="C")
    pdf.ln(5)

    total = 0

    for e in expenses:
        total += e.amount
        pdf.cell(
            0,
            8,
            f"{e.date} - {e.category}: Rs. {e.amount:.2f}",
            ln=True
        )

    pdf.ln(5)
    pdf.cell(0, 10, f"Total: Rs. {total:.2f}", ln=True)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="expenses.pdf"
    )
# ---------- AI assistant ----------
@app.route("/ask", methods=["POST"])
@login_required
@limiter.limit("40 per hour")
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Ask a valid question."})

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:
        return jsonify({
            "answer": "No expenses found to analyze."
        })

    df = pd.DataFrame([
        {
            "date": e.date,
            "category": e.category,
            "amount": e.amount
        }
        for e in expenses
    ])

    summary = (
        df.groupby("category")["amount"]
        .sum()
        .to_dict()
    )

    total_spent = df["amount"].sum()

    prompt = f"""
User spending summary:

Total spent: Rs. {total_spent:.2f}

Category totals:
{summary}

User question:
{question}

Give a short practical financial answer.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful personal finance assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=250
        )

        answer = response.choices[0].message.content

        return jsonify({
            "answer": answer
        })

    except Exception:
        highest = max(summary, key=summary.get)

        return jsonify({
            "answer":
            f"Total spending: Rs. {total_spent:.2f}. "
            f"Highest category: {highest} "
            f"(Rs. {summary[highest]:.2f})."
        })
# ---------- Prediction ----------
@app.route("/predict_future")
@login_required
def predict_future():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    if len(expenses) < 2:
        return jsonify({"prediction": "Not enough data for prediction."})
    df = pd.DataFrame([{"date": e.date, "amount": e.amount} for e in expenses])
    df["date"] = pd.to_datetime(df["date"])
    df["day_index"] = (df["date"] - df["date"].min()).dt.days
    model = LinearRegression()
    model.fit(df[["day_index"]], df["amount"])
    next_day = np.array([[df["day_index"].max() + 1]])
    predicted = model.predict(next_day)[0]
    return jsonify({"prediction": f"Estimated spending tomorrow: Rs. {predicted:.2f}"
})

# ---------- Utility: health ----------
@app.route("/health")
def health():
    return "ok", 200

# ---------- Run ----------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=(os.environ.get("FLASK_DEBUG") == "1")
    )