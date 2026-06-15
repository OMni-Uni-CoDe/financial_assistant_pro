# app.py
import os
import re
import io
from markupsafe import Markup
from datetime import datetime, timedelta
from functools import wraps
from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, send_file, flash, abort, Response, make_response)
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
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
import numpy as np
from openai import OpenAI
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
import csv



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

# ==================================================
# DATABASE MODELS
# ==================================================
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
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    date = db.Column(
        db.Date,
        default=datetime.utcnow
    )

    category = db.Column(
        db.String(120)
    )

    subcategory = db.Column(
        db.String(120),
        default=""
    )

    amount = db.Column(
        db.Float
    )


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


class FinancePDF(FPDF):

    def footer(self):

        self.set_y(-15)

        self.set_font(
            "Arial",
            "I",
            8
        )

        self.set_text_color(
            120,
            120,
            120
        )

        self.cell(
            0,
            10,
            f"Finance Pro Confidential | Page {self.page_no()}",
            align="C"
        )

        self.set_text_color(
            0,
            0,
            0
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


@app.route("/upgrade_subcategory")
def upgrade_subcategory():

    try:

        with db.engine.connect() as conn:

            conn.exec_driver_sql(
                """
                ALTER TABLE expense
                ADD COLUMN IF NOT EXISTS subcategory VARCHAR(120)
                """
            )

            conn.commit()

        return "Subcategory column added successfully."

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

# ==================================================
# AUTHENTICATION ROUTES
# ==================================================
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
    "subcategory": e.subcategory,
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

# ==================================================
# DASHBOARD ROUTES
# ==================================================
@app.route("/")
@login_required
def home_redirect():
    return redirect(url_for("splash"))

@app.route("/dashboard")
@login_required
@confirmed_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

# ==================================================
# EXPENSE MANAGEMENT
# ==================================================
@app.route("/add_expense", methods=["POST"])
@login_required
@limiter.limit("60 per hour")
def add_expense():
    category = request.form.get("category", "Other")
    subcategory = request.form.get(
    "subcategory",
    ""
)
    try:
        amount = float(request.form.get("amount", 0))
    except ValueError:
        return jsonify({"message": "Invalid amount."}), 400
    entry = Expense(
            user_id=current_user.id,
            category=category,
            subcategory=subcategory,
            amount=amount,
            date=datetime.utcnow().date()
    )
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
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .to_dict()
    )

    daily_totals = (
        df.groupby("date")["amount"]
        .sum()
        .to_dict()
    )

    return jsonify({
        "categories": list(category_totals.keys()),
        "totals": list(category_totals.values()),
        "dates": [str(d) for d in daily_totals.keys()],
        "daily_totals": list(daily_totals.values())
    })

@app.route("/get_top_subcategory")
@login_required
def get_top_subcategory():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "category": "-",
            "subcategory": "-",
            "amount": 0,
            "percentage": 0
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    subcategory_totals = (
        df.groupby(
            ["category", "subcategory"]
        )["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_entry = subcategory_totals.index[0]

    top_amount = (
        subcategory_totals.iloc[0]
    )

    percentage = (
        top_amount /
        total_spent
    ) * 100

    return jsonify({

        "category":
            top_entry[0],

        "subcategory":
            top_entry[1],

        "amount":
            round(top_amount, 2),

        "percentage":
            round(percentage, 1)

    })

@app.route("/get_subcategory_breakdown")
@login_required
def get_subcategory_breakdown():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:
        return jsonify({})

    df = pd.DataFrame([
        {
            "category": e.category,
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    grouped = (
        df.groupby(
            ["category", "subcategory"]
        )["amount"]
        .sum()
        .reset_index()
    )

    result = {}

    for category in grouped["category"].unique():

        category_rows = grouped[
            grouped["category"] == category
        ]

        result[category] = [

            {
                "subcategory": row["subcategory"],
                "amount": round(
                    row["amount"],
                    2
                )
            }

            for _, row in category_rows.iterrows()

        ]

    return jsonify(result)



# ==================================================
# BUDGET MANAGEMENT
# ==================================================
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

# ==================================================
# ANALYTICS & INSIGHTS
# ==================================================
@app.route("/get_insights")
@login_required
def get_insights():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "insights": []
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    insights = []

    # ==========================
    # Spending Concentration
    # ==========================

    subcategory_totals = (
        df.groupby(
            ["category", "subcategory"]
        )["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_entry = subcategory_totals.index[0]

    top_amount = subcategory_totals.iloc[0]

    top_percent = (
        top_amount /
        total_spent
    ) * 100

    insights.append({

        "title":
            "📊 Spending Concentration",

        "message":
            f"{top_percent:.1f}% of spending is in "
            f"{top_entry[0]} -> {top_entry[1]}"

    })

    # ==========================
    # Budget Status
    # ==========================

    if (
        current_user.budget and
        current_user.budget > 0
    ):

        usage = (
            total_spent /
            current_user.budget
        ) * 100

        if usage >= 80:

            message = (
                f"⚠ {usage:.1f}% of monthly budget used."
            )

        else:

            message = (
                f"{usage:.1f}% of monthly budget used."
            )

        insights.append({

            "title":
                "📈 Budget Status",

            "message":
                message

        })

    # ==========================
    # Daily Average
    # ==========================

    today = datetime.utcnow().date()

    daily_average = (
        total_spent /
        max(today.day, 1)
    )

    insights.append({

        "title":
            "💰 Daily Average",

        "message":
            f"Rs. {daily_average:.0f}/day"

    })

    # ==========================
    # Largest Expense
    # ==========================

    insights.append({

        "title":
            "🏆 Largest Expense",

        "message":
            f"{top_entry[0]} -> {top_entry[1]}"

    })

    # ==========================
    # Savings Goal
    # ==========================

    goal = SavingsGoal.query.filter_by(
        user_id=current_user.id
    ).first()

    if goal and goal.target_amount > 0:

        percentage = round(
            (
                goal.current_amount /
                goal.target_amount
            ) * 100,
            1
        )

        insights.append({

            "title":
                "🎯 Savings Goal",

            "message":
                f"{goal.goal_name} is "
                f"{percentage}% complete"

        })

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
        f"Rs. {projected:.2f}"
    )

    budget = current_user.budget or 0

    if budget > 0:

        difference = projected - budget

        if difference > 0:

            message += (
                f"\n⚠ You may exceed your budget "
                f"by Rs. {difference:.2f}"
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
                {
                    "title": "ℹ No Data",
                    "message":
                        "Add expenses to receive recommendations."
                }
            ]
        })

    df = pd.DataFrame([
        {
            "category": e.category,
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    recommendations = []

    today = datetime.utcnow().date()

    days_passed = max(
        today.day,
        1
    )

    projected_spending = (
        total_spent /
        days_passed
    ) * 30

    budget = current_user.budget or 0

    # ==========================
    # Top Expense
    # ==========================

    subcategory_totals = (
        df.groupby(
            ["category", "subcategory"]
        )["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_entry = subcategory_totals.index[0]

    top_amount = (
        subcategory_totals.iloc[0]
    )

    top_percent = (
        top_amount /
        total_spent
    ) * 100

    # ==========================
    # OVER BUDGET
    # ==========================

    if (
        budget > 0 and
        projected_spending > budget
    ):

        excess = (
            projected_spending -
            budget
        )

        recommendations.append({

            "title":
                "⚠ Budget Risk",

            "message":
                f"Projected overspend: "
                f"Rs. {excess:.0f}"

        })

        remaining_budget = (
            budget -
            total_spent
        )

        days_remaining = max(
            30 - today.day,
            1
        )

        daily_limit = (
            remaining_budget /
            days_remaining
        )

        recommendations.append({

            "title":
                "⚠ Daily Limit",

            "message":
                f"Keep spending below "
                f"Rs. {daily_limit:.0f}/day"

        })

        recommendations.append({

            "title":
                "⚠ Expense Reduction",

            "message":
                f"{top_entry[0]} -> "
                f"{top_entry[1]} consumes "
                f"{top_percent:.1f}% "
                f"of spending"

        })

    # ==========================
    # UNDER BUDGET
    # ==========================

    elif budget > 0:

        surplus = (
            budget -
            projected_spending
        )

        recommendations.append({

            "title":
                "✅ Budget Surplus",

            "message":
                f"Projected Rs. {surplus:.0f} "
                f"under budget"

        })

        days_remaining = max(
            30 - today.day,
            1
        )

        remaining_budget = (
            budget -
            total_spent
        )

        daily_limit = (
            remaining_budget /
            days_remaining
        )

        recommendations.append({

            "title":
                "✅ Safe Spending Limit",

            "message":
                f"You can spend "
                f"Rs. {daily_limit:.0f}/day"

        })

    # ==========================
    # Goal Acceleration
    # ==========================

    goal = SavingsGoal.query.filter_by(
        user_id=current_user.id
    ).first()

    if (
    goal and
    goal.target_amount > 0 and
    budget > 0
):

        remaining_amount = (
            goal.target_amount -
            goal.current_amount
        )

        current_monthly_saving = (
            budget * 0.20
        )

        improved_monthly_saving = (
            current_monthly_saving +
            1000
        )

        if current_monthly_saving > 0:

            current_eta = (
                remaining_amount /
                current_monthly_saving
            )

            improved_eta = (
                remaining_amount /
                improved_monthly_saving
            )

            months_saved = (
                current_eta -
                improved_eta
            )

            recommendations.append({

                "title":
                    "🎯 Goal Acceleration",

                "message":
                    f"Current ETA: "
                    f"{current_eta:.1f} months. "
                    f"Saving an extra Rs. 1000/month "
                    f"could reduce it to "
                    f"{improved_eta:.1f} months "
                    f"({months_saved:.1f} months faster)."

            })

    # ==========================
    # Savings Opportunity
    # ==========================

    if (
        budget > 0 and
        projected_spending < budget
    ):

        recommendations.append({

            "title":
                "💡 Savings Opportunity",

            "message":
                "Current spending leaves "
                "room for additional saving."

        })

    return jsonify({
        "recommendations":
            recommendations[:4]
    })

# --------- Monthly Comparison ---------
@app.route("/get_monthly_comparison")
@login_required
def get_monthly_comparison():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).all()

    if not expenses:

        return jsonify({
            "comparison":
            "Not enough data available."
        })

    today = datetime.utcnow().date()

    current_month = today.month
    current_year = today.year

    if current_month == 1:

        previous_month = 12
        previous_year = current_year - 1

    else:

        previous_month = current_month - 1
        previous_year = current_year

    current_total = sum(

        e.amount

        for e in expenses

        if e.date.month == current_month
        and e.date.year == current_year

    )

    previous_total = sum(

        e.amount

        for e in expenses

        if e.date.month == previous_month
        and e.date.year == previous_year

    )

    if previous_total == 0:

        return jsonify({
            "comparison":
            "No previous month data available."
        })

    percentage_change = (
        (
            current_total -
            previous_total
        )
        / previous_total
    ) * 100

    if percentage_change > 0:

        trend = (
            f"↑ {percentage_change:.1f}% "
            f"higher than last month"
        )

    elif percentage_change < 0:

        trend = (
            f"↓ {abs(percentage_change):.1f}% "
            f"lower than last month"
        )

    else:

        trend = (
            "No change from last month"
        )

    return jsonify({

        "current_month":
            round(current_total, 2),

        "previous_month":
            round(previous_total, 2),

        "trend":
            trend

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

# ==================================================
# SAVINGS GOALS
# ==================================================

@app.route("/set_goal", methods=["POST"])
@csrf.exempt
@login_required
def set_goal():

    goal_name = request.form.get(
        "goal_name"
    )

    target_amount = float(
        request.form.get(
            "target_amount",
            0
        )
    )

    current_amount = float(
        request.form.get(
            "current_amount",
            0
        )
    )

    goal = SavingsGoal.query.filter_by(
        user_id=current_user.id
    ).first()

    if not goal:

        goal = SavingsGoal(
            user_id=current_user.id
        )

        db.session.add(goal)

    goal.goal_name = goal_name
    goal.target_amount = target_amount
    goal.current_amount = current_amount

    db.session.commit()

    return jsonify({
        "message":
        "Savings goal saved."
    })


@app.route("/get_goal")
@login_required
def get_goal():

    goal = SavingsGoal.query.filter_by(
        user_id=current_user.id
    ).first()

    if not goal:

        return jsonify({
            "goal_name": "",
            "target_amount": 0,
            "current_amount": 0,
            "percentage": 0,
            "milestone": "",
            "eta": ""
        })

    percentage = 0

    if goal.target_amount > 0:

        percentage = round(
            (
                goal.current_amount /
                goal.target_amount
            ) * 100,
            1
        )

    remaining_amount =(
        goal.target_amount -
        goal.current_amount
    )

    monthly_saving = (
        current_user.budget * 0.20
        if current_user.budget and current_user.budget > 0
        else 0
    )

    # ==========================
    # ETA
    # ==========================

    if remaining_amount <= 0:

        eta = "🏆 Goal already achieved!"

    elif monthly_saving > 0:

        months_left = round(
            remaining_amount /
            monthly_saving,
            1
        )

        eta = (
            f"Estimated completion: "
            f"{months_left} month(s)"
        )

    else:

        eta = (
            "Set a budget to estimate "
            "goal completion."
        )

    # ==========================
    # Milestone
    # ==========================

    if percentage >= 100:

        milestone = "🏆 Goal Achieved!"

    elif percentage >= 75:

        milestone = "🎉 Almost There!"

    elif percentage >= 50:

        milestone = "🎉 Halfway There!"

    elif percentage >= 25:

        milestone = "🎉 25% Complete!"

    else:

        milestone = ""

    return jsonify({

        "goal_name":
            goal.goal_name,

        "target_amount":
            goal.target_amount,

        "current_amount":
            goal.current_amount,

        "percentage":
            percentage,

        "milestone":
            milestone,

        "eta":
            eta

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

# ==================================================
# REPORT EXPORTS
# ==================================================
@app.route("/download_csv")
@login_required
def download_csv():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Expense.date.desc()
    ).all()

    if not expenses:

        return jsonify({
            "error": "No expenses found."
        }), 404

    # ==========================
    # Expense Data
    # ==========================

    df = pd.DataFrame([
        {
            "Date": e.date,
            "Category": e.category,
            "Subcategory": e.subcategory,
            "Amount": e.amount,
            "Month": e.date.strftime("%B"),
            "Year": e.date.year
        }
        for e in expenses
    ])

    total_spent = df["Amount"].sum()

    budget = current_user.budget or 0

    remaining_budget = (
        budget - total_spent
    )

    # ==========================
    # Top Category
    # ==========================

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_category = (
        category_totals.index[0]
        if not category_totals.empty
        else "-"
    )

    # ==========================
    # Top Subcategory
    # ==========================

    subcategory_totals = (
        df.groupby(
            ["Category", "Subcategory"]
        )["Amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    if not subcategory_totals.empty:

        top_subcategory = (
            f"{subcategory_totals.index[0][0]}"
            f" -> "
            f"{subcategory_totals.index[0][1]}"
        )

    else:

        top_subcategory = "-"

    # ==========================
    # Forecast
    # ==========================

    today = datetime.utcnow().date()

    days_passed = max(
        today.day,
        1
    )

    forecast = round(
        (
            total_spent /
            days_passed
        ) * 30,
        2
    )

    # ==========================
    # Health Score
    # ==========================

    if budget > 0:

        usage = (
            total_spent /
            budget
        ) * 100

        if usage <= 50:
            health_score = 90

        elif usage <= 80:
            health_score = 70

        else:
            health_score = 50

    else:

        health_score = 0

    # ==========================
    # Goal Progress
    # ==========================

    goal = SavingsGoal.query.filter_by(
        user_id=current_user.id
    ).first()

    goal_progress = "N/A"

    if (
        goal and
        goal.target_amount > 0
    ):

        percentage = round(
            (
                goal.current_amount /
                goal.target_amount
            ) * 100,
            1
        )

        goal_progress = (
            f"{percentage}%"
        )

    # ==========================
    # Budget Usage %
    # ==========================

    if budget > 0:

        df["Budget Usage %"] = (
            (
                df["Amount"] /
                budget
            ) * 100
        ).round(2)

    else:

        df["Budget Usage %"] = 0

    # ==========================
    # Build Report
    # ==========================

    report_rows = [

        ["FINANCE PRO PROFESSIONAL REPORT", ""],
        ["Generated On", today.strftime("%d %B %Y")],
        ["User", current_user.username],
        ["Currency", "INR"],
        ["", ""],

        ["REPORT SUMMARY", ""],

        ["Total Spending", round(total_spent, 2)],
        ["Budget", budget],
        ["Remaining Budget", round(remaining_budget, 2)],
        ["Financial Health Score", f"{health_score}/100"],

        ["Top Category", top_category],
        ["Top Subcategory", top_subcategory],

        ["Projected Month-End Spending", forecast],
        ["Expense Count", len(df)],

        [
            "Budget Status",
            "Within Budget"
            if forecast <= budget
            else "Over Budget"
        ],

        ["", ""]
    ]

    if goal:

        report_rows.extend([

            ["SAVINGS GOAL", ""],

            ["Goal Name", goal.goal_name],
            ["Target Amount", goal.target_amount],
            ["Current Savings", goal.current_amount],
            ["Progress", goal_progress],

            ["", ""]
    ])

    report_rows.extend([

        ["TOP CATEGORIES", ""],

        ["Category", "Amount"]
    ])

    for category, amount in category_totals.head(5).items():

        report_rows.append([
            category,
            round(amount, 2)
    ])

    report_rows.append(["", ""])

    
    
    report_rows.extend([

        ["TOP SUBCATEGORIES", ""],

        ["Category -> Subcategory", "Amount"]
    ])

    for (
        category,
        subcategory
    ), amount in subcategory_totals.head(5).items():

        report_rows.append([
            f"{category} -> {subcategory}",
            round(amount, 2)
        ])

    report_rows.append(["", ""])

    report_rows.extend([

        ["RECOMMENDATIONS", ""]
    ])

    if forecast <= budget:

        report_rows.append([

            "Budget Opportunity",

            f"Projected to finish Rs. {round(budget-forecast,2)} under budget"

    ])

    else:

        report_rows.append([
    
            "Budget Risk",

            f"Projected to exceed budget by Rs. {round(forecast-budget,2)}"

    ])

    if budget > 0:

        remaining_days = max(
            30 - today.day,
            1
        )

        daily_limit = round(
            (budget - total_spent)
            / remaining_days,
            2
        )

        report_rows.append([

            "Daily Spending Limit",

            f"Rs. {daily_limit}/day"

        ])
        report_rows.append(["", ""])

    output = io.StringIO()

    writer = csv.writer(output)

    for row in report_rows:

        writer.writerow(row)

    writer.writerow([])

    writer.writerow(df.columns)

    for row in df.values:

        writer.writerow(row)

    output.seek(0)

    return Response(

        output.getvalue(),

        mimetype="text/csv",

        headers={

            "Content-Disposition":
                "attachment; "
                "filename=finance_pro_report.csv"

        }
    )

@app.route("/download_pdf")
@login_required
def download_pdf():

    expenses = Expense.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Expense.date.desc()
    ).all()

    if not expenses:
        return jsonify({
            "error": "No expenses found."
        }), 404

    pdf = FinancePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ==========================
    # PAGE 1
    # ==========================

    pdf.add_page()

    pdf.set_fill_color(30, 64, 175)

    pdf.set_font(
        "Arial",
        "B",
        20
    )

    pdf.set_text_color(
        255,
        255,
        255
    )

    pdf.cell(
        0,
        15,
        "FINANCE PRO",
        ln=True,
        align="C",
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        12
    )

    pdf.cell(
        0,
        8,
        "Professional Financial Report",
        ln=True,
        align="C"
    )

    pdf.set_text_color(
        0,
        0,
        0
    )

    pdf.ln(5)

    today = datetime.utcnow().date()

    pdf.cell(
        0,
        8,
        f"Generated: {today.strftime('%d %B %Y')}",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"User: {current_user.username}",
        ln=True
    )

    pdf.ln(5)
    
    df = pd.DataFrame([
        {
            "date": e.date,
            "category": e.category,
            "subcategory": e.subcategory,
            "amount": e.amount
        }
        for e in expenses
    ])

    total_spent = df["amount"].sum()

    budget = current_user.budget or 0

    remaining_budget = budget - total_spent

    forecast = round(
        (total_spent / max(today.day, 1)) * 30,
        2
    )

    pdf.set_font("Arial", "B", 14)
    
    pdf.set_fill_color(
        230,
        230,
        230
    )

    pdf.cell(
        0,
        10,
        "FINANCIAL SUMMARY",
        ln=True,
        fill=True
    )

    health = 100

    if budget > 0:

        spending_ratio = total_spent / budget

        if spending_ratio > 1:
            health -= 40

        elif spending_ratio > 0.8:
            health -= 20

        if forecast > budget:
            health -= 20

    else:

        health = 80

    health = max(0, min(100, health))

    status = (
        "Excellent" if health >= 90 else
        "Good" if health >= 75 else
        "Average" if health >= 60 else
        "Needs Improvement"
    )

    budget_status = (
        "WITHIN BUDGET"
        if forecast <= budget
        else "OVER BUDGET"
    )

    pdf.ln(3)

    pdf.set_font("Arial", "", 12)

    pdf.cell(
        0,
        8,
        f"Health Score: {health}/100",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"Rating: {status}",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"Budget Status: {budget_status}",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    pdf.cell(
        0,
        8,
        f"Total Spending: Rs. {total_spent:.2f}",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"Budget: Rs. {budget:.2f}",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"Remaining Budget: Rs. {remaining_budget:.2f}",
        ln=True
    )

    pdf.cell(
        0,
        8,
        f"Forecast: Rs. {forecast:.2f}",
        ln=True
    )

    

    # ==========================
    # SAVINGS GOAL
    # ==========================

    goal = SavingsGoal.query.filter_by(
    user_id=current_user.id
    ).first()

    if goal and goal.target_amount > 0:

        progress = round(
            (
                goal.current_amount /
                goal.target_amount
            ) * 100,
            1
        )

        pdf.set_font("Arial", "B", 14)

        pdf.line(
            10,
            pdf.get_y(),
            200,
            pdf.get_y()
        )

        pdf.ln(3)

        pdf.cell(
            0,
            10,
            "SAVINGS GOAL",
            ln=True
        )

        pdf.set_font("Arial", "", 12)

        pdf.cell(
            0,
            8,
            f"Goal: {goal.goal_name}",
            ln=True
        )

        pdf.cell(
            0,
            8,
            f"Target Amount: Rs. {goal.target_amount}",
            ln=True
        )

        pdf.cell(
            0,
            8,
            f"Current Savings: Rs. {goal.current_amount}",
            ln=True
        )

        pdf.cell(
            0,
            8,
            f"Progress: {progress}%",
            ln=True
        )

        monthly_saving = max(
            current_user.budget * 0.20,
            1
        )

        months_remaining = max(
            0,
            (goal.target_amount - goal.current_amount)
            / monthly_saving
        )

        pdf.cell(
            0,
            8,
            f"Estimated Completion: {months_remaining:.1f} month(s)",
            ln=True
        )

        pdf.ln(5)

    if pdf.get_y() > 220:
        pdf.add_page()

    pdf.set_font("Arial", "B", 16)

    pdf.line(
        10,
        pdf.get_y(),
        200,
        pdf.get_y()
    )

    pdf.ln(3)

    pdf.cell(
        0,
        10,
        "SPENDING ANALYTICS",
        ln=True
    )

    pdf.ln(3)

    # ==========================
    # TOP CATEGORIES
    # ==========================

    category_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    # ==========================
    # PIE CHART IMAGE
    # ==========================

    pie_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.figure(figsize=(5, 5))

    category_totals.head(5).plot(
        kind="pie",
        autopct="%1.1f%%"
    )

    plt.ylabel("")
    plt.title("Top Spending Categories")

    plt.tight_layout()

    plt.savefig(
        pie_file.name,
        bbox_inches="tight"
    )

    plt.close()


    # ==========================
    # TREND CHART IMAGE
    # ==========================

    daily_spending = (
        df.groupby("date")["amount"]
        .sum()
    )

    trend_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    plt.figure(figsize=(6, 3))

    plt.plot(
        daily_spending.index.astype(str),
        daily_spending.values,
        marker="o"
    )

    plt.xticks(rotation=45)

    plt.title("Daily Spending Trend")

    plt.tight_layout()

    plt.savefig(
        trend_file.name,
        bbox_inches="tight"
    )

    plt.close()


    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "TOP CATEGORIES", ln=True)

    pdf.set_font("Arial", "B", 11)

    pdf.set_fill_color(
    220,
    220,
    220
    )

    pdf.cell(
        70,
        8,
        "Category",
        1,
        0,
        fill=True
    )

    pdf.cell(
        50,
        8,
        "Amount",
        1,
        0,
        fill=True
    )

    pdf.cell(
        30,
        8,
        "%",
        1,
        1,
        fill=True
    )

    pdf.set_font("Arial", "", 11)

    for category, amount in category_totals.head(5).items():

        percent = (
            amount / total_spent * 100
            if total_spent > 0
            else 0
        )

        pdf.cell(
            70,
            8,
            str(category),
            1
        )

        pdf.cell(
            50,
            8,
            f"Rs. {amount:.0f}",
            1
        )

        pdf.cell(
            30,
            8,
            f"{percent:.1f}",
            1,
            ln=True
        )

    pdf.ln(5)

    # ==========================
    # CATEGORY DISTRIBUTION
    # ==========================

    if pdf.get_y() > 140:
        pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "CATEGORY DISTRIBUTION",
        ln=True
    )

    pdf.image(
        pie_file.name,
        x=35,
        w=90
    )

    pdf.ln(5)

    # ==========================
    # SPENDING TREND
    # ==========================

    pdf.set_font(
        "Arial",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "SPENDING TREND",
        ln=True
    )

    pdf.image(
    trend_file.name,
    x=20,
    w=150
    )

    pdf.set_x(pdf.l_margin)

    pdf.ln(5)

    # ==========================
    # TOP SUBCATEGORIES
    # ==========================

    subcategory_totals = (
        df.groupby(
            ["category", "subcategory"]
        )["amount"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if pdf.get_y() > 220:
       pdf.add_page()

    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        0,
        10,
        "TOP SUBCATEGORIES",
        ln=True
    )

    pdf.set_font("Arial", "", 12)

    pdf.set_x(pdf.l_margin)

    for (
        category,
        subcategory
    ), amount in subcategory_totals.head(5).items():

        percentage = (
            amount /
            total_spent
        ) * 100

        pdf.cell(
            0,
            8,
            f"{category} -> {subcategory}: "
            f"Rs. {amount:.2f} "
            f"({percentage:.1f}%)",
            ln=True
        )

    pdf.ln(5)

    pdf.set_fill_color(
    245,
    245,
    245
    )

    pdf.set_font(
        "Arial",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "KEY INSIGHTS",
        border=1,
        ln=True,
        fill=True
    )

    pdf.set_font(
        "Arial",
        "",
        11
    )

    highest_category = (
        category_totals.index[0]
        if not category_totals.empty
        else "N/A"
    )

    pdf.cell(
        0,
        8,
        f"- Highest spending category: {highest_category}",
        border=1,
        ln=True
    )

    pdf.cell(
        0,
        8,
        (
            f"- Budget utilization: {(total_spent/budget*100):.1f}%"
            if budget > 0
            else "- No budget set"
        ),
        border=1,
        ln=True
    )

    pdf.ln(5)

    pdf.set_x(pdf.l_margin)

    # ==========================
    # RECOMMENDATIONS
    # ==========================

    pdf.set_font("Arial", "B", 14)

    pdf.line(
        10,
        pdf.get_y(),
        200,
        pdf.get_y()
    )

    pdf.ln(3)

    pdf.set_fill_color(
        245,
        245,
        245
    )

    pdf.cell(
        0,
        10,
        "RECOMMENDATIONS",
        border=1,
        ln=True,
        fill=True
    )

    pdf.set_font("Arial", "", 12)

    if budget > 0:

        if forecast > budget:

            pdf.cell(
                0,
                8,
                f"[WARNING] Budget Risk: "
                f"Projected overspend of "
                f"Rs. {forecast-budget:.0f}",
                ln=True
            )

        else:

            pdf.cell(
                0,
                8,
                f"[OK] Budget Surplus: "
                f"Projected Rs. {budget-forecast:.0f} "
                f"under budget.",
                ln=True
            )
    if budget > total_spent:

        remaining_days = max(
            30 - today.day,
            1
        )

        daily_limit = (
            budget - total_spent
        ) / remaining_days

        pdf.cell(
            0,
            8,
            f"Daily Spending Limit: "
            f"Rs. {daily_limit:.0f}/day",
            ln=True
        )

        if goal and goal.target_amount > 0:

            pdf.cell(
                0,
                8,
                "Saving an extra Rs. 1000/month "
                "could accelerate goal completion.",
                ln=True
            )

            pdf.ln(5)

    # ==========================
    # OUTPUT PDF
    # ==========================

    try:
        pdf_output = pdf.output(dest="S")
    except Exception as e:
        return jsonify({
            "error": f"PDF generation failed: {str(e)}"
        }), 500

    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode("latin-1")

    
    try:
        os.remove(pie_file.name)
    except:
        pass

    try:
        os.remove(trend_file.name)
    except:
        pass


    response = make_response(pdf_output)

    response.headers[
        "Content-Disposition"
    ] = "attachment; filename=finance_pro_report.pdf"

    response.headers[
        "Content-Type"
    ] = "application/pdf"

    return response

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