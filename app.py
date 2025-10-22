# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import openai
import os
import io
from fpdf import FPDF
from sklearn.linear_model import LinearRegression
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = 'mysecretkey123'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///financial_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

openai.api_key = os.getenv("OPENAI_API_KEY")


# --- Models ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    expenses = db.relationship('Expense', backref='user', lazy=True)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(20))
    category = db.Column(db.String(100))
    amount = db.Column(db.Float)

with app.app_context():
    db.create_all()


# --- Auth ---

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            return "Username already exists."

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        return "Invalid credentials."

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# --- Dashboard ---

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


# --- Expense Handling ---

@app.route("/add_expense", methods=["POST"])
@login_required
def add_expense():
    category = request.form["category"]
    amount = float(request.form["amount"])
    date = datetime.now().strftime("%Y-%m-%d")

    expense = Expense(user_id=current_user.id, date=date, category=category, amount=amount)
    db.session.add(expense)
    db.session.commit()

    return jsonify({"message": "Expense added successfully."})

@app.route("/get_data")
@login_required
def get_data():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()

    if not expenses:
        return jsonify({"categories": [], "totals": [], "dates": [], "daily_totals": []})

    df = pd.DataFrame([(e.date, e.category, e.amount) for e in expenses], columns=["date", "category", "amount"])

    category_totals = df.groupby("category")["amount"].sum().to_dict()
    daily_totals = df.groupby("date")["amount"].sum().to_dict()

    return jsonify({
        "categories": list(category_totals.keys()),
        "totals": list(category_totals.values()),
        "dates": list(daily_totals.keys()),
        "daily_totals": list(daily_totals.values())
    })


# --- CSV & PDF Download ---

@app.route("/download_csv")
@login_required
def download_csv():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    df = pd.DataFrame([(e.date, e.category, e.amount) for e in expenses], columns=["Date", "Category", "Amount"])
    csv_data = df.to_csv(index=False)

    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="expenses.csv"
    )

@app.route("/download_pdf")
@login_required
def download_pdf():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    df = pd.DataFrame([(e.date, e.category, e.amount) for e in expenses], columns=["Date", "Category", "Amount"])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Expense Report", ln=True, align='C')
    pdf.ln(10)

    for _, row in df.iterrows():
        pdf.cell(200, 8, txt=f"{row['Date']} - {row['Category']}: ${row['Amount']}", ln=True)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name="expenses.pdf"
    )


# --- AI Assistant ---

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    question = request.form["question"]

    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    if not expenses:
        return jsonify({"answer": "No expenses found to analyze."})

    df = pd.DataFrame([(e.date, e.category, e.amount) for e in expenses], columns=["date", "category", "amount"])
    summary = df.groupby("category")["amount"].sum().to_dict()
    total_spent = df["amount"].sum()

    prompt = f"""
    You're an AI financial assistant. Here's a summary:
    Total Spent: ${total_spent:.2f}
    Categories: {summary}
    Question: {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message["content"]
    return jsonify({"answer": answer})


# --- Prediction (Next Week) ---

@app.route("/predict_future")
@login_required
def predict_future():
    expenses = Expense.query.filter_by(user_id=current_user.id).all()
    if not expenses:
        return jsonify({"prediction": "No data available for prediction."})

    df = pd.DataFrame([(e.date, e.amount) for e in expenses], columns=["date", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["day_index"] = (df["date"] - df["date"].min()).dt.days

    model = LinearRegression()
    model.fit(df[["day_index"]], df["amount"])

    next_week = np.array([[df["day_index"].max() + 7]])
    predicted = model.predict(next_week)[0]

    return jsonify({"prediction": f"Estimated spending next week: ${predicted:.2f}"})


# --- Run App ---

if __name__ == "__main__":
    app.run(debug=True)
