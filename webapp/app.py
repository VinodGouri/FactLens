"""FactLens - RAG-Enhanced Fake News Detection web application (Flask + SQLite).

Uses the saved best model from models/best_model (trained in code/01_...ipynb) for predictions.
Run:  python app.py      (default admin: admin / Admin@123)
"""
import os, re, json, secrets, sqlite3, functools
from datetime import datetime, timedelta, timezone

import pandas as pd
from flask import (Flask, g, render_template, request, redirect, url_for, session, flash,
                   jsonify, abort)
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
DB_PATH = os.environ.get("FACTLENS_DB", os.path.join(INSTANCE_DIR, "factlens.db"))
BEST_MODEL_DIR = os.environ.get("FACTLENS_MODEL_DIR", os.path.join(PROJECT_DIR, "models", "best_model"))
EVAL_DIR = os.path.join(PROJECT_DIR, "models", "model_evaluation_results")


def _secret_key():
    if os.environ.get("FACTLENS_SECRET_KEY"):
        return os.environ["FACTLENS_SECRET_KEY"]
    path = os.path.join(INSTANCE_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(path, 0o600)
    return open(path).read().strip()


app = Flask(__name__)
app.config.update(
    SECRET_KEY=_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FACTLENS_SECURE_COOKIES", "0") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    MAX_CONTENT_LENGTH=256 * 1024,
)

# ----------------------------------------------------------------------------- database
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    username TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')) DEFAULT 'user',
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED')) DEFAULT 'PENDING',
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    last_login TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    prediction TEXT NOT NULL,
    p_fake REAL NOT NULL,
    confidence REAL NOT NULL,
    model_name TEXT NOT NULL,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    details TEXT,
    ip TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_act_user ON activity_log(user_id, created_at);
"""


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and the default admin account (admin / Admin@123) if missing."""
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    if not db.execute("SELECT 1 FROM users WHERE role='admin' AND username='admin'").fetchone():
        db.execute("INSERT INTO users (name, username, password_hash, role, status, created_at, reviewed_at) "
                   "VALUES (?, ?, ?, 'admin', 'APPROVED', ?, ?)",
                   ("Administrator", "admin", generate_password_hash("Admin@123"), now(), now()))
    db.commit()
    db.close()


def log_activity(action, details="", user_id=None):
    uid = user_id if user_id is not None else session.get("user_id")
    get_db().execute("INSERT INTO activity_log (user_id, action, details, ip, created_at) VALUES (?, ?, ?, ?, ?)",
                     (uid, action, details[:500], request.remote_addr, now()))
    get_db().commit()


# ----------------------------------------------------------------------------- model
from predictor import FakeNewsPredictor  # noqa: E402

predictor = FakeNewsPredictor(BEST_MODEL_DIR)


def model_summary():
    """Best-model info + evaluation table written by the training notebook (real results)."""
    info_path = os.path.join(BEST_MODEL_DIR, "best_model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else None
    table_path = os.path.join(EVAL_DIR, "test_metrics.csv")
    table = pd.read_csv(table_path).round(4).to_dict("records") if os.path.exists(table_path) else []
    return info, table


# ----------------------------------------------------------------------------- security helpers
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def csrf_protect():
    if request.method == "POST":
        sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            if request.path.startswith("/api/"):
                return jsonify(error="Invalid or missing CSRF token"), 400
            abort(400, "Invalid or missing CSRF token")


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    if "user_id" in session:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def current_user():
    """Reload the user from the DB on every request so role/status changes apply immediately."""
    if "user" not in g:
        uid = session.get("user_id")
        g.user = get_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return g.user


@app.context_processor
def inject_user():
    return {"me": current_user()}


def login_required(role):
    def deco(view):
        @functools.wraps(view)
        def wrapped(*a, **kw):
            u = current_user()
            is_api = request.path.startswith("/api/")
            if u is None:
                session.pop("user_id", None)  # drop stale login, keep CSRF token
                if is_api:
                    return jsonify(error="Authentication required"), 401
                flash("Please log in to continue.", "warning")
                return redirect(url_for("admin_login" if role == "admin" else "login"))
            if u["role"] != role:
                if is_api:
                    return jsonify(error="Forbidden"), 403
                abort(403)
            if u["status"] != "APPROVED":
                session.clear()
                if is_api:
                    return jsonify(error="Account not approved"), 403
                flash("Your account is not approved.", "danger")
                return redirect(url_for("login"))
            return view(*a, **kw)
        return wrapped
    return deco


def start_session(user):
    session.clear()  # prevent session fixation
    session.permanent = True
    session["user_id"] = user["id"]
    csrf_token()
    get_db().execute("UPDATE users SET last_login=? WHERE id=?", (now(), user["id"]))
    get_db().commit()


# ----------------------------------------------------------------------------- public pages
@app.route("/")
def home():
    info, _ = model_summary()
    return render_template("home.html", info=info)


@app.route("/about")
def about():
    info, table = model_summary()
    return render_template("about.html", info=info, table=table)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        errors = []
        if not (2 <= len(name) <= 80) or not re.match(r"^[A-Za-z][A-Za-z .'-]*$", name):
            errors.append("Name must be 2-80 letters (spaces, . ' - allowed).")
        if not EMAIL_RE.match(email) or len(email) > 120:
            errors.append("Enter a valid email address.")
        if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            errors.append("Password must be at least 8 characters and contain letters and digits.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if not errors and get_db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            errors.append("An account with this email already exists.")
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("register.html", name=name, email=email), 400
        # role and status are fixed server-side: public registration can never create an admin
        cur = get_db().execute("INSERT INTO users (name, email, password_hash, role, status, created_at) "
                               "VALUES (?, ?, ?, 'user', 'PENDING', ?)",
                               (name, email, generate_password_hash(password), now()))
        get_db().commit()
        log_activity("REGISTER", f"New registration: {email}", user_id=cur.lastrowid)
        return render_template("pending.html", name=name)
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        u = get_db().execute("SELECT * FROM users WHERE email=? AND role='user'", (email,)).fetchone()
        if u is None or not check_password_hash(u["password_hash"], password):
            if u is not None:
                log_activity("LOGIN_FAILED", "Wrong password", user_id=u["id"])
            flash("Invalid email or password.", "danger")
            return render_template("login.html", email=email), 401
        if u["status"] == "PENDING":
            log_activity("LOGIN_BLOCKED", "Account pending approval", user_id=u["id"])
            return render_template("pending.html", name=u["name"]), 403
        if u["status"] == "REJECTED":
            log_activity("LOGIN_BLOCKED", "Account rejected", user_id=u["id"])
            flash("Your registration was rejected by the administrator. You cannot log in.", "danger")
            return render_template("login.html", email=email), 403
        start_session(u)
        log_activity("LOGIN", "User logged in")
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        u = get_db().execute("SELECT * FROM users WHERE username=? AND role='admin'", (username,)).fetchone()
        if u is None or not check_password_hash(u["password_hash"], password):
            flash("Invalid admin credentials.", "danger")
            return render_template("admin_login.html", username=username), 401
        start_session(u)
        log_activity("ADMIN_LOGIN", "Admin logged in")
        return redirect(url_for("admin_dashboard"))
    return render_template("admin_login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if current_user() is not None:
        log_activity("LOGOUT", "Logged out")
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ----------------------------------------------------------------------------- user area
def user_stats(uid):
    db = get_db()
    rows = db.execute("SELECT prediction, COUNT(*) c FROM predictions WHERE user_id=? GROUP BY prediction", (uid,)).fetchall()
    counts = {r["prediction"]: r["c"] for r in rows}
    daily = db.execute("SELECT substr(created_at,1,10) d, SUM(prediction='Fake') fake, SUM(prediction='Real') real "
                       "FROM predictions WHERE user_id=? GROUP BY d ORDER BY d DESC LIMIT 14", (uid,)).fetchall()
    avg = db.execute("SELECT AVG(confidence) FROM predictions WHERE user_id=?", (uid,)).fetchone()[0]
    return {"total": sum(counts.values()), "fake": counts.get("Fake", 0), "real": counts.get("Real", 0),
            "avg_confidence": round(avg, 4) if avg is not None else None,
            "daily": [dict(r) for r in reversed(daily)]}


@app.route("/dashboard")
@login_required("user")
def dashboard():
    u = current_user()
    recent = get_db().execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 5", (u["id"],)).fetchall()
    info, table = model_summary()
    return render_template("dashboard.html", stats=user_stats(u["id"]), recent=recent, info=info, table=table)


def validate_statement(text):
    text = (text or "").strip()
    if len(text) < 10:
        return None, "Please enter a news statement of at least 10 characters."
    if len(text) > 1000:
        return None, "Statement is too long (max 1000 characters)."
    if len(re.findall(r"[A-Za-z]", text)) < 5:
        return None, "Statement must contain English words."
    return text, None


def run_and_store(text):
    res = predictor.predict([text])[0]
    model_name = predictor.model_info["best_model"]
    db = get_db()
    cur = db.execute("INSERT INTO predictions (user_id, statement, prediction, p_fake, confidence, model_name, "
                     "evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (session["user_id"], text, res["prediction"], res["p_fake"], res["confidence"], model_name,
                      json.dumps(res["evidence"]), now()))
    db.commit()
    log_activity("PREDICT", f"#{cur.lastrowid}: {res['prediction']} ({res['confidence']:.2%})")
    res["id"], res["model"] = cur.lastrowid, model_name
    return res


@app.route("/predict", methods=["GET", "POST"])
@login_required("user")
def predict_page():
    result, text = None, ""
    if request.method == "POST":
        text, err = validate_statement(request.form.get("statement"))
        if err:
            flash(err, "danger")
            text = request.form.get("statement", "")[:1000]
        else:
            result = run_and_store(text)
    return render_template("predict.html", result=result, text=text)


@app.route("/api/predict", methods=["POST"])
@login_required("user")
def api_predict():
    data = request.get_json(silent=True) or {}
    text, err = validate_statement(data.get("statement"))
    if err:
        return jsonify(error=err), 400
    try:
        return jsonify(run_and_store(text))
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify(error=f"Prediction failed: {e}"), 500


@app.route("/api/my/stats")
@login_required("user")
def api_my_stats():
    return jsonify(user_stats(current_user()["id"]))


@app.route("/history")
@login_required("user")
def history():
    u = current_user()
    rows = get_db().execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (u["id"],)).fetchall()
    acts = get_db().execute("SELECT * FROM activity_log WHERE user_id=? ORDER BY id DESC LIMIT 50", (u["id"],)).fetchall()
    return render_template("history.html", rows=rows, acts=acts)


@app.route("/history/<int:pid>")
@login_required("user")
def history_detail(pid):
    row = get_db().execute("SELECT * FROM predictions WHERE id=? AND user_id=?", (pid, session["user_id"])).fetchone()
    if row is None:
        abort(404)  # users can only see their own records
    return render_template("history_detail.html", r=row, evidence=json.loads(row["evidence_json"] or "[]"))


@app.route("/profile", methods=["GET", "POST"])
@login_required("user")
def profile():
    u = current_user()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "name":
            name = request.form.get("name", "").strip()
            if not (2 <= len(name) <= 80) or not re.match(r"^[A-Za-z][A-Za-z .'-]*$", name):
                flash("Name must be 2-80 letters.", "danger")
            else:
                get_db().execute("UPDATE users SET name=? WHERE id=?", (name, u["id"])); get_db().commit()
                log_activity("PROFILE_UPDATE", "Changed name"); flash("Profile updated.", "success")
        elif action == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not check_password_hash(u["password_hash"], cur):
                flash("Current password is incorrect.", "danger")
            elif len(new) < 8 or not re.search(r"[A-Za-z]", new) or not re.search(r"\d", new) or new != conf:
                flash("New password must be 8+ characters with letters and digits, and match confirmation.", "danger")
            else:
                get_db().execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new), u["id"]))
                get_db().commit(); log_activity("PASSWORD_CHANGE", "Changed password"); flash("Password changed.", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html", stats=user_stats(u["id"]))


# ----------------------------------------------------------------------------- admin area
@app.route("/admin")
@login_required("admin")
def admin_dashboard():
    db = get_db()
    counts = {r["status"]: r["c"] for r in db.execute("SELECT status, COUNT(*) c FROM users WHERE role='user' GROUP BY status")}
    pred_counts = {r["prediction"]: r["c"] for r in db.execute("SELECT prediction, COUNT(*) c FROM predictions GROUP BY prediction")}
    daily = [dict(r) for r in db.execute(
        "SELECT substr(created_at,1,10) d, COUNT(*) n FROM predictions GROUP BY d ORDER BY d DESC LIMIT 14")][::-1]
    pending = db.execute("SELECT * FROM users WHERE role='user' AND status='PENDING' ORDER BY id").fetchall()
    acts = db.execute("SELECT a.*, u.name, u.email, u.username FROM activity_log a LEFT JOIN users u ON u.id=a.user_id "
                      "ORDER BY a.id DESC LIMIT 10").fetchall()
    info, table = model_summary()
    return render_template("admin_dashboard.html", counts=counts, pred_counts=pred_counts, daily=daily,
                           pending=pending, acts=acts, info=info, table=table)


@app.route("/admin/users")
@login_required("admin")
def admin_users():
    status = request.args.get("status", "")
    q = ("SELECT u.*, (SELECT COUNT(*) FROM predictions p WHERE p.user_id=u.id) n_pred FROM users u WHERE role='user'"
         + (" AND status=?" if status in ("PENDING", "APPROVED", "REJECTED") else "") + " ORDER BY id DESC")
    rows = get_db().execute(q, (status,) if status in ("PENDING", "APPROVED", "REJECTED") else ()).fetchall()
    return render_template("admin_users.html", rows=rows, status=status)


@app.route("/admin/users/<int:uid>/<action>", methods=["POST"])
@login_required("admin")
def admin_user_action(uid, action):
    db = get_db()
    u = db.execute("SELECT * FROM users WHERE id=? AND role='user'", (uid,)).fetchone()
    if u is None:
        abort(404)  # admin accounts cannot be modified through this endpoint
    if action in ("approve", "reject"):
        new = "APPROVED" if action == "approve" else "REJECTED"
        db.execute("UPDATE users SET status=?, reviewed_at=? WHERE id=?", (new, now(), uid))
        log_activity(f"ADMIN_{action.upper()}", f"{u['email']} -> {new}")
        flash(f"{u['name']} is now {new}.", "success")
    elif action == "delete":
        db.execute("DELETE FROM users WHERE id=?", (uid,))
        log_activity("ADMIN_DELETE_USER", f"Deleted {u['email']}")
        flash(f"Deleted {u['name']}.", "success")
    else:
        abort(400)
    db.commit()
    return redirect(request.referrer if request.referrer and request.host in request.referrer else url_for("admin_users"))


@app.route("/admin/records")
@login_required("admin")
def admin_records():
    rows = get_db().execute("SELECT p.*, u.name, u.email FROM predictions p JOIN users u ON u.id=p.user_id "
                            "ORDER BY p.id DESC LIMIT 500").fetchall()
    return render_template("admin_records.html", rows=rows)


@app.route("/admin/activity")
@login_required("admin")
def admin_activity():
    rows = get_db().execute("SELECT a.*, u.name, u.email, u.username FROM activity_log a LEFT JOIN users u ON u.id=a.user_id "
                            "ORDER BY a.id DESC LIMIT 500").fetchall()
    return render_template("admin_activity.html", rows=rows)


@app.errorhandler(403)
def forbidden(_e):
    return render_template("error.html", code=403, msg="You do not have permission to access this page."), 403


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404, msg="Page not found."), 404


@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", code=400, msg=getattr(e, "description", "Bad request")), 400


init_db()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
