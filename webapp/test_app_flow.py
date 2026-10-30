"""End-to-end test of the web app with Flask's test client (uses a temporary database).
Run from webapp/:  python test_app_flow.py"""
import os, re, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ["TRUTHLENS_DB"] = os.path.join(tmp, "test.db")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as appmod  # noqa: E402

app = appmod.app
ok = 0
def check(cond, msg):
    global ok
    if not cond:
        raise AssertionError("FAILED: " + msg)
    ok += 1; print("  PASS", msg)

def csrf(client, url):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)

def post(client, url, data, page=None, **kw):
    data = dict(data, csrf_token=csrf(client, page or url))
    return client.post(url, data=data, **kw)

u = app.test_client()
a = app.test_client()
anon = app.test_client()

print("1. Public pages")
for p in ["/", "/about", "/login", "/register", "/admin/login"]:
    check(anon.get(p).status_code == 200, f"GET {p} is public")

print("2. Unauthorized access is blocked")
for p in ["/dashboard", "/predict", "/history", "/profile", "/admin", "/admin/users", "/admin/records", "/admin/activity"]:
    r = anon.get(p)
    check(r.status_code == 302 and "login" in r.headers["Location"], f"anonymous GET {p} redirected to login")
check(anon.post("/api/predict", json={"statement": "x" * 20}).status_code == 400, "POST without CSRF token rejected")
check(anon.post("/api/predict", json={"statement": "x" * 20}, headers={"X-CSRF-Token": csrf(anon, "/login")}).status_code == 401,
      "API predict requires login")

print("3. Registration -> pending")
r = post(u, "/register", {"name": "Test User", "email": "test@example.com", "password": "Passw0rd1", "confirm": "Passw0rd1"},
         follow_redirects=True)
check("PENDING" in r.get_data(as_text=True), "registration shows pending message")
check(b"already exists" in post(u, "/register", {"name": "Dup", "email": "test@example.com", "password": "Passw0rd1",
                                                 "confirm": "Passw0rd1"}).data, "duplicate email rejected")
check(b"valid email" in post(u, "/register", {"name": "Bad", "email": "bad", "password": "Passw0rd1", "confirm": "Passw0rd1"}).data,
      "invalid email rejected")
check(b"at least 8" in post(u, "/register", {"name": "Weak", "email": "w@example.com", "password": "short", "confirm": "short"}).data,
      "weak password rejected")
post(u, "/register", {"name": "Rejected User", "email": "rej@example.com", "password": "Passw0rd1", "confirm": "Passw0rd1"})
db = appmod.sqlite3.connect(os.environ["TRUTHLENS_DB"])
check(db.execute("SELECT role, status FROM users WHERE email='test@example.com'").fetchone() == ("user", "pending"),
      "new account stored as role=user, status=pending")
check(db.execute("SELECT password_hash FROM users WHERE email='test@example.com'").fetchone()[0] != "Passw0rd1",
      "password stored hashed")
r = post(u, "/login", {"email": "test@example.com", "password": "Passw0rd1"})
check(b"pending approval" in r.data and u.get("/dashboard").status_code == 302, "pending user cannot log in")

print("4. Admin login and approval")
check(b"Invalid admin" in post(a, "/admin/login", {"username": "admin", "password": "wrong"}).data, "wrong admin password rejected")
check(b"Invalid admin" in post(a, "/admin/login", {"username": "test@example.com", "password": "Passw0rd1"}).data,
      "normal user cannot use admin login")
r = post(a, "/admin/login", {"username": "admin", "password": "Admin@123"})
check(r.status_code == 302 and r.headers["Location"].endswith("/admin"), "default admin (admin / Admin@123) logs in")
check(b"test@example.com" in a.get("/admin").data, "pending user listed on admin dashboard")
uid = db.execute("SELECT id FROM users WHERE email='test@example.com'").fetchone()[0]
rid = db.execute("SELECT id FROM users WHERE email='rej@example.com'").fetchone()[0]
post(a, f"/admin/users/{uid}/approve", {}, page="/admin/users")
post(a, f"/admin/users/{rid}/reject", {}, page="/admin/users")
check(db.execute("SELECT status FROM users WHERE id=?", (uid,)).fetchone()[0] == "approved", "admin approved user")
check(b"rejected" in post(app.test_client(), "/login", {"email": "rej@example.com", "password": "Passw0rd1"}).data,
      "rejected user cannot log in")

print("5. User login -> dashboard -> prediction -> history")
r = post(u, "/login", {"email": "test@example.com", "password": "Passw0rd1"})
check(r.status_code == 302 and r.headers["Location"].endswith("/dashboard"), "approved user logs in")
check(u.get("/dashboard").status_code == 200, "dashboard accessible")
check(u.get("/admin").status_code == 403, "user blocked from admin pages")
stmt = "Says the federal government will confiscate all privately owned guns starting next month."
r = post(u, "/predict", {"statement": stmt}, page="/predict")
html = r.get_data(as_text=True)
check(r.status_code == 200 and ("FAKE NEWS" in html or "REAL NEWS" in html), "prediction rendered with verdict")
check(b"too short" in post(u, "/predict", {"statement": "short"}, page="/predict").data or
      b"between 10 and 1000" in post(u, "/predict", {"statement": "short"}, page="/predict").data, "too-short input rejected")
r = u.post("/api/predict", json={"statement": "The unemployment rate dropped to its lowest level in a decade last year."},
           headers={"X-CSRF-Token": csrf(u, "/predict")})
j = r.get_json()
check(r.status_code == 200 and j["prediction"] in ("Fake", "Real") and 0 <= j["p_fake"] <= 1, f"JSON API works: {j['prediction']} p_fake={j['p_fake']}")
row = db.execute("SELECT statement, prediction, p_fake FROM predictions WHERE user_id=? ORDER BY id", (uid,)).fetchall()
check(len(row) == 2 and row[0][0] == stmt, "predictions saved to history with timestamps")
check(stmt[:60].encode() in u.get("/history").data, "user sees own history")
pid = db.execute("SELECT id FROM predictions WHERE user_id=? LIMIT 1", (uid,)).fetchone()[0]
check(u.get(f"/history/{pid}").status_code == 200, "history detail page")

print("6. History isolation between users")
post(app.test_client(), "/register", {"name": "Other", "email": "other@example.com", "password": "Passw0rd1", "confirm": "Passw0rd1"})
oid = db.execute("SELECT id FROM users WHERE email='other@example.com'").fetchone()[0]
post(a, f"/admin/users/{oid}/approve", {}, page="/admin/users")
o = app.test_client(); post(o, "/login", {"email": "other@example.com", "password": "Passw0rd1"})
check(o.get(f"/history/{pid}").status_code == 404, "user cannot open another user's record")
check(stmt[:60].encode() not in o.get("/history").data, "other user's history not listed")

print("7. Admin views all records and activity")
check(stmt[:60].encode() in a.get("/admin/records").data, "admin sees all records")
act = a.get("/admin/activity").get_data(as_text=True)
check("prediction" in act and "admin_approve" in act and "register" in act, "activity log records actions")

print("8. Approval revoked mid-session + logout")
post(a, f"/admin/users/{uid}/reject", {}, page="/admin/users")
check(u.get("/dashboard").status_code == 302, "session of a user rejected later is blocked")
post(o, "/logout", {}, page="/dashboard")
check(o.get("/dashboard").status_code == 302, "logout ends session")
check(db.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0] == 1, "registration never created an admin")

print(f"\nALL {ok} CHECKS PASSED")
