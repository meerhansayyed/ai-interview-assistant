import streamlit as st
from dotenv import load_dotenv
import os
from db import init_db, create_user, authenticate_user

# ==========================
# INIT
# ==========================
load_dotenv()
init_db()

# Auto-create default admin account if it doesn't exist
if not authenticate_user("admin", "admin123"):
    create_user("admin", "admin@company.com", "admin123", role="admin")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="centered"
)

# ==========================
# SESSION STATE DEFAULTS
# ==========================
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", "")
st.session_state.setdefault("show_admin_signup", False)

# ==========================
# REDIRECT IF ALREADY LOGGED IN
# ==========================
if st.session_state.logged_in:
    if st.session_state.role == "admin":
        st.switch_page("pages/admin_dashboard.py")
    else:
        st.switch_page("pages/user_interview.py")

# ==========================
# SECRET ADMIN CODE
# Change this to whatever secret you want
# ==========================
ADMIN_SECRET_CODE = "ADMIN@2024"

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.login-hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
}

.login-hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #1a1a2e;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}

.login-hero .sub {
    color: #666;
    font-size: 1.05rem;
    margin-top: 0.25rem;
}

.role-card {
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border: 2px solid transparent;
    margin-bottom: 0.5rem;
}

.role-card-candidate {
    background: #f0f7ff;
    border-color: #c0d8f8;
}

.role-card-admin {
    background: #fff8f0;
    border-color: #f8d8b0;
}

.role-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.role-label-candidate { color: #2563eb; }
.role-label-admin     { color: #d97706; }

.hint-box {
    background: #eef2ff;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #3d52a0;
    font-size: 0.9rem;
    margin-top: 0.75rem;
    text-align: center;
}

.admin-unlock-box {
    background: #fffbeb;
    border: 1px dashed #f59e0b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
    color: #92400e;
    font-size: 0.9rem;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 500;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HERO HEADER
# ==========================
st.markdown("""
<div class="login-hero">
    <h1>🤖 AI Interview<br>Assistant</h1>
    <p class="sub">Create an account or sign in to get started</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# TABS
# ==========================
tab_signup, tab_signin = st.tabs(["📝 Sign Up", "🔑 Sign In"])

# ============================================================
# SIGN UP TAB
# ============================================================
with tab_signup:

    st.markdown("**Choose your role:**")
    role_col1, role_col2 = st.columns(2)

    with role_col1:
        st.markdown("""
        <div class="role-card role-card-candidate">
            <div class="role-label role-label-candidate">👤 Candidate</div>
            <div style="font-size:0.9rem;color:#444;margin-top:4px;">
                Take AI-powered mock interviews and get detailed feedback
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign up as Candidate", use_container_width=True, key="pick_candidate"):
            st.session_state.show_admin_signup = False
            st.rerun()

    with role_col2:
        st.markdown("""
        <div class="role-card role-card-admin">
            <div class="role-label role-label-admin">🛡️ Admin</div>
            <div style="font-size:0.9rem;color:#444;margin-top:4px;">
                Manage interviews, view analytics and candidate reports
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign up as Admin", use_container_width=True, key="pick_admin"):
            st.session_state.show_admin_signup = True
            st.rerun()

    st.markdown("---")

    # ── CANDIDATE SIGNUP FORM ──────────────────────────────
    if not st.session_state.show_admin_signup:
        st.markdown("""
        <div class="role-label role-label-candidate" style="margin-bottom:0.75rem;">
            👤 New Candidate Registration
        </div>
        """, unsafe_allow_html=True)

        with st.form("signup_candidate_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username")
                new_email    = st.text_input("Email")
            with col2:
                new_password = st.text_input("Password",         type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")

            if st.form_submit_button("Create Candidate Account →", use_container_width=True):
                if not new_username.strip() or not new_email.strip() or not new_password:
                    st.error("❌ All fields are required.")
                elif new_password != confirm_pass:
                    st.error("❌ Passwords don't match!")
                elif len(new_password) < 4:
                    st.error("❌ Password must be at least 4 characters.")
                elif create_user(new_username.strip(), new_email.strip(), new_password, role="user"):
                    st.success("✅ Account created! Switch to the Sign In tab.")
                else:
                    st.error("❌ Username or email already exists!")

    # ── ADMIN SIGNUP FORM ──────────────────────────────────
    else:
        st.markdown("""
        <div class="role-label role-label-admin" style="margin-bottom:0.5rem;">
            🛡️ Admin Registration
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="admin-unlock-box">
            🔐 Admin accounts require an <b>authorization code</b>.
            Contact your system administrator to obtain one.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        with st.form("signup_admin_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                adm_username = st.text_input("Username")
                adm_email    = st.text_input("Email")
            with col2:
                adm_password = st.text_input("Password",         type="password")
                adm_confirm  = st.text_input("Confirm Password", type="password")

            adm_code = st.text_input(
                "🔑 Admin Authorization Code",
                type="password",
                placeholder="Enter the secret code from your administrator"
            )

            if st.form_submit_button("Create Admin Account →", use_container_width=True):
                if not adm_username.strip() or not adm_email.strip() or not adm_password:
                    st.error("❌ All fields are required.")
                elif adm_password != adm_confirm:
                    st.error("❌ Passwords don't match!")
                elif len(adm_password) < 4:
                    st.error("❌ Password must be at least 4 characters.")
                elif adm_code != ADMIN_SECRET_CODE:
                    st.error("❌ Invalid authorization code.")
                elif create_user(adm_username.strip(), adm_email.strip(), adm_password, role="admin"):
                    st.success("✅ Admin account created! Switch to the Sign In tab.")
                else:
                    st.error("❌ Username or email already exists!")

# ============================================================
# SIGN IN TAB
# ============================================================
with tab_signin:

    st.markdown("**Quick Demo Access:**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🛡️ Login as Admin", use_container_width=True):
            user = authenticate_user("admin", "admin123")
            if user:
                st.session_state.logged_in = True
                st.session_state.username  = user["username"]
                st.session_state.role      = user["role"]
                st.rerun()
            else:
                st.error("Admin account not found. Restart the app.")

    with col2:
        if st.button("👤 Login as Demo Candidate", use_container_width=True):
            if not authenticate_user("demo_user", "demo123"):
                create_user("demo_user", "demo@example.com", "demo123", role="user")
            st.session_state.logged_in = True
            st.session_state.username  = "demo_user"
            st.session_state.role      = "user"
            st.rerun()

    st.markdown("---")

    with st.form("signin_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")

        if st.form_submit_button("🔑 Sign In", use_container_width=True):
            if not login_username.strip() or not login_password:
                st.error("❌ Please enter username and password.")
            else:
                user = authenticate_user(login_username.strip(), login_password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username  = user["username"]
                    st.session_state.role      = user["role"]
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

    st.markdown("""
    <div class="hint-box">
        💡 Demo credentials — Admin: <b>admin / admin123</b>
    </div>
    """, unsafe_allow_html=True)