import streamlit as st
import sys
import os
import json
from datetime import datetime
import io

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import (
    get_all_interviews, get_interview_by_id,
    delete_interview, get_stats, init_db
)

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Admin Dashboard · AI Interview",
    page_icon="🛡️",
    layout="wide"
)

# ==========================
# AUTH GUARD
# ==========================
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", "")

if not st.session_state.logged_in:
    st.warning("🔒 Please log in first.")
    if st.button("← Go to Login"):
        st.switch_page("app.py")
    st.stop()

if st.session_state.role != "admin" and st.session_state.username != "admin":
    st.error("⛔ Admin access required.")
    if st.button("← Go to Login"):
        st.switch_page("app.py")
    st.stop()

init_db()

# ==========================
# HELPERS
# ==========================
HIRE_COLOR = {
    "Strong Hire": "#27ae60",
    "Hire":        "#2ecc71",
    "Maybe":       "#f39c12",
    "No Hire":     "#e74c3c",
}
HIRE_EMOJI = {
    "Strong Hire": "🟢",
    "Hire":        "🟡",
    "Maybe":       "🟠",
    "No Hire":     "🔴",
}
SENTIMENT_EMOJI = {"Positive":"😊","Neutral":"😐","Negative":"😟"}
TONE_EMOJI = {
    "Assertive":"💪","Calm":"😌","Hesitant":"🤔",
    "Nervous":"😰","Enthusiastic":"🔥","Flat":"😑"
}


# ==========================
# PDF EXPORT FOR ADMIN
# ==========================
def generate_admin_pdf(data: dict) -> io.BytesIO:
    """Generate a detailed PDF for a single interview record."""
    sc  = data["scorecard"]
    sl  = data["sentiment_log"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                              textColor=colors.HexColor("#1a1a2e"), spaceAfter=4)
    sec_s   = ParagraphStyle("S", parent=styles["Heading2"], fontSize=12,
                              textColor=colors.HexColor("#16213e"), spaceBefore=12, spaceAfter=4)
    body_s  = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                              leading=14, textColor=colors.HexColor("#2c2c2c"))
    label_s = ParagraphStyle("L", parent=styles["Normal"], fontSize=9,
                              textColor=colors.HexColor("#666666"), leading=12)
    small_s = ParagraphStyle("SM", parent=styles["Normal"], fontSize=8,
                              leading=11, textColor=colors.HexColor("#444444"))
    elems   = []

    hire    = sc.get("hire_recommendation","N/A")
    overall = sc.get("overall_score", 0)

    elems.append(Paragraph("AI Interview Report — Admin Export", title_s))
    elems.append(Paragraph(
        f"Candidate: <b>{data['username']}</b> &nbsp;|&nbsp; "
        f"Domain: <b>{data['domain']}</b> &nbsp;|&nbsp; "
        f"Date: <b>{data['created_at']}</b>", label_s))
    elems.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1a1a2e"), spaceAfter=8))

    # Score summary table
    oc = colors.HexColor("#27ae60" if overall >= 75 else "#f39c12" if overall >= 50 else "#e74c3c")
    summary_data = [[
        Paragraph("<b>Overall Score</b>", body_s),
        Paragraph(f"<b>{overall} / 100</b>", ParagraphStyle(
            "OS", parent=styles["Normal"], fontSize=18, textColor=oc, alignment=1)),
        Paragraph("<b>Recommendation</b>", body_s),
        Paragraph(f"<b>{hire}</b>", ParagraphStyle(
            "HR", parent=styles["Normal"], fontSize=12, textColor=oc, alignment=1)),
    ]]
    st_table = Table(summary_data, colWidths=[1.5*inch]*4)
    st_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f0f4ff")),
        ("BOX",       (0,0),(-1,-1), 1, colors.HexColor("#c0c8e8")),
        ("INNERGRID", (0,0),(-1,-1), 0.5, colors.HexColor("#c0c8e8")),
        ("ALIGN",     (0,0),(-1,-1),"CENTER"),
        ("VALIGN",    (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    elems += [st_table, Spacer(1,6)]
    elems.append(Paragraph(sc.get("overall_summary",""), body_s))
    comm = sc.get("communication_insight","")
    if comm:
        elems += [Spacer(1,4), Paragraph(f"<i>{comm}</i>", small_s)]
    elems.append(Spacer(1,10))

    # Question scores
    qs_list = sc.get("question_scores",[])
    if qs_list:
        elems.append(Paragraph("Question Scores", sec_s))
        header = [Paragraph(f"<b>{h}</b>", body_s)
                  for h in ["#","Question","Clarity","Depth","Accuracy","Avg"]]
        rows   = [header]
        for i, qs in enumerate(qs_list, 1):
            cl,dp,ac = qs.get("clarity",0),qs.get("depth",0),qs.get("technical_accuracy",0)
            avg = round((cl+dp+ac)/3,1)
            rows.append([
                Paragraph(str(i), body_s),
                Paragraph(qs.get("question","")[:80], body_s),
                Paragraph(str(cl), body_s),
                Paragraph(str(dp), body_s),
                Paragraph(str(ac), body_s),
                Paragraph(f"<b>{avg}</b>", body_s),
            ])
        t = Table(rows, colWidths=[0.3*inch,3.5*inch,0.7*inch,0.7*inch,0.9*inch,0.65*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f7f9ff")]),
            ("BOX",       (0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID", (0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("ALIGN",     (2,0),(-1,-1),"CENTER"),
            ("VALIGN",    (0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),4),
        ]))
        elems += [t, Spacer(1,12)]

    # Strengths/weaknesses
    strengths  = sc.get("strengths",[])
    weaknesses = sc.get("weaknesses",[])
    if strengths or weaknesses:
        elems.append(Paragraph("Strengths & Weaknesses", sec_s))
        sw_data = [[
            Paragraph("<b>✅ Strengths</b>", ParagraphStyle("SH", parent=styles["Normal"],
                      fontSize=10, textColor=colors.white, fontName="Helvetica-Bold")),
            Paragraph("<b>⚠️ Weaknesses</b>", ParagraphStyle("WH", parent=styles["Normal"],
                      fontSize=10, textColor=colors.white, fontName="Helvetica-Bold")),
        ]]
        for i in range(max(len(strengths),len(weaknesses))):
            s = f"• {strengths[i]}"  if i < len(strengths)  else ""
            w = f"• {weaknesses[i]}" if i < len(weaknesses) else ""
            sw_data.append([Paragraph(s, body_s), Paragraph(w, body_s)])
        swt = Table(sw_data, colWidths=[3.5*inch,3.5*inch])
        swt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,0), colors.HexColor("#27ae60")),
            ("BACKGROUND",(1,0),(1,0), colors.HexColor("#e74c3c")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),7),
        ]))
        elems += [swt, Spacer(1,12)]

    # Sentiment summary
    if sl:
        elems.append(Paragraph("Sentiment & Confidence Log", sec_s))
        sl_header = [Paragraph(f"<b>{h}</b>", body_s)
                     for h in ["#","Sentiment","Tone","Confidence","Filler Types","Words"]]
        sl_rows   = [sl_header]
        for i, sa in enumerate(sl, 1):
            sl_rows.append([
                Paragraph(str(i), body_s),
                Paragraph(f"{SENTIMENT_EMOJI.get(sa.get('sentiment',''),'')}{sa.get('sentiment','')}", body_s),
                Paragraph(sa.get("tone",""), body_s),
                Paragraph(f"{sa.get('confidence_score',0)}/10", body_s),
                Paragraph(str(len(sa.get("filler_words_detected",[]))), body_s),
                Paragraph(str(sa.get("word_count",0)), body_s),
            ])
        slt = Table(sl_rows, colWidths=[0.3*inch,1.2*inch,1.1*inch,1.1*inch,1.2*inch,1.0*inch])
        slt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9ff")]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("ALIGN",(2,0),(-1,-1),"CENTER"),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ]))
        elems += [slt, Spacer(1,12)]

    elems += [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")),
        Spacer(1,4),
        Paragraph("Generated by AI Interview Assistant — Admin Dashboard",
                  ParagraphStyle("Footer", parent=styles["Normal"],
                                 fontSize=8, textColor=colors.HexColor("#999999"), alignment=1))
    ]
    doc.build(elems)
    buf.seek(0)
    return buf


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown(f"### 🛡️ {st.session_state.username}")
    st.caption("Role: Administrator")
    st.divider()
    page = st.radio("Navigate", ["📊 Overview", "📋 All Interviews", "🔍 Interview Detail"])
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("app.py")


# ==========================
# PAGE: OVERVIEW
# ==========================
if "Overview" in page:
    st.title("📊 Admin Dashboard")
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %H:%M')}")

    stats = get_stats()

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Interviews",  stats["total"])
    k2.metric("Avg Overall Score", f"{stats['avg_score']} / 100")

    hire_counts = {r["hire_recommendation"]: r["cnt"] for r in stats["hire_counts"]}
    strong_hire = hire_counts.get("Strong Hire",0) + hire_counts.get("Hire",0)
    no_hire     = hire_counts.get("No Hire",0)
    k3.metric("✅ Hire / Strong Hire", strong_hire)
    k4.metric("❌ No Hire",            no_hire)

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Hire Recommendation Breakdown")
        if hire_counts:
            for label, count in hire_counts.items():
                pct   = round(count / stats["total"] * 100) if stats["total"] else 0
                emoji = HIRE_EMOJI.get(label,"⚪")
                color = HIRE_COLOR.get(label,"#888")
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>"
                    f"<span style='width:110px;font-size:0.9rem;'>{emoji} {label}</span>"
                    f"<div style='flex:1;background:#e0e0e0;border-radius:6px;height:14px;'>"
                    f"<div style='width:{pct}%;background:{color};border-radius:6px;height:14px;'>"
                    f"</div></div>"
                    f"<span style='width:70px;text-align:right;font-size:0.85rem;color:#666;'>"
                    f"{count} ({pct}%)</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No data yet.")

    with col_right:
        st.subheader("Interviews by Domain")
        if stats["domain_counts"]:
            domain_data = {r["domain"]: r["cnt"] for r in stats["domain_counts"]}
            st.bar_chart(domain_data, use_container_width=True, height=220)
        else:
            st.info("No data yet.")

    st.divider()
    st.subheader("🕐 Recent Interviews")
    if stats["recent"]:
        for r in stats["recent"]:
            hire  = r.get("hire_recommendation","N/A")
            emoji = HIRE_EMOJI.get(hire,"⚪")
            score = r.get("overall_score",0)
            score_color = "#27ae60" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;"
                f"padding:10px 14px;background:#f8f9fa;border-radius:10px;"
                f"margin-bottom:6px;border-left:4px solid {HIRE_COLOR.get(hire,'#ccc')};'>"
                f"<span style='font-weight:600;min-width:100px;'>👤 {r['username']}</span>"
                f"<span style='color:#666;min-width:140px;'>🏷️ {r['domain']}</span>"
                f"<span style='font-weight:700;color:{score_color};min-width:90px;'>"
                f"🎯 {score}/100</span>"
                f"<span>{emoji} {hire}</span>"
                f"<span style='margin-left:auto;color:#aaa;font-size:0.8rem;'>"
                f"🕒 {r['created_at']}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No interviews recorded yet.")

    # Refresh button
    if st.button("🔄 Refresh Stats"):
        st.rerun()


# ==========================
# PAGE: ALL INTERVIEWS
# ==========================
elif "All Interviews" in page:
    st.title("📋 All Interview Records")

    interviews = get_all_interviews()
    if not interviews:
        st.info("No interviews in the database yet.")
        st.stop()

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        all_users   = sorted(set(r["username"] for r in interviews))
        user_filter = st.multiselect("Filter by User", all_users, default=all_users)
    with f2:
        all_domains   = sorted(set(r["domain"] for r in interviews))
        domain_filter = st.multiselect("Filter by Domain", all_domains, default=all_domains)
    with f3:
        all_hire    = sorted(set(r["hire_recommendation"] for r in interviews))
        hire_filter = st.multiselect("Filter by Recommendation", all_hire, default=all_hire)

    filtered = [
        r for r in interviews
        if r["username"] in user_filter
        and r["domain"] in domain_filter
        and r["hire_recommendation"] in hire_filter
    ]

    st.caption(f"Showing **{len(filtered)}** of **{len(interviews)}** records")
    st.divider()

    if not filtered:
        st.warning("No records match your filters.")
    else:
        for r in filtered:
            hire  = r.get("hire_recommendation","N/A")
            emoji = HIRE_EMOJI.get(hire,"⚪")
            score = r.get("overall_score",0)
            score_color = "#27ae60" if score >= 75 else "#f39c12" if score >= 50 else "#e74c3c"

            with st.expander(
                f"#{r['id']} · {r['username']} · {r['domain']} · "
                f"{score}/100 · {emoji} {hire} · {r['created_at']}"
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score",          f"{score}/100")
                c2.metric("Domain",         r["domain"])
                c3.metric("Recommendation", f"{emoji} {hire}")
                c4.metric("Date",           r["created_at"][:10])

                col_view, col_pdf, col_del = st.columns([2, 2, 1])

                with col_view:
                    if st.button("🔍 View Full Detail", key=f"view_{r['id']}"):
                        st.session_state["detail_id"] = r["id"]
                        st.info("Switch to **🔍 Interview Detail** in the sidebar to view.")

                with col_pdf:
                    # Generate PDF inline for download
                    full_data = get_interview_by_id(r["id"])
                    if full_data:
                        try:
                            pdf_buf = generate_admin_pdf(full_data)
                            st.download_button(
                                "⬇️ Download PDF",
                                data=pdf_buf,
                                file_name=f"interview_{r['id']}_{r['username']}.pdf",
                                mime="application/pdf",
                                key=f"pdf_{r['id']}",
                            )
                        except Exception as e:
                            st.warning(f"PDF failed: {e}")

                with col_del:
                    if st.button("🗑️ Delete", key=f"del_{r['id']}", type="secondary"):
                        delete_interview(r["id"])
                        st.success(f"Deleted interview #{r['id']}")
                        st.rerun()


# ==========================
# PAGE: INTERVIEW DETAIL
# ==========================
elif "Detail" in page:
    st.title("🔍 Interview Detail Viewer")

    interviews = get_all_interviews()
    if not interviews:
        st.info("No interviews recorded yet.")
        st.stop()

    options = {
        f"#{r['id']} · {r['username']} · {r['domain']} · {r['created_at'][:10]}": r["id"]
        for r in interviews
    }

    # Pre-select from session if navigated from All Interviews
    default_idx = 0
    if "detail_id" in st.session_state:
        for idx, (label, rid) in enumerate(options.items()):
            if rid == st.session_state["detail_id"]:
                default_idx = idx
                break

    selected_label = st.selectbox(
        "Select an interview to inspect:",
        list(options.keys()),
        index=default_idx
    )
    interview_id = options[selected_label]
    data = get_interview_by_id(interview_id)

    if not data:
        st.error("Could not load this interview.")
        st.stop()

    sc = data["scorecard"]
    sl = data["sentiment_log"]
    tr = data["transcript"]

    hire       = sc.get("hire_recommendation","N/A")
    overall    = sc.get("overall_score",0)
    hire_emoji = HIRE_EMOJI.get(hire,"⚪")
    score_color = "#27ae60" if overall >= 75 else "#f39c12" if overall >= 50 else "#e74c3c"

    # Header
    st.markdown(
        f"<div style='background:#f0f4ff;border-radius:14px;padding:1.2rem 1.5rem;"
        f"border:1px solid #c0c8e8;margin-bottom:1rem;'>"
        f"<h3 style='margin:0;'>👤 {data['username']} &nbsp;|&nbsp; "
        f"🏷️ {data['domain']} &nbsp;|&nbsp; 🕒 {data['created_at']}</h3>"
        f"<div style='margin-top:8px;font-size:1.1rem;'>"
        f"🎯 Score: <b style='color:{score_color};'>{overall}/100</b>"
        f"&nbsp;&nbsp; Recommendation: <b>{hire_emoji} {hire}</b>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.write(sc.get("overall_summary",""))
    comm = sc.get("communication_insight","")
    if comm:
        st.info(f"🗣️ **Communication Insight:** {comm}")

    tabs = st.tabs(["📋 Scores", "🧠 Sentiment", "💬 Transcript", "📚 Recommendations"])

    # Tab 1: Scores
    with tabs[0]:
        st.subheader("Question-by-Question Scores")
        qs_list = sc.get("question_scores",[])
        if not qs_list:
            st.info("No question scores available.")
        else:
            for i, qs in enumerate(qs_list, 1):
                cl, dp, ac = qs.get("clarity",0), qs.get("depth",0), qs.get("technical_accuracy",0)
                avg = round((cl+dp+ac)/3, 1)
                with st.expander(f"Q{i} — {qs.get('question','')[:80]}"):
                    st.caption(qs.get("answer_summary",""))
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Clarity",  f"{cl}/10")
                    c2.metric("Depth",    f"{dp}/10")
                    c3.metric("Accuracy", f"{ac}/10")
                    c4.metric("Average",  f"{avg}/10")
                    st.progress(avg/10)

        st.divider()
        col_s, col_w = st.columns(2)
        with col_s:
            st.subheader("✅ Strengths")
            for s in sc.get("strengths",[]): st.success(s)
        with col_w:
            st.subheader("⚠️ Weaknesses")
            for w in sc.get("weaknesses",[]): st.warning(w)

    # Tab 2: Sentiment
    with tabs[1]:
        if not sl:
            st.info("No sentiment data for this interview.")
        else:
            avg_conf      = round(sum(s.get("confidence_score",0) for s in sl)/len(sl), 1)
            sentiments    = [s.get("sentiment","Neutral") for s in sl]
            total_fillers = sum(len(s.get("filler_words_detected",[])) for s in sl)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Avg Confidence",       f"{avg_conf}/10")
            m2.metric("😊 Positive Answers",  sentiments.count("Positive"))
            m3.metric("😐 Neutral Answers",   sentiments.count("Neutral"))
            m4.metric("⚠️ Total Filler Types", total_fillers)

            conf_scores = [s.get("confidence_score",0) for s in sl]
            if len(conf_scores) > 1:
                st.line_chart({"Confidence Score": conf_scores}, height=150)

            rows = []
            for i, sa in enumerate(sl, 1):
                s_e = SENTIMENT_EMOJI.get(sa.get("sentiment","Neutral"),"😐")
                t_e = TONE_EMOJI.get(sa.get("tone",""),"")
                rows.append({
                    "Answer":     f"#{i}",
                    "Sentiment":  f"{s_e} {sa.get('sentiment','')}",
                    "Tone":       f"{t_e} {sa.get('tone','')}",
                    "Confidence": f"{sa.get('confidence_score',0)}/10",
                    "Fillers":    ", ".join(sa.get("filler_words_detected",[])) or "None",
                    "Words":      sa.get("word_count",0),
                    "Top Phrase": sa.get("key_strength_phrase",""),
                    "Tip":        sa.get("improvement_tip",""),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # Tab 3: Transcript
    with tabs[2]:
        if not tr:
            st.info("No transcript available.")
        else:
            for msg in tr:
                role = msg.get("role","user")
                with st.chat_message(role):
                    st.write(msg.get("content",""))

    # Tab 4: Recommendations
    with tabs[3]:
        recs = sc.get("recommendations",[])
        if not recs:
            st.info("No recommendations recorded.")
        else:
            for rec in recs:
                with st.container(border=True):
                    st.markdown(f"**🎯 {rec.get('topic','')}**")
                    st.markdown(f"📖 *{rec.get('resource','')}*")
                    st.caption(rec.get("reason",""))

    st.divider()

    # Export actions
    col_json, col_pdf, col_del = st.columns(3)

    with col_json:
        export_data = {
            "id":            data["id"],
            "username":      data["username"],
            "domain":        data["domain"],
            "created_at":    data["created_at"],
            "scorecard":     sc,
            "sentiment_log": sl,
            "transcript":    tr,
        }
        st.download_button(
            "⬇️ Export as JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"interview_{data['id']}_{data['username']}.json",
            mime="application/json",
        )

    with col_pdf:
        try:
            pdf_buf = generate_admin_pdf(data)
            st.download_button(
                "📄 Export as PDF",
                data=pdf_buf,
                file_name=f"interview_{data['id']}_{data['username']}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF failed: {e}")

    with col_del:
        if st.button("🗑️ Delete This Interview", type="secondary"):
            delete_interview(interview_id)
            st.success("Interview deleted.")
            if "detail_id" in st.session_state:
                del st.session_state["detail_id"]
            st.rerun()