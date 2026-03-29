import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import whisper
from streamlit_mic_recorder import mic_recorder
from PyPDF2 import PdfReader
import os
import tempfile
import time
import json
import re
from datetime import datetime
import subprocess
import io

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import save_interview, init_db

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Interview Session · AI Assistant",
    page_icon="🎙️",
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

if st.session_state.role not in ("user", "admin"):
    st.error("⛔ Access denied.")
    if st.button("← Go to Login"):
        st.switch_page("app.py")
    st.stop()

# ==========================
# ENV + MODELS
# ==========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

FILLER_WORDS = [
    "um", "uh", "hmm", "like", "you know", "kind of", "sort of",
    "i think", "i guess", "maybe", "i mean", "actually", "basically",
    "literally", "right", "so", "well", "i'm not sure", "i don't know",
    "probably", "perhaps", "might be"
]

# ==========================
# SESSION STATE
# ==========================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("domain", "")
st.session_state.setdefault("interview_ended", False)
st.session_state.setdefault("audio_processed", False)
st.session_state.setdefault("scorecard", None)
st.session_state.setdefault("sentiment_log", [])
st.session_state.setdefault("saved_to_db", False)
st.session_state.setdefault("api_error", None)

SENTIMENT_EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}
TONE_EMOJI = {
    "Assertive": "💪", "Calm": "😌", "Hesitant": "🤔",
    "Nervous": "😰", "Enthusiastic": "🔥", "Flat": "😑"
}

# ==========================
# SIDEBAR NAV
# ==========================
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.caption("Role: Candidate")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("app.py")


# ==========================
# RATE-LIMIT SAFE API CALL
# ==========================
def call_gemini_with_retry(prompt: str, max_retries: int = 3, base_wait: int = 20) -> str:
    """
    Call Gemini with exponential backoff on 429 errors.
    Raises RuntimeError if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt).text
        except ResourceExhausted as e:
            # Parse retry_delay from error message if present
            wait = base_wait * (2 ** attempt)
            match = re.search(r"retry_delay \{ seconds: (\d+)", str(e))
            if match:
                wait = int(match.group(1)) + 2  # add 2s buffer

            if attempt < max_retries - 1:
                st.warning(
                    f"⏳ Gemini rate limit hit. Waiting {wait}s before retry "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"❌ Gemini API quota exhausted after {max_retries} attempts. "
                    "You've hit the free-tier daily limit (20 requests/day for gemini-2.5-flash). "
                    "Options:\n"
                    "- Wait until tomorrow for quota reset\n"
                    "- Upgrade to a paid Gemini plan at https://ai.dev\n"
                    "- Switch to `gemini-1.5-flash` which has higher free limits"
                )
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")


# ==========================
# UTILS
# ==========================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def get_ai_response(prompt, resume, domain):
    ctx = f"""You are a technical interviewer conducting a {domain} interview.
Resume: {resume}

Conversation so far:
{prompt}

Ask ONE concise, relevant interview question based on the candidate's answers and resume.
Do NOT repeat questions already asked. Do NOT add preamble like "Great answer!" - just ask the question."""
    return call_gemini_with_retry(ctx)


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# ==========================
# SENTIMENT & CONFIDENCE
# ==========================
def detect_fillers_locally(text: str) -> list:
    lower = text.lower()
    found = []
    for fw in FILLER_WORDS:
        pattern = r'\b' + re.escape(fw) + r'\b'
        matches = re.findall(pattern, lower)
        if matches:
            found.append(f"{fw} (x{len(matches)})")
    return found


def analyze_sentiment_locally(text: str) -> dict:
    """
    Lightweight local fallback for sentiment analysis — uses NO API quota.
    Used when Gemini quota is exhausted.
    """
    filler_hits = detect_fillers_locally(text)
    word_count  = len(text.split())
    lower       = text.lower()

    # Simple heuristic sentiment
    positive_words = ["good","great","excellent","strong","confident","sure","definitely","yes","absolutely","successfully","achieved","built","designed","implemented","solved","optimized"]
    negative_words = ["not sure","don't know","never","failed","difficult","confused","uncertain","wrong","bad","poor","struggle"]
    hesitant_phrases = ["i think","i guess","maybe","i'm not sure","kind of","sort of","probably","perhaps","might be","i don't know"]

    pos = sum(1 for w in positive_words if w in lower)
    neg = sum(1 for w in negative_words if w in lower)
    hes = sum(1 for w in hesitant_phrases if w in lower)

    sentiment = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")

    # Tone
    if hes >= 2:
        tone = "Hesitant"
    elif pos >= 3:
        tone = "Assertive"
    elif neg >= 2:
        tone = "Nervous"
    else:
        tone = "Calm"

    # Confidence score (1-10) based on length, fillers, hedges
    base_conf = min(10, max(1, word_count // 10))
    conf_penalty = min(4, len(filler_hits) + hes)
    confidence = max(1, base_conf - conf_penalty)

    return {
        "sentiment": sentiment,
        "sentiment_reason": "Estimated from keyword analysis (API quota exceeded).",
        "confidence_score": confidence,
        "confidence_reason": f"Based on {word_count} words and {len(filler_hits)} filler types.",
        "tone": tone,
        "key_strength_phrase": "",
        "improvement_tip": "Try to reduce filler words and be more direct in your answers." if filler_hits else "Keep up the clear communication!",
        "filler_words_detected": filler_hits,
        "word_count": word_count,
        "answer_text": text,
        "_used_fallback": True,
    }


def analyze_sentiment_confidence(text: str) -> dict:
    """Try Gemini first; fall back to local analysis if quota is hit."""
    filler_hits = detect_fillers_locally(text)
    word_count  = len(text.split())

    prompt = f"""You are a communication coach analyzing a spoken interview answer.
Answer text:
\"\"\"{text}\"\"\"
Return a JSON object with EXACTLY this structure (no markdown, no preamble):
{{
  "sentiment": "<Positive | Neutral | Negative>",
  "sentiment_reason": "<one sentence why>",
  "confidence_score": <integer 1-10>,
  "confidence_reason": "<one sentence why>",
  "tone": "<Assertive | Calm | Hesitant | Nervous | Enthusiastic | Flat>",
  "key_strength_phrase": "<the strongest phrase or empty string>",
  "improvement_tip": "<one concrete tip>"
}}
Return ONLY the raw JSON."""

    try:
        response = clean_json(call_gemini_with_retry(prompt))
        ai_result = json.loads(response)
        ai_result["filler_words_detected"] = filler_hits
        ai_result["word_count"]  = word_count
        ai_result["answer_text"] = text
        ai_result["_used_fallback"] = False
        return ai_result
    except RuntimeError:
        st.warning("⚠️ Gemini quota exceeded — using local analysis for this answer.")
        return analyze_sentiment_locally(text)
    except (json.JSONDecodeError, Exception):
        return analyze_sentiment_locally(text)


def confidence_bar_color(score: int) -> str:
    if score <= 3:   return "#e74c3c"
    elif score <= 6: return "#f39c12"
    return "#27ae60"


def render_sentiment_card(sa: dict, answer_num: int):
    sentiment = sa.get("sentiment", "Neutral")
    tone      = sa.get("tone", "Calm")
    conf      = sa.get("confidence_score", 5)
    fillers   = sa.get("filler_words_detected", [])
    tip       = sa.get("improvement_tip", "")
    ksp       = sa.get("key_strength_phrase", "")
    wc        = sa.get("word_count", 0)
    s_emoji   = SENTIMENT_EMOJI.get(sentiment, "😐")
    t_emoji   = TONE_EMOJI.get(tone, "")
    bar_color = confidence_bar_color(conf)
    fallback  = sa.get("_used_fallback", False)

    with st.container(border=True):
        label = f"**🧠 Answer #{answer_num} — Sentiment & Confidence**"
        if fallback:
            label += " *(local estimate — API quota exceeded)*"
        st.markdown(label)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sentiment",  f"{s_emoji} {sentiment}")
        col2.metric("Tone",       f"{t_emoji} {tone}")
        col3.metric("Confidence", f"{conf}/10")
        col4.metric("Words",      wc)
        st.markdown(
            f"<div style='background:#e0e0e0;border-radius:6px;height:10px;"
            f"margin:4px 0 10px 0;'>"
            f"<div style='width:{conf*10}%;background:{bar_color};"
            f"border-radius:6px;height:10px;'></div></div>",
            unsafe_allow_html=True,
        )
        if ksp:     st.success(f"💬 **Strongest phrase:** *\"{ksp}\"*")
        if fillers: st.warning(f"⚠️ **Filler words detected:** {', '.join(fillers)}")
        if tip:     st.info(f"💡 **Tip:** {tip}")


# ==========================
# SCORECARD GENERATION
# ==========================
def generate_scorecard(messages, resume, domain, sentiment_log):
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    sentiment_summary = ""
    for i, sa in enumerate(sentiment_log, 1):
        sentiment_summary += (
            f"Answer {i}: sentiment={sa.get('sentiment')}, "
            f"confidence={sa.get('confidence_score')}/10, "
            f"tone={sa.get('tone')}, "
            f"filler_types={len(sa.get('filler_words_detected', []))}\n"
        )
    prompt = f"""You are an expert technical interviewer evaluating a candidate for a {domain} role.
Resume: {resume}
Interview Transcript: {convo}
Sentiment & Confidence Log: {sentiment_summary}

Return a JSON object with EXACTLY this structure (no extra text, no markdown):
{{
  "overall_score": <integer 0-100>,
  "overall_summary": "<2-3 sentence summary>",
  "communication_insight": "<2-3 sentences on communication style>",
  "question_scores": [
    {{
      "question": "<the interviewer question>",
      "answer_summary": "<one-line summary>",
      "clarity": <integer 1-10>,
      "depth": <integer 1-10>,
      "technical_accuracy": <integer 1-10>
    }}
  ],
  "strengths":       ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses":      ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
  "recommendations": [
    {{
      "topic":    "<topic to improve>",
      "resource": "<specific book, course, or resource>",
      "reason":   "<why recommended>"
    }}
  ],
  "hire_recommendation": "<Strong Hire | Hire | Maybe | No Hire>"
}}
Return ONLY the raw JSON."""

    try:
        response = clean_json(call_gemini_with_retry(prompt, max_retries=3, base_wait=30))
        return json.loads(response)
    except RuntimeError as e:
        st.error(str(e))
        # Return a minimal scorecard so the app doesn't crash
        return {
            "overall_score": 0,
            "overall_summary": "Scorecard could not be generated due to API quota limits.",
            "communication_insight": "",
            "question_scores": [],
            "strengths": [],
            "weaknesses": ["API quota exhausted — please retry tomorrow or upgrade plan."],
            "recommendations": [],
            "hire_recommendation": "Maybe",
        }


# ==========================
# PDF HELPERS
# ==========================
def score_color_pdf(score, max_score=10):
    ratio = score / max_score
    if ratio >= 0.75: return colors.HexColor("#27ae60")
    elif ratio >= 0.5: return colors.HexColor("#f39c12")
    return colors.HexColor("#e74c3c")

def overall_color(score):
    if score >= 75: return colors.HexColor("#27ae60")
    elif score >= 50: return colors.HexColor("#f39c12")
    return colors.HexColor("#e74c3c")


def generate_pdf_report(scorecard, domain, sentiment_log):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles   = getSampleStyleSheet()
    title_s  = ParagraphStyle("T", parent=styles["Title"], fontSize=22,
                               textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    section_s= ParagraphStyle("S", parent=styles["Heading2"], fontSize=13,
                               textColor=colors.HexColor("#16213e"), spaceBefore=14, spaceAfter=6)
    body_s   = ParagraphStyle("B", parent=styles["Normal"], fontSize=10,
                               leading=14, textColor=colors.HexColor("#2c2c2c"))
    bullet_s = ParagraphStyle("BL", parent=styles["Normal"], fontSize=10,
                               leading=14, leftIndent=14, textColor=colors.HexColor("#2c2c2c"), bulletIndent=4)
    label_s  = ParagraphStyle("L", parent=styles["Normal"], fontSize=9,
                               textColor=colors.HexColor("#666666"), leading=12)
    small_s  = ParagraphStyle("SM", parent=styles["Normal"], fontSize=8,
                               leading=11, textColor=colors.HexColor("#444444"))
    elems = []

    # Header
    elems.append(Paragraph("AI Interview Report", title_s))
    elems.append(Paragraph(
        f"Domain: <b>{domain}</b> &nbsp;|&nbsp; "
        f"Date: <b>{datetime.now().strftime('%d %b %Y, %H:%M')}</b>", label_s))
    elems.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1a1a2e"), spaceAfter=10))

    hire    = scorecard.get("hire_recommendation","N/A")
    overall = scorecard.get("overall_score", 0)
    oc      = overall_color(overall)

    overall_data = [[
        Paragraph("<b>Overall Score</b>", body_s),
        Paragraph(f"<b>{overall} / 100</b>", ParagraphStyle(
            "OS", parent=styles["Normal"], fontSize=20, textColor=oc, alignment=1)),
        Paragraph("<b>Hire Recommendation</b>", body_s),
        Paragraph(f"<b>{hire}</b>", ParagraphStyle(
            "HR", parent=styles["Normal"], fontSize=13, textColor=oc, alignment=1)),
    ]]
    ot = Table(overall_data, colWidths=[1.5*inch, 1.5*inch, 2*inch, 2*inch])
    ot.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1), colors.HexColor("#f0f4ff")),
        ("BOX",       (0,0),(-1,-1), 1, colors.HexColor("#c0c8e8")),
        ("INNERGRID", (0,0),(-1,-1), 0.5, colors.HexColor("#c0c8e8")),
        ("ALIGN",     (0,0),(-1,-1), "CENTER"),
        ("VALIGN",    (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
    ]))
    elems += [ot, Spacer(1,6),
              Paragraph(scorecard.get("overall_summary",""), body_s)]
    comm = scorecard.get("communication_insight","")
    if comm:
        elems += [Spacer(1,4), Paragraph(f"<i>{comm}</i>", small_s)]
    elems.append(Spacer(1,12))

    # Per-question scores
    elems.append(Paragraph("Question-by-Question Scorecard", section_s))
    q_header = [Paragraph(f"<b>{h}</b>", body_s)
                for h in ["#","Question & Answer Summary","Clarity","Depth","Accuracy","Avg"]]
    q_rows = [q_header]
    for i, qs in enumerate(scorecard.get("question_scores",[]), 1):
        cl, dp, ac = qs.get("clarity",0), qs.get("depth",0), qs.get("technical_accuracy",0)
        avg = round((cl+dp+ac)/3, 1)
        q_rows.append([
            Paragraph(str(i), body_s),
            Paragraph(f"<b>Q:</b> {qs.get('question','')}<br/>"
                      f"<i>A: {qs.get('answer_summary','')}</i>", body_s),
            Paragraph(str(cl), body_s), Paragraph(str(dp), body_s),
            Paragraph(str(ac), body_s), Paragraph(f"<b>{avg}</b>", body_s),
        ])
    qt = Table(q_rows, colWidths=[0.3*inch,3.7*inch,0.65*inch,0.65*inch,0.85*inch,0.6*inch])
    qs_style = [
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
        ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f7f9ff")]),
        ("BOX",       (0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID", (0,0),(-1,-1), 0.25, colors.HexColor("#dddddd")),
        ("ALIGN",     (2,0),(-1,-1), "CENTER"),
        ("VALIGN",    (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",(0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",(0,0),(-1,-1), 5),
    ]
    for ri, qs_item in enumerate(scorecard.get("question_scores",[]), 1):
        for ci, sc_val in enumerate([qs_item.get("clarity",0), qs_item.get("depth",0), qs_item.get("technical_accuracy",0)], start=2):
            qs_style += [
                ("TEXTCOLOR",(ci,ri),(ci,ri), score_color_pdf(sc_val)),
                ("FONTNAME", (ci,ri),(ci,ri), "Helvetica-Bold"),
            ]
    qt.setStyle(TableStyle(qs_style))
    elems += [qt, Spacer(1,14)]

    # Strengths & Weaknesses
    elems.append(Paragraph("Strengths & Areas for Improvement", section_s))
    strengths  = scorecard.get("strengths",[])
    weaknesses = scorecard.get("weaknesses",[])
    sw_data = [[
        Paragraph("<b>✅ Strengths</b>", ParagraphStyle("SH", parent=styles["Normal"],
                  fontSize=10, textColor=colors.white, fontName="Helvetica-Bold")),
        Paragraph("<b>⚠️ Weaknesses</b>", ParagraphStyle("WH", parent=styles["Normal"],
                  fontSize=10, textColor=colors.white, fontName="Helvetica-Bold")),
    ]]
    for i in range(max(len(strengths), len(weaknesses))):
        s = f"• {strengths[i]}"  if i < len(strengths)  else ""
        w = f"• {weaknesses[i]}" if i < len(weaknesses) else ""
        sw_data.append([Paragraph(s, bullet_s), Paragraph(w, bullet_s)])
    swt = Table(sw_data, colWidths=[3.5*inch, 3.5*inch])
    swt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,0), colors.HexColor("#27ae60")),
        ("BACKGROUND",(1,0),(1,0), colors.HexColor("#e74c3c")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f9f9f9")]),
        ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.HexColor("#dddddd")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),8),
    ]))
    elems += [swt, Spacer(1,14)]

    # Recommendations
    elems.append(Paragraph("Personalized Learning Recommendations", section_s))
    rec_rows = [[Paragraph(f"<b>{h}</b>", body_s)
                 for h in ["Topic","Recommended Resource","Why?"]]]
    for rec in scorecard.get("recommendations",[]):
        rec_rows.append([
            Paragraph(rec.get("topic",""), body_s),
            Paragraph(f"<i>{rec.get('resource','')}</i>", body_s),
            Paragraph(rec.get("reason",""), body_s),
        ])
    if len(rec_rows) > 1:
        rect = Table(rec_rows, colWidths=[1.5*inch, 2.2*inch, 3.05*inch])
        rect.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f7f9ff")]),
            ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1), 0.25, colors.HexColor("#dddddd")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),
            ("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        elems.append(rect)

    elems += [
        Spacer(1,20),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")),
        Spacer(1,4),
        Paragraph("Generated by AI Interview Assistant · Powered by Gemini",
                  ParagraphStyle("Footer", parent=styles["Normal"],
                                 fontSize=8, textColor=colors.HexColor("#999999"), alignment=1))
    ]
    doc.build(elems)
    buf.seek(0)
    return buf


# ==========================
# SCORECARD UI
# ==========================
def render_scorecard_ui(scorecard, sentiment_log):
    overall   = scorecard.get("overall_score", 0)
    hire      = scorecard.get("hire_recommendation", "N/A")
    hire_icon = {"Strong Hire":"🟢","Hire":"🟡","Maybe":"🟠","No Hire":"🔴"}.get(hire,"⚪")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Score",       f"{overall} / 100")
    col2.metric("Hire Recommendation", f"{hire_icon} {hire}")
    col3.metric("Questions Evaluated", len(scorecard.get("question_scores",[])))

    st.write(scorecard.get("overall_summary",""))
    comm = scorecard.get("communication_insight","")
    if comm:
        st.info(f"🗣️ **Communication Insight:** {comm}")
    st.divider()

    if sentiment_log:
        st.subheader("🧠 Confidence & Sentiment Overview")
        avg_conf      = round(sum(s.get("confidence_score",0) for s in sentiment_log)/len(sentiment_log), 1)
        sentiments    = [s.get("sentiment","Neutral") for s in sentiment_log]
        total_fillers = sum(len(s.get("filler_words_detected",[])) for s in sentiment_log)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Avg Confidence",      f"{avg_conf}/10")
        mc2.metric("😊 Positive Answers", sentiments.count("Positive"))
        mc3.metric("😐 Neutral Answers",  sentiments.count("Neutral"))
        mc4.metric("⚠️ Filler Word Types", total_fillers)
        conf_scores = [s.get("confidence_score",0) for s in sentiment_log]
        if len(conf_scores) > 1:
            st.line_chart({"Confidence Score": conf_scores}, use_container_width=True, height=160)
        rows = []
        for i, sa in enumerate(sentiment_log, 1):
            fallback_note = " *(local estimate)*" if sa.get("_used_fallback") else ""
            rows.append({
                "Answer":       f"#{i}{fallback_note}",
                "Sentiment":    f"{SENTIMENT_EMOJI.get(sa.get('sentiment','Neutral'),'😐')} {sa.get('sentiment','')}",
                "Tone":         f"{TONE_EMOJI.get(sa.get('tone',''),'')}{sa.get('tone','')}",
                "Confidence":   f"{sa.get('confidence_score',0)}/10",
                "Filler Types": len(sa.get("filler_words_detected",[])),
                "Words":        sa.get("word_count",0),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.divider()

    st.subheader("📋 Question-by-Question Breakdown")
    qs_list = scorecard.get("question_scores",[])
    if not qs_list:
        st.info("No question scores available.")
    else:
        for i, qs in enumerate(qs_list, 1):
            badge = ""
            if i-1 < len(sentiment_log):
                sa    = sentiment_log[i-1]
                s_e   = SENTIMENT_EMOJI.get(sa.get("sentiment","Neutral"),"")
                badge = f" {s_e} conf:{sa.get('confidence_score','')}/10"
            with st.expander(f"Q{i}{badge} — {qs.get('question','')[:70]}..."):
                st.caption(f"**Answer Summary:** {qs.get('answer_summary','')}")
                c1,c2,c3 = st.columns(3)
                cl, dp, ac = qs.get("clarity",0), qs.get("depth",0), qs.get("technical_accuracy",0)
                avg = round((cl+dp+ac)/3, 1)
                c1.metric("Clarity",            f"{cl}/10")
                c2.metric("Depth",              f"{dp}/10")
                c3.metric("Technical Accuracy", f"{ac}/10")
                st.progress(avg/10, text=f"Average: {avg}/10")
                if i-1 < len(sentiment_log):
                    sa = sentiment_log[i-1]
                    if sa.get("key_strength_phrase"):
                        st.success(f"💬 Strongest phrase: *\"{sa['key_strength_phrase']}\"*")
                    if sa.get("filler_words_detected"):
                        st.warning(f"⚠️ Fillers: {', '.join(sa['filler_words_detected'])}")
                    if sa.get("improvement_tip"):
                        st.info(f"💡 {sa['improvement_tip']}")
    st.divider()

    col_s, col_w = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        for s in scorecard.get("strengths",[]):
            st.success(s)
    with col_w:
        st.subheader("⚠️ Weaknesses")
        for w in scorecard.get("weaknesses",[]):
            st.warning(w)
    st.divider()

    recs = scorecard.get("recommendations",[])
    if recs:
        st.subheader("📚 Learning Recommendations")
        for rec in recs:
            with st.container(border=True):
                st.markdown(f"**🎯 {rec.get('topic','')}**")
                st.markdown(f"📖 *{rec.get('resource','')}*")
                st.caption(rec.get("reason",""))


# ==========================
# MAIN
# ==========================
def main():
    left, right = st.columns([2, 3])

    with left:
        st.header("🎙️ Interview Session")
        st.caption(f"Logged in as **{st.session_state.username}**")

        # Show quota warning banner if we've seen errors
        if st.session_state.api_error:
            st.error(
                "⚠️ **Gemini API Quota Exhausted**\n\n"
                "Sentiment analysis is running locally. Interview questions may not work. "
                "Please wait for quota reset or upgrade your plan at https://ai.dev"
            )

        domain = st.selectbox(
            "Select Domain",
            ["Machine Learning","Data Science","Software Engineering","Cloud","Cybersecurity"]
        )
        st.session_state.domain = domain

        uploaded = st.file_uploader("Upload Resume (PDF)", type="pdf")
        if uploaded and not st.session_state.resume_text:
            with st.spinner("Extracting resume..."):
                st.session_state.resume_text = extract_text_from_pdf(uploaded)
            if len(st.session_state.messages) == 0:
                with st.spinner("Starting interview..."):
                    try:
                        first_q = get_ai_response(
                            "The candidate just uploaded their resume. Start the interview with a warm welcome and your first question.",
                            st.session_state.resume_text,
                            st.session_state.domain
                        )
                        st.session_state.messages.append({"role":"assistant","content":first_q})
                        st.rerun()
                    except RuntimeError as e:
                        st.session_state.api_error = str(e)
                        st.error(str(e))

        if not st.session_state.interview_ended:
            if st.button("🛑 End Interview & Analyze", type="primary", use_container_width=True):
                user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
                if len(user_msgs) < 1:
                    st.warning("Please answer at least one question before ending.")
                else:
                    with st.spinner("🧠 Generating scorecard..."):
                        scorecard = generate_scorecard(
                            st.session_state.messages,
                            st.session_state.resume_text,
                            st.session_state.domain,
                            st.session_state.sentiment_log,
                        )
                    st.session_state.scorecard = scorecard
                    st.session_state.interview_ended = True
                    st.rerun()

        # Live confidence tracker sidebar
        if not st.session_state.interview_ended and st.session_state.sentiment_log:
            st.divider()
            st.markdown("**📊 Live Confidence Tracker**")
            for i, sa in enumerate(st.session_state.sentiment_log, 1):
                conf      = sa.get("confidence_score", 0)
                bar_color = confidence_bar_color(conf)
                s_e = SENTIMENT_EMOJI.get(sa.get("sentiment","Neutral"),"")
                t_e = TONE_EMOJI.get(sa.get("tone",""),"")
                st.markdown(f"**A{i}:** {conf}/10 &nbsp; {s_e} &nbsp; {t_e}")
                st.markdown(
                    f"<div style='background:#e0e0e0;border-radius:4px;height:8px;margin-bottom:8px;'>"
                    f"<div style='width:{conf*10}%;background:{bar_color};border-radius:4px;height:8px;'>"
                    f"</div></div>", unsafe_allow_html=True,
                )

    with right:
        if st.session_state.interview_ended and st.session_state.scorecard:
            st.subheader("📊 Interview Scorecard")
            render_scorecard_ui(st.session_state.scorecard, st.session_state.sentiment_log)

            # Save to DB once
            if not st.session_state.saved_to_db:
                sc = st.session_state.scorecard
                try:
                    save_interview(
                        username=st.session_state.username,
                        domain=st.session_state.domain,
                        overall_score=sc.get("overall_score", 0),
                        hire_recommendation=sc.get("hire_recommendation", "N/A"),
                        scorecard=sc,
                        sentiment_log=st.session_state.sentiment_log,
                        messages=st.session_state.messages,
                    )
                    st.session_state.saved_to_db = True
                    st.toast("✅ Interview saved to database!", icon="💾")
                except Exception as e:
                    st.warning(f"Could not save to database: {e}")

            st.divider()
            try:
                pdf = generate_pdf_report(
                    st.session_state.scorecard,
                    st.session_state.domain,
                    st.session_state.sentiment_log,
                )
                st.download_button(
                    "⬇️ Download Full PDF Report", pdf,
                    file_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

            if st.button("🔄 Start New Interview", use_container_width=True):
                for key in ["messages","resume_text","domain","interview_ended",
                            "audio_processed","scorecard","sentiment_log","saved_to_db","api_error"]:
                    st.session_state[key] = (
                        [] if key in ["messages","sentiment_log"] else
                        False if key in ["interview_ended","audio_processed","saved_to_db"] else
                        None if key in ["scorecard","api_error"] else ""
                    )
                st.rerun()

        else:
            st.subheader("💬 Interview Session")

            if not st.session_state.resume_text:
                st.info("👈 Please upload your resume to start the interview.")
                return

            # Render conversation
            answer_idx = 0
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "user" and answer_idx < len(st.session_state.sentiment_log):
                        sa   = st.session_state.sentiment_log[answer_idx]
                        s_e  = SENTIMENT_EMOJI.get(sa.get("sentiment","Neutral"),"😐")
                        t_e  = TONE_EMOJI.get(sa.get("tone",""),"")
                        conf = sa.get("confidence_score", 0)
                        fb   = " *(local)*" if sa.get("_used_fallback") else ""
                        st.caption(
                            f"{s_e} {sa.get('sentiment','')} · "
                            f"{t_e} {sa.get('tone','')} · "
                            f"Confidence {conf}/10{fb}"
                        )
                        answer_idx += 1

            # ── Text input fallback + audio ─────────────────
            st.divider()
            tab_audio, tab_text = st.tabs(["🎙️ Voice Answer", "⌨️ Type Answer"])

            with tab_text:
                text_answer = st.text_area("Type your answer here:", height=120, key="text_answer_input")
                if st.button("Submit Answer", key="submit_text"):
                    if text_answer.strip():
                        _process_answer(text_answer.strip())

            with tab_audio:
                audio = mic_recorder(
                    start_prompt="🎙️ Start Recording",
                    stop_prompt="⏹️ Stop Recording",
                    key="recorder"
                )

                if audio and not st.session_state.audio_processed:
                    st.session_state.audio_processed = True
                    st.audio(audio["bytes"])

                    base_path = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time())}")
                    webm_path = base_path + ".webm"
                    wav_path  = base_path + ".wav"

                    with open(webm_path, "wb") as f:
                        f.write(audio["bytes"])

                    result_ffmpeg = subprocess.run([
                        "ffmpeg","-y","-i",webm_path,"-ar","16000","-ac","1",wav_path
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if result_ffmpeg.returncode != 0:
                        st.error("❌ Audio conversion failed. Please try the text input tab.")
                        st.session_state.audio_processed = False
                    else:
                        with st.spinner("Transcribing..."):
                            result = whisper_model.transcribe(wav_path)
                            text   = result["text"].strip()

                        if not text:
                            st.error("❌ Could not transcribe audio. Try speaking more clearly or use text input.")
                            st.session_state.audio_processed = False
                        else:
                            st.success(f"📝 You said: *{text}*")
                            _process_answer(text)

                    # Cleanup temp files
                    for path in [webm_path, wav_path]:
                        if os.path.exists(path):
                            os.remove(path)

                if not audio:
                    st.session_state.audio_processed = False


def _process_answer(text: str):
    """Shared logic for processing a submitted answer (voice or text)."""
    answer_num = len(st.session_state.sentiment_log) + 1

    with st.spinner("🧠 Analyzing tone & confidence..."):
        sa = analyze_sentiment_confidence(text)

    st.session_state.sentiment_log.append(sa)
    render_sentiment_card(sa, answer_num)

    st.session_state.messages.append({"role": "user", "content": text})

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
    with st.spinner("💭 Generating next question..."):
        try:
            reply = get_ai_response(convo, st.session_state.resume_text, st.session_state.domain)
        except RuntimeError as e:
            st.session_state.api_error = str(e)
            reply = "I'm unable to generate the next question due to API quota limits. Please end the interview and view your scorecard, or try again tomorrow."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.audio_processed = False
    st.rerun()


if __name__ == "__main__":
    main()