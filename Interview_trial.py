import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError
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
# SESSION STATE DEFAULTS
# ==========================
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("role", None)
st.session_state.setdefault("username", "")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("domain", "")
st.session_state.setdefault("interview_ended", False)
st.session_state.setdefault("audio_processed", False)
st.session_state.setdefault("last_audio_id", None)   # tracks which audio clip was last processed
st.session_state.setdefault("processing_answer", False)  # prevents double submission
st.session_state.setdefault("scorecard", None)
st.session_state.setdefault("sentiment_log", [])
st.session_state.setdefault("saved_to_db", False)
st.session_state.setdefault("api_error", None)

# ==========================
# AUTH GUARD
# ==========================
if not st.session_state.logged_in:
    st.error("🔒 Please log in first.")
    st.page_link("app.py", label="← Go to Login", icon="🔑")
    st.stop()

if st.session_state.role not in ("user", "admin"):
    st.error("⛔ Access denied.")
    st.page_link("app.py", label="← Go to Login", icon="🔑")
    st.stop()

# ==========================
# ENV + GROQ CLIENT
# ==========================
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model to use — llama-3.3-70b is free and very capable
# Fallback: llama3-8b-8192 (faster, lighter)
GROQ_MODEL = "llama-3.3-70b-versatile"

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

FILLER_WORDS = [
    "um","uh","hmm","like","you know","kind of","sort of",
    "i think","i guess","maybe","i mean","actually","basically",
    "literally","right","so","well","i'm not sure","i don't know",
    "probably","perhaps","might be"
]

SENTIMENT_EMOJI = {"Positive":"😊","Neutral":"😐","Negative":"😟"}
TONE_EMOJI = {
    "Assertive":"💪","Calm":"😌","Hesitant":"🤔",
    "Nervous":"😰","Enthusiastic":"🔥","Flat":"😑"
}

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    st.caption("Role: Candidate")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Logged out.")
        st.page_link("app.py", label="← Back to Login", icon="🔑")
        st.stop()


# ==========================
# GROQ API CALL WITH RETRY
# ==========================
def call_groq(messages: list, max_retries: int = 3) -> str:
    """
    Call Groq API with retry on rate limit.
    `messages` is a list of {"role": ..., "content": ...} dicts.
    """
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            wait = 30 * (attempt + 1)
            # Try to read retry-after from headers if available
            if hasattr(e, 'response') and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait = int(retry_after) + 2
            if attempt < max_retries - 1:
                st.warning(f"⏳ Rate limit hit. Waiting {wait}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    "❌ Groq API rate limit exceeded after retries.\n"
                    "This resets every minute — please wait a moment and try again."
                )
        except Exception as e:
            raise RuntimeError(f"Groq API error: {e}")


# ==========================
# UTILS
# ==========================
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def get_ai_response(conversation_history: list, resume: str, domain: str) -> str:
    """Get next interview question from Groq."""
    system_prompt = f"""You are a professional technical interviewer conducting a {domain} interview.
The candidate's resume is provided below for context on their background.
Resume:
{resume}

STRICT RULES — follow these exactly:
1. Ask ONE technical question at a time about {domain}
2. NEVER comment on the candidate's name or identity — names in answers are irrelevant, ignore them
3. NEVER say the candidate is wrong person, mixed up, or not the correct candidate
4. NEVER repeat a question that was already asked in this conversation
5. NEVER say things like "Great answer!", "Good job!", or give feedback — just ask the next question
6. If an answer seems off-topic or unclear, acknowledge briefly and move to the next technical question
7. After 5-6 questions, you may wrap up by saying "Thank you, that concludes our interview."
8. Base questions on {domain} concepts and the resume background — stay technical"""

    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    return call_groq(messages)


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# ==========================
# SENTIMENT ANALYSIS
# ==========================
def detect_fillers_locally(text: str) -> list:
    lower = text.lower()
    found = []
    for fw in FILLER_WORDS:
        matches = re.findall(r'\b' + re.escape(fw) + r'\b', lower)
        if matches:
            found.append(f"{fw} (x{len(matches)})")
    return found


def analyze_sentiment_locally(text: str) -> dict:
    """Zero-API fallback using keyword heuristics."""
    filler_hits = detect_fillers_locally(text)
    word_count  = len(text.split())
    lower       = text.lower()

    pos_words = ["good","great","excellent","strong","confident","sure","definitely",
                 "successfully","achieved","built","designed","implemented","solved","optimized"]
    neg_words = ["not sure","don't know","never","failed","difficult","confused","uncertain","wrong"]
    hes_words = ["i think","i guess","maybe","i'm not sure","kind of","sort of",
                 "probably","perhaps","might be","i don't know"]

    pos = sum(1 for w in pos_words if w in lower)
    neg = sum(1 for w in neg_words if w in lower)
    hes = sum(1 for w in hes_words if w in lower)

    sentiment  = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
    tone       = "Hesitant" if hes >= 2 else ("Assertive" if pos >= 3 else ("Nervous" if neg >= 2 else "Calm"))
    confidence = max(1, min(10, word_count // 10) - min(4, len(filler_hits) + hes))

    return {
        "sentiment":             sentiment,
        "sentiment_reason":      "Estimated from keyword analysis.",
        "confidence_score":      confidence,
        "confidence_reason":     f"Based on {word_count} words and {len(filler_hits)} filler types.",
        "tone":                  tone,
        "key_strength_phrase":   "",
        "improvement_tip":       "Reduce filler words and be more direct." if filler_hits else "Good clarity!",
        "filler_words_detected": filler_hits,
        "word_count":            word_count,
        "answer_text":           text,
        "_used_fallback":        True,
    }


def analyze_sentiment_confidence(text: str) -> dict:
    filler_hits = detect_fillers_locally(text)
    word_count  = len(text.split())

    system = "You are a communication coach. Respond ONLY with raw JSON, no markdown, no explanation."
    user   = f"""Analyze this spoken interview answer:
\"\"\"{text}\"\"\"

Return ONLY this JSON structure:
{{
  "sentiment": "<Positive | Neutral | Negative>",
  "sentiment_reason": "<one sentence>",
  "confidence_score": <integer 1-10>,
  "confidence_reason": "<one sentence>",
  "tone": "<Assertive | Calm | Hesitant | Nervous | Enthusiastic | Flat>",
  "key_strength_phrase": "<strongest phrase or empty string>",
  "improvement_tip": "<one concrete tip>"
}}"""

    try:
        raw    = call_groq([{"role":"system","content":system},{"role":"user","content":user}])
        result = json.loads(clean_json(raw))
        result["confidence_score"]      = int(result.get("confidence_score", 5) or 5)
        result["filler_words_detected"] = filler_hits
        result["word_count"]            = word_count
        result["answer_text"]           = text
        result["_used_fallback"]        = False
        return result
    except Exception:
        return analyze_sentiment_locally(text)


def confidence_bar_color(score) -> str:
    score = int(score) if score else 0
    if score <= 3:   return "#e74c3c"
    elif score <= 6: return "#f39c12"
    return "#27ae60"


def render_sentiment_card(sa: dict, answer_num: int):
    sentiment = sa.get("sentiment","Neutral")
    tone      = sa.get("tone","Calm")
    conf      = int(sa.get("confidence_score", 5) or 5)
    fillers   = sa.get("filler_words_detected",[])
    tip       = sa.get("improvement_tip","")
    ksp       = sa.get("key_strength_phrase","")
    wc        = int(sa.get("word_count", 0) or 0)
    fallback  = sa.get("_used_fallback",False)
    bar_color = confidence_bar_color(conf)

    with st.container(border=True):
        label = f"**🧠 Answer #{answer_num} — Sentiment & Confidence**"
        if fallback:
            label += " *(local estimate)*"
        st.markdown(label)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sentiment",  f"{SENTIMENT_EMOJI.get(sentiment,'😐')} {sentiment}")
        c2.metric("Tone",       f"{TONE_EMOJI.get(tone,'')} {tone}")
        c3.metric("Confidence", f"{conf}/10")
        c4.metric("Words",      wc)
        st.markdown(
            f"<div style='background:#e0e0e0;border-radius:6px;height:10px;margin:4px 0 10px 0;'>"
            f"<div style='width:{conf*10}%;background:{bar_color};border-radius:6px;height:10px;'>"
            f"</div></div>", unsafe_allow_html=True,
        )
        if ksp:     st.success(f"💬 **Strongest phrase:** *\"{ksp}\"*")
        if fillers: st.warning(f"⚠️ **Filler words:** {', '.join(fillers)}")
        if tip:     st.info(f"💡 **Tip:** {tip}")


# ==========================
# SCORECARD GENERATION
# ==========================
def generate_scorecard(messages, resume, domain, sentiment_log):
    # Build transcript string
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
    sentiment_summary = "".join(
        f"Answer {i}: sentiment={sa.get('sentiment')}, confidence={sa.get('confidence_score')}/10, "
        f"tone={sa.get('tone')}, fillers={len(sa.get('filler_words_detected',[]))}\n"
        for i, sa in enumerate(sentiment_log, 1)
    )
    system = "You are an expert technical interviewer. Respond ONLY with raw JSON, no markdown."
    user   = f"""Evaluate this {domain} candidate.
Resume: {resume}
Transcript:
{convo}
Sentiment Log:
{sentiment_summary}

Return ONLY this JSON:
{{
  "overall_score": <0-100>,
  "overall_summary": "<2-3 sentences>",
  "communication_insight": "<2-3 sentences>",
  "question_scores": [
    {{"question":"<q>","answer_summary":"<one line>","clarity":<1-10>,"depth":<1-10>,"technical_accuracy":<1-10>}}
  ],
  "strengths":       ["<s1>","<s2>","<s3>"],
  "weaknesses":      ["<w1>","<w2>","<w3>"],
  "recommendations": [{{"topic":"<t>","resource":"<r>","reason":"<why>"}}],
  "hire_recommendation": "<Strong Hire | Hire | Maybe | No Hire>"
}}"""

    try:
        raw = call_groq([{"role":"system","content":system},{"role":"user","content":user}],
                        max_retries=3)
        return json.loads(clean_json(raw))
    except RuntimeError as e:
        st.error(str(e))
        return {
            "overall_score": 0,
            "overall_summary": "Scorecard unavailable — API error.",
            "communication_insight": "",
            "question_scores": [],
            "strengths": [],
            "weaknesses": ["Could not generate scorecard."],
            "recommendations": [],
            "hire_recommendation": "Maybe",
        }


# ==========================
# PDF REPORT
# ==========================
def generate_pdf_report(scorecard, domain, sentiment_log):
    buf     = io.BytesIO()
    doc     = SimpleDocTemplate(buf, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch,  bottomMargin=0.75*inch)
    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle("T",  parent=styles["Title"],   fontSize=22, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    sec_s   = ParagraphStyle("S",  parent=styles["Heading2"],fontSize=13, textColor=colors.HexColor("#16213e"), spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=10, leading=14, textColor=colors.HexColor("#2c2c2c"))
    bullet_s= ParagraphStyle("BL", parent=styles["Normal"],  fontSize=10, leading=14, leftIndent=14, textColor=colors.HexColor("#2c2c2c"))
    label_s = ParagraphStyle("L",  parent=styles["Normal"],  fontSize=9,  textColor=colors.HexColor("#666666"), leading=12)
    small_s = ParagraphStyle("SM", parent=styles["Normal"],  fontSize=8,  leading=11, textColor=colors.HexColor("#444444"))
    elems   = []

    hire    = scorecard.get("hire_recommendation","N/A")
    overall = scorecard.get("overall_score",0)
    oc      = colors.HexColor("#27ae60" if overall>=75 else "#f39c12" if overall>=50 else "#e74c3c")

    elems.append(Paragraph("AI Interview Report", title_s))
    elems.append(Paragraph(
        f"Domain: <b>{domain}</b> &nbsp;|&nbsp; Date: <b>{datetime.now().strftime('%d %b %Y, %H:%M')}</b>",
        label_s))
    elems.append(HRFlowable(width="100%",thickness=1.5,color=colors.HexColor("#1a1a2e"),spaceAfter=10))

    ot = Table([[
        Paragraph("<b>Overall Score</b>", body_s),
        Paragraph(f"<b>{overall}/100</b>", ParagraphStyle("OS",parent=styles["Normal"],fontSize=20,textColor=oc,alignment=1)),
        Paragraph("<b>Hire Recommendation</b>", body_s),
        Paragraph(f"<b>{hire}</b>", ParagraphStyle("HR",parent=styles["Normal"],fontSize=13,textColor=oc,alignment=1)),
    ]], colWidths=[1.5*inch]*4)
    ot.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f0f4ff")),
        ("BOX",(0,0),(-1,-1),1,colors.HexColor("#c0c8e8")),
        ("INNERGRID",(0,0),(-1,-1),0.5,colors.HexColor("#c0c8e8")),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
    ]))
    elems += [ot, Spacer(1,6), Paragraph(scorecard.get("overall_summary",""),body_s)]
    if scorecard.get("communication_insight"):
        elems += [Spacer(1,4), Paragraph(f"<i>{scorecard['communication_insight']}</i>",small_s)]
    elems.append(Spacer(1,12))

    qs_list = scorecard.get("question_scores",[])
    if qs_list:
        elems.append(Paragraph("Question-by-Question Scorecard", sec_s))
        q_rows = [[Paragraph(f"<b>{h}</b>",body_s) for h in ["#","Question & Summary","Clarity","Depth","Accuracy","Avg"]]]
        for i, qs in enumerate(qs_list, 1):
            cl,dp,ac = qs.get("clarity",0),qs.get("depth",0),qs.get("technical_accuracy",0)
            q_rows.append([
                Paragraph(str(i),body_s),
                Paragraph(f"<b>Q:</b> {qs.get('question','')}<br/><i>A: {qs.get('answer_summary','')}</i>",body_s),
                Paragraph(str(cl),body_s),Paragraph(str(dp),body_s),
                Paragraph(str(ac),body_s),Paragraph(f"<b>{round((cl+dp+ac)/3,1)}</b>",body_s),
            ])
        qt = Table(q_rows, colWidths=[0.3*inch,3.7*inch,0.65*inch,0.65*inch,0.85*inch,0.6*inch])
        qt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16213e")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9ff")]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("ALIGN",(2,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        elems += [qt, Spacer(1,14)]

    strengths  = scorecard.get("strengths",[])
    weaknesses = scorecard.get("weaknesses",[])
    if strengths or weaknesses:
        elems.append(Paragraph("Strengths & Areas for Improvement", sec_s))
        sw_data = [[
            Paragraph("<b>✅ Strengths</b>",  ParagraphStyle("SH",parent=styles["Normal"],fontSize=10,textColor=colors.white,fontName="Helvetica-Bold")),
            Paragraph("<b>⚠️ Weaknesses</b>", ParagraphStyle("WH",parent=styles["Normal"],fontSize=10,textColor=colors.white,fontName="Helvetica-Bold")),
        ]]
        for i in range(max(len(strengths),len(weaknesses))):
            sw_data.append([
                Paragraph(f"• {strengths[i]}"  if i < len(strengths)  else "", bullet_s),
                Paragraph(f"• {weaknesses[i]}" if i < len(weaknesses) else "", bullet_s),
            ])
        swt = Table(sw_data, colWidths=[3.5*inch,3.5*inch])
        swt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(0,0),colors.HexColor("#27ae60")),
            ("BACKGROUND",(1,0),(1,0),colors.HexColor("#e74c3c")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f9f9f9")]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),8),
        ]))
        elems += [swt, Spacer(1,14)]

    recs = scorecard.get("recommendations",[])
    if recs:
        elems.append(Paragraph("Personalized Learning Recommendations", sec_s))
        rec_rows = [[Paragraph(f"<b>{h}</b>",body_s) for h in ["Topic","Resource","Why?"]]]
        for rec in recs:
            rec_rows.append([
                Paragraph(rec.get("topic",""),body_s),
                Paragraph(f"<i>{rec.get('resource','')}</i>",body_s),
                Paragraph(rec.get("reason",""),body_s),
            ])
        rect = Table(rec_rows, colWidths=[1.5*inch,2.2*inch,3.05*inch])
        rect.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16213e")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9ff")]),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.HexColor("#dddddd")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),5),
        ]))
        elems.append(rect)

    elems += [
        Spacer(1,20),
        HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#aaaaaa")),
        Spacer(1,4),
        Paragraph("Generated by AI Interview Assistant · Powered by Groq + Llama",
                  ParagraphStyle("Footer",parent=styles["Normal"],fontSize=8,
                                 textColor=colors.HexColor("#999999"),alignment=1))
    ]
    doc.build(elems)
    buf.seek(0)
    return buf


# ==========================
# SCORECARD UI
# ==========================
def render_scorecard_ui(scorecard, sentiment_log):
    overall   = scorecard.get("overall_score",0)
    hire      = scorecard.get("hire_recommendation","N/A")
    hire_icon = {"Strong Hire":"🟢","Hire":"🟡","Maybe":"🟠","No Hire":"🔴"}.get(hire,"⚪")

    c1,c2,c3 = st.columns(3)
    c1.metric("Overall Score",       f"{overall} / 100")
    c2.metric("Hire Recommendation", f"{hire_icon} {hire}")
    c3.metric("Questions Evaluated", len(scorecard.get("question_scores",[])))

    st.write(scorecard.get("overall_summary",""))
    if scorecard.get("communication_insight"):
        st.info(f"🗣️ **Communication Insight:** {scorecard['communication_insight']}")
    st.divider()

    if sentiment_log:
        st.subheader("🧠 Confidence & Sentiment Overview")
        avg_conf = round(sum(int(s.get("confidence_score",0) or 0) for s in sentiment_log)/len(sentiment_log),1)
        sentiments = [s.get("sentiment","Neutral") for s in sentiment_log]
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("Avg Confidence",       f"{avg_conf}/10")
        mc2.metric("😊 Positive Answers",  sentiments.count("Positive"))
        mc3.metric("😐 Neutral Answers",   sentiments.count("Neutral"))
        mc4.metric("⚠️ Filler Word Types", sum(len(s.get("filler_words_detected",[])) for s in sentiment_log))
        conf_scores = [int(s.get("confidence_score",0) or 0) for s in sentiment_log]
        if len(conf_scores) > 1:
            st.line_chart({"Confidence Score": conf_scores}, use_container_width=True, height=160)
        rows = []
        for i, sa in enumerate(sentiment_log, 1):
            rows.append({
                "Answer":       f"#{i}" + (" *(local)*" if sa.get("_used_fallback") else ""),
                "Sentiment":    f"{SENTIMENT_EMOJI.get(sa.get('sentiment','Neutral'),'😐')} {sa.get('sentiment','')}",
                "Tone":         f"{TONE_EMOJI.get(sa.get('tone',''),'')}{sa.get('tone','')}",
                "Confidence":   f"{int(sa.get('confidence_score',0) or 0)}/10",
                "Filler Types": len(sa.get("filler_words_detected",[])),
                "Words":        sa.get("word_count",0),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.divider()

    st.subheader("📋 Question-by-Question Breakdown")
    for i, qs in enumerate(scorecard.get("question_scores",[]), 1):
        badge = ""
        if i-1 < len(sentiment_log):
            sa    = sentiment_log[i-1]
            badge = f" {SENTIMENT_EMOJI.get(sa.get('sentiment','Neutral'),'')} conf:{int(sa.get('confidence_score',0) or 0)}/10"
        with st.expander(f"Q{i}{badge} — {qs.get('question','')[:70]}..."):
            st.caption(f"**Answer Summary:** {qs.get('answer_summary','')}")
            cl,dp,ac = qs.get("clarity",0),qs.get("depth",0),qs.get("technical_accuracy",0)
            c1,c2,c3 = st.columns(3)
            c1.metric("Clarity",           f"{cl}/10")
            c2.metric("Depth",             f"{dp}/10")
            c3.metric("Technical Accuracy",f"{ac}/10")
            st.progress(round((cl+dp+ac)/3,1)/10, text=f"Average: {round((cl+dp+ac)/3,1)}/10")
            if i-1 < len(sentiment_log):
                sa = sentiment_log[i-1]
                if sa.get("key_strength_phrase"): st.success(f"💬 *\"{sa['key_strength_phrase']}\"*")
                if sa.get("filler_words_detected"): st.warning(f"⚠️ Fillers: {', '.join(sa['filler_words_detected'])}")
                if sa.get("improvement_tip"): st.info(f"💡 {sa['improvement_tip']}")
    st.divider()

    col_s, col_w = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        for s in scorecard.get("strengths",[]): st.success(s)
    with col_w:
        st.subheader("⚠️ Weaknesses")
        for w in scorecard.get("weaknesses",[]): st.warning(w)

    recs = scorecard.get("recommendations",[])
    if recs:
        st.divider()
        st.subheader("📚 Learning Recommendations")
        for rec in recs:
            with st.container(border=True):
                st.markdown(f"**🎯 {rec.get('topic','')}**")
                st.markdown(f"📖 *{rec.get('resource','')}*")
                st.caption(rec.get("reason",""))


# ==========================
# PROCESS ANSWER
# ==========================
MAX_QUESTIONS = 6  # Interview ends after this many answers

def process_answer(text: str):
    # Guard: prevent double processing on rerun
    if st.session_state.processing_answer:
        return
    st.session_state.processing_answer = True

    answer_num = len(st.session_state.sentiment_log) + 1
    with st.spinner("🧠 Analysing tone & confidence..."):
        sa = analyze_sentiment_confidence(text)
    st.session_state.sentiment_log.append(sa)
    render_sentiment_card(sa, answer_num)

    # Add to conversation history in Groq format
    st.session_state.messages.append({"role":"user","content":text})

    # Auto-end after MAX_QUESTIONS answers
    if answer_num >= MAX_QUESTIONS:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thank you for your answers — that concludes our interview. Click **End Interview & Analyze** to see your scorecard."
        })
        st.session_state.audio_processed   = False
        st.session_state.processing_answer  = False
        st.rerun()
        return

    with st.spinner("💭 Generating next question..."):
        try:
            reply = get_ai_response(
                st.session_state.messages,
                st.session_state.resume_text,
                st.session_state.domain
            )
        except RuntimeError as e:
            st.session_state.api_error = str(e)
            reply = "Unable to generate next question right now. You can end the interview to see your scorecard."

    st.session_state.messages.append({"role":"assistant","content":reply})
    st.session_state.audio_processed  = False
    st.session_state.processing_answer = False
    st.rerun()


# ==========================
# MAIN
# ==========================
def main():
    left, right = st.columns([2, 3])

    with left:
        st.header("🎙️ Interview Session")
        st.caption(f"Logged in as **{st.session_state.username}**")

        if st.session_state.api_error:
            st.error(f"⚠️ {st.session_state.api_error}")

        domain = st.selectbox("Select Domain",
            ["Machine Learning","Data Science","Software Engineering","Cloud","Cybersecurity"])
        st.session_state.domain = domain

        uploaded = st.file_uploader("Upload Resume (PDF)", type="pdf")
        if uploaded and not st.session_state.resume_text:
            with st.spinner("Extracting resume..."):
                st.session_state.resume_text = extract_text_from_pdf(uploaded)
            if not st.session_state.messages:
                with st.spinner("Starting interview..."):
                    try:
                        first_q = get_ai_response(
                            [],   # empty history — first question
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
                if not any(m["role"] == "user" for m in st.session_state.messages):
                    st.warning("Please answer at least one question first.")
                else:
                    with st.spinner("🧠 Generating scorecard..."):
                        scorecard = generate_scorecard(
                            st.session_state.messages,
                            st.session_state.resume_text,
                            st.session_state.domain,
                            st.session_state.sentiment_log,
                        )
                    st.session_state.scorecard       = scorecard
                    st.session_state.interview_ended = True
                    st.rerun()

        if not st.session_state.interview_ended and st.session_state.sentiment_log:
            st.divider()
            st.markdown("**📊 Live Confidence Tracker**")
            for i, sa in enumerate(st.session_state.sentiment_log, 1):
                conf      = int(sa.get("confidence_score",0) or 0)
                bar_color = confidence_bar_color(conf)
                st.markdown(
                    f"**A{i}:** {conf}/10 &nbsp; "
                    f"{SENTIMENT_EMOJI.get(sa.get('sentiment','Neutral'),'')} &nbsp; "
                    f"{TONE_EMOJI.get(sa.get('tone',''),'')}"
                )
                st.markdown(
                    f"<div style='background:#e0e0e0;border-radius:4px;height:8px;margin-bottom:8px;'>"
                    f"<div style='width:{conf*10}%;background:{bar_color};border-radius:4px;height:8px;'>"
                    f"</div></div>", unsafe_allow_html=True,
                )

    with right:
        if st.session_state.interview_ended and st.session_state.scorecard:
            st.subheader("📊 Interview Scorecard")
            render_scorecard_ui(st.session_state.scorecard, st.session_state.sentiment_log)

            if not st.session_state.saved_to_db:
                sc = st.session_state.scorecard
                try:
                    save_interview(
                        username=st.session_state.username,
                        domain=st.session_state.domain,
                        overall_score=sc.get("overall_score",0),
                        hire_recommendation=sc.get("hire_recommendation","N/A"),
                        scorecard=sc,
                        sentiment_log=st.session_state.sentiment_log,
                        messages=st.session_state.messages,
                    )
                    st.session_state.saved_to_db = True
                    st.toast("✅ Interview saved!", icon="💾")
                except Exception as e:
                    st.warning(f"Could not save: {e}")

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
                    mime="application/pdf", use_container_width=True, type="primary"
                )
            except Exception as e:
                st.warning(f"PDF failed: {e}")

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
                st.info("👈 Upload your resume on the left to begin.")
                return

            # Render conversation
            answer_idx = 0
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "user" and answer_idx < len(st.session_state.sentiment_log):
                        sa   = st.session_state.sentiment_log[answer_idx]
                        conf = sa.get("confidence_score",0)
                        fb   = " *(local)*" if sa.get("_used_fallback") else ""
                        st.caption(
                            f"{SENTIMENT_EMOJI.get(sa.get('sentiment','Neutral'),'😐')} {sa.get('sentiment','')} · "
                            f"{TONE_EMOJI.get(sa.get('tone',''),'')}{sa.get('tone','')} · "
                            f"Confidence {conf}/10{fb}"
                        )
                        answer_idx += 1

            st.divider()

            # Hide input once max questions answered
            answers_given = len(st.session_state.sentiment_log)
            if answers_given >= MAX_QUESTIONS:
                st.success(f"✅ You've completed all {MAX_QUESTIONS} questions! Click **End Interview & Analyze** on the left to see your scorecard.")
            else:
                st.caption(f"Question {answers_given + 1} of {MAX_QUESTIONS}")
                tab_voice, tab_text = st.tabs(["🎙️ Voice Answer", "⌨️ Type Answer"])

                with tab_text:
                    text_ans = st.text_area("Type your answer here:", height=120, key="text_answer_input")
                    if st.button("Submit Answer ➜", key="submit_text", disabled=st.session_state.processing_answer):
                        if text_ans.strip():
                            process_answer(text_ans.strip())
                        else:
                            st.warning("Please type something before submitting.")

                with tab_voice:
                    audio = mic_recorder(
                        start_prompt="🎙️ Start Recording",
                        stop_prompt="⏹️ Stop Recording",
                        key="recorder"
                    )
                    if audio:
                        # Use a hash of the audio bytes as a unique ID
                        import hashlib
                        audio_id = hashlib.md5(audio["bytes"]).hexdigest()

                        # Only process if this is a NEW recording we haven't seen yet
                        if audio_id != st.session_state.last_audio_id:
                            st.session_state.last_audio_id = audio_id
                            st.audio(audio["bytes"])

                            base_path = os.path.join(tempfile.gettempdir(), f"audio_{int(time.time())}")
                            webm_path = base_path + ".webm"
                            wav_path  = base_path + ".wav"

                            with open(webm_path,"wb") as f:
                                f.write(audio["bytes"])

                            result_ff = subprocess.run(
                                ["ffmpeg","-y","-i",webm_path,"-ar","16000","-ac","1",wav_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE
                            )
                            if result_ff.returncode != 0:
                                st.error("❌ Audio conversion failed. Use the Type Answer tab instead.")
                            else:
                                with st.spinner("Transcribing..."):
                                    result = whisper_model.transcribe(wav_path)
                                    text   = result["text"].strip()
                                if not text:
                                    st.error("❌ Could not transcribe. Try speaking clearly or use text input.")
                                else:
                                    st.success(f"📝 You said: *{text}*")
                                    process_answer(text)

                            for path in [webm_path, wav_path]:
                                if os.path.exists(path):
                                    os.remove(path)


if __name__ == "__main__":
    main()