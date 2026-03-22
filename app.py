""" 🧩 مولّد أدلة الكلمات المتقاطعة العربية نسخة متقدمة — متعدد المزودين — بدون مكتبات ثقيلة """

import streamlit as st
import pandas as pd
import re
import requests
import time
import json
import hashlib
from datetime import datetime
from io import BytesIO

# ── استيرادات اختيارية ──
try:
    from huggingface_hub import InferenceClient
    HF_HUB = True
except ImportError:
    HF_HUB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False


# ══════════════════════════════════════════════
# 1. إعدادات الصفحة
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="مولّد أدلة الكلمات المتقاطعة العربية",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════
# 2. CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
*{font-family:'Tajawal',sans-serif}
.main-title{text-align:center;color:#1a5276;font-size:2.6rem;font-weight:900;direction:rtl;margin-bottom:.3rem}
.sub-title{text-align:center;color:#5d6d7e;font-size:1.05rem;direction:rtl;margin-bottom:1.5rem}
.rtl{direction:rtl;text-align:right;line-height:1.9}
.box{padding:16px;border-radius:10px;direction:rtl;text-align:right;margin:8px 0}
.box-info{background:#fef9e7;border-right:5px solid #f39c12}
.box-ok{background:#d5f5e3;border-right:5px solid #27ae60}
.box-err{background:#fadbd8;border-right:5px solid #e74c3c}
.box-clue{background:#d5f5e3;border-right:4px solid #27ae60;font-size:1.1rem}
.box-result{background:#eaf2f8;border-right:5px solid #2980b9}
.box-hist{background:#f4ecf7;border-right:4px solid #8e44ad;font-size:.95rem}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.82rem;font-weight:700;margin:0 4px}
.badge-easy{background:#d5f5e3;color:#1e8449}
.badge-med{background:#fef9e7;color:#b7950b}
.badge-hard{background:#fadbd8;color:#c0392b}
.score-bar{height:6px;border-radius:3px;margin-top:4px}
.stat-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff;padding:20px;border-radius:14px;text-align:center;margin:6px 0}
.stat-card h2{margin:0;font-size:2rem}
.stat-card p{margin:0;font-size:.9rem;opacity:.85}
.stTextArea textarea,.stTextInput input{direction:rtl;text-align:right;font-family:'Tajawal',sans-serif}
div[data-testid="stSidebar"]{direction:rtl;text-align:right}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 3. ثوابت
# ══════════════════════════════════════════════
PROVIDERS_CFG = {
    "Hugging Face 🤗": {
        "key_name": "HF_API_KEY",
        "placeholder": "hf_xxxxxxxxxxxxxxxxxxxx",
        "url": "https://huggingface.co/settings/tokens",
        "models": {
            "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
            "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
            "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
            "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
            "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
        },
    },
    "Mistral AI 🌀": {
        "key_name": "MISTRAL_API_KEY",
        "placeholder": "your-mistral-api-key",
        "url": "https://console.mistral.ai/api-keys/",
        "models": {
            "mistral-large-latest": "mistral-large-latest",
            "mistral-small-latest": "mistral-small-latest",
            "open-mistral-nemo": "open-mistral-nemo",
        },
    },
    "OpenAI-Compatible 🔌": {
        "key_name": "CUSTOM_API_KEY",
        "placeholder": "sk-xxxx or any key",
        "url": "",
        "models": {},
    },
}

CATEGORIES = [
    "دين", "تاريخ", "جغرافيا", "علوم", "أدب", "رياضة", "فن", "سياسة", "اقتصاد", "تكنولوجيا", "طب", "ثقافة عامة", "أخرى",
]

CAT_KEYWORDS = {
    "دين": ["الله", "القرآن", "الإسلام", "النبي", "محمد", "صلاة", "مسجد", "رمضان", "حج"],
    "تاريخ": ["تاريخ", "قرن", "حضارة", "معركة", "حرب", "ثورة", "خليفة", "سلطان"],
    "جغرافيا": ["مدينة", "دولة", "نهر", "جبل", "بحر", "صحراء", "عاصمة", "قارة", "جزيرة"],
    "علوم": ["ذرة", "خلية", "طاقة", "فيزياء", "كيمياء", "تجربة", "نظرية"],
    "أدب": ["شعر", "قصيدة", "رواية", "كاتب", "شاعر", "قصة", "ديوان"],
    "رياضة": ["كرة", "ملعب", "بطولة", "فريق", "لاعب", "مباراة", "كأس"],
    "فن": ["رسم", "لوحة", "موسيقى", "فيلم", "نحت", "فنان", "سينما"],
    "سياسة": ["رئيس", "حكومة", "برلمان", "انتخابات", "حزب", "دستور", "وزير"],
    "اقتصاد": ["اقتصاد", "تجارة", "بنك", "عملة", "سوق", "استثمار"],
    "تكنولوجيا": ["حاسوب", "إنترنت", "برمجة", "ذكاء اصطناعي", "تطبيق", "بيانات"],
    "طب": ["طب", "مرض", "علاج", "جراحة", "دواء", "مستشفى", "لقاح"],
}

STYLES = {
    "🎯 دقيق": {
        "temp": 0.1,
        "addon": "كن دقيقاً ومباشراً. قدم أدلة واضحة تعتمد على حقائق.",
        "desc": "أدلة واضحة ومحددة",
    },
    "⚖️ متوازن": {
        "temp": 0.4,
        "addon": "وازن بين الوضوح والتشويق في الأدلة.",
        "desc": "بين الدقة والإبداع",
    },
    "🎨 إبداعي": {
        "temp": 0.7,
        "addon": "كن إبداعياً. استخدم التلميحات والاستعارات.",
        "desc": "أدلة ذكية غير مباشرة",
    },
}

SYSTEM_PROMPT_BASE = (
    "أنت خبير في إنشاء أدلة الكلمات المتقاطعة باللغة العربية.\n"
    "القواعد:\n"
    "١- لا تذكر الكلمة المفتاحية في الدليل أبداً.\n"
    "٢- اكتب كل دليل في سطر: الدليل 1: ...\n"
    "٣- اجعل الدليل مختصراً (3–15 كلمة).\n"
    "٤- نوّع الأدلة ولا تكرر نفس الفكرة.\n"
)

USER_TMPL = (
    "أنشئ {n} أدلة كلمات متقاطعة عربية.\n\n"
    "النص:\n{text}\n\n"
    "الكلمة المفتاحية: {keyword}\n\n"
    "الفئة: {category}"
)


# ══════════════════════════════════════════════
# 4. Session State
# ══════════════════════════════════════════════
_DEFAULTS = dict(
    history=[],
    favorites=[],
    api_calls=0,
    total_clues=0,
)

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════
# 5. دوال مساعدة
# ══════════════════════════════════════════════
def secret(key, fallback=""):
    try:
        return st.secrets[key]
    except Exception:
        return fallback


def suggest_category(text: str) -> str:
    scores = {c: sum(k in text for k in kws) for c, kws in CAT_KEYWORDS.items()}
    best = max(scores, key=scores.get) if scores else "ثقافة عامة"
    return best if scores.get(best, 0) > 0 else "ثقافة عامة"


def validate_inputs(text: str, keyword: str):
    warns, infos = [], []
    if len(text.split()) < 5:
        warns.append("النص قصير جداً — يُفضّل 20 كلمة على الأقل.")
    if keyword and keyword not in text:
        warns.append(f"الكلمة «{keyword}» غير موجودة في النص؛ قد تتأثر الجودة.")
    ar = len(re.findall(r'[\u0600-\u06FF]', text))
    total = len(re.findall(r'\S', text))
    if total and ar / total < 0.5:
        warns.append("نسبة العربية منخفضة في النص.")
    infos.append(f"📊 {len(text.split())} كلمة · {len(text)} حرف")
    return warns, infos


def difficulty(clue: str) -> str:
    w = len(clue.split())
    if w <= 3:
        return "سهل"
    return "متوسط" if w <= 7 else "صعب"


def diff_badge(d: str) -> str:
    cls = {"سهل": "badge-easy", "متوسط": "badge-med", "صعب": "badge-hard"}
    return f'<span class="badge {cls.get(d,"")}">{d}</span>'


def clue_score(clue: str, keyword: str, text: str) -> int:
    s = 50
    if keyword in clue:
        s -= 30
    w = len(clue.split())
    if w < 2:
        s -= 20
    if w > 15:
        s -= 10
    shared = set(clue.split()) & set(text.split())
    s += min(len(shared) * 5, 30)
    ar = len(re.findall(r'[\u0600-\u06FF]', clue)) / max(len(clue), 1)
    if ar > 0.6:
        s += 10
    return max(0, min(100, s))


def score_color(s: int) -> str:
    if s >= 70:
        return "#27ae60"
    return "#f39c12" if s >= 40 else "#e74c3c"


def extract_clues(raw: str, n: int = 3) -> list:
    if not raw:
        return []
    pats = [
        r'(?:الدليل|دليل)\s*\d+\s*:\s*(.+?)(?=(?:الدليل|دليل)\s*\d+|$)',
        r'CLUE\s*\d+\s*:\s*(.+?)(?=CLUE\s*\d+|$)',
        r'\d+\s*[\.\-\)]\s*(.+?)(?=\d+\s*[\.\-\)]|$)',
        r'[•\-\*]\s*(.+?)(?=[•\-\*]|$)',
    ]
    clues = []
    for p in pats:
        ms = re.findall(p, raw, re.S | re.I)
        if ms:
            clues = [m.strip() for m in ms if m.strip() and len(m.strip()) > 3]
            break
    if not clues:
        clues = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 3]
    # تنظيف + حذف التكرار
    seen, out = set(), []
    for c in clues:
        c = re.sub(r"^\s*[\-\*•\d\.\)]+\s*", "", c).strip()
        c = re.sub(r"\s+", " ", c)
        key = c[:30]
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out[:n]


def build_messages(text, keyword, category, num, style_addon=""):
    sys = SYSTEM_PROMPT_BASE + (f"\n{style_addon}" if style_addon else "")
    usr = USER_TMPL.format(n=num, text=text, keyword=keyword, category=category)
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def push_history(rec: dict):
    rec["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec["id"] = hashlib.md5(f"{rec['keyword']}{rec['ts']}".encode()).hexdigest()[:8]
    st.session_state.history.insert(0, rec)
    st.session_state.total_clues += len(rec.get("clues", []))
    st.session_state.api_calls += 1
    st.session_state.history = st.session_state.history[:200]


# ══════════════════════════════════════════════
# 6. دوال API
# ══════════════════════════════════════════════
def _rest_chat(url, key, model, msgs, temp, tokens):
    """طلب REST عام لنقاط نهاية OpenAI-compatible."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": msgs,
        "temperature": max(temp, 0.01),
        "max_tokens": tokens,
        "top_p": 0.95,
    }
    for i in range(3):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                return {"ok": True, "text": r.json()["choices"][0]["message"]["content"]}
            if r.status_code in (503, 529):
                time.sleep(15 * (i + 1))
                continue
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
            if r.status_code == 401:
                return {"ok": False, "err": "مفتاح API غير صالح."}
            return {"ok": False, "err": f"({r.status_code}) {r.text[:200]}"}
        except requests.exceptions.Timeout:
            if i < 2:
                time.sleep(5)
                continue
            return {"ok": False, "err": "انتهت مهلة الاتصال."}
        except Exception as e:
            return {"ok": False, "err": str(e)[:250]}
    return {"ok": False, "err": "فشلت جميع المحاولات."}


def api_hf(msgs, key, model, temp, tokens):
    """Hugging Face — InferenceClient أو REST."""
    if HF_HUB:
        client = InferenceClient(model=model, token=key, timeout=120)
        for i in range(3):
            try:
                r = client.chat_completion(
                    messages=msgs,
                    max_tokens=tokens,
                    temperature=max(temp, 0.01),
                    top_p=0.95,
                )
                if r and r.choices:
                    return {"ok": True, "text": r.choices[0].message.content}
                return {"ok": False, "err": "رد فارغ من النموذج."}
            except Exception as e:
                err = str(e)
                if any(x in err for x in ["503", "loading"]):
                    time.sleep(15 * (i + 1))
                    continue
                if "429" in err:
                    time.sleep(10 * (i + 1))
                    continue
                if "401" in err:
                    return {"ok": False, "err": "مفتاح API غير صالح."}
                return {"ok": False, "err": err[:250]}
        return {"ok": False, "err": "النموذج لا يزال قيد التحميل."}
    # ── fallback REST ──
    return _rest_chat(
        f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions",
        key,
        model,
        msgs,
        temp,
        tokens,
    )


def api_mistral(msgs, key, model, temp, tokens):
    """Mistral AI — REST فقط (بدون مكتبة)."""
    return _rest_chat(
        "https://api.mistral.ai/v1/chat/completions",
        key,
        model,
        msgs,
        temp,
        tokens,
    )


def api_custom(msgs, key, model, base_url, temp, tokens):
    """أي خادم متوافق مع OpenAI."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    return _rest_chat(url, key, model, msgs, temp, tokens)


# ══════════════════════════════════════════════
# 7. دالة التوليد الرئيسية
# ══════════════════════════════════════════════
def generate(
    text,
    keyword,
    category,
    provider,
    api_key,
    model_id,
    temp,
    max_tok,
    num_clues,
    style_addon="",
    base_url="",
):
    msgs = build_messages(text, keyword, category, num_clues, style_addon)
    t0 = time.time()
    if "Hugging" in provider:
        res = api_hf(msgs, api_key, model_id, temp, max_tok)
    elif "Mistral" in provider:
        res = api_mistral(msgs, api_key, model_id, temp, max_tok)
    else:
        res = api_custom(msgs, api_key, model_id, base_url, temp, max_tok)
    elapsed = round(time.time() - t0, 2)
    if not res["ok"]:
        return {
            "success": False,
            "error": res["err"],
            "clues": [],
            "details": [],
            "raw": "",
            "time": elapsed,
        }
    raw = res["text"]
    clues = extract_clues(raw, num_clues)
    details = [
        {
            "text": c,
            "difficulty": difficulty(c),
            "score": clue_score(c, keyword, text),
        }
        for c in clues
    ]
    return {
        "success": True,
        "clues": clues,
        "details": details,
        "raw": raw,
        "time": elapsed,
    }


# ══════════════════════════════════════════════
# 8. الشريط الجانبي
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")
    st.markdown("---")

    # ── المزوّد ──
    provider = st.radio("مزوّد الخدمة", list(PROVIDERS_CFG.keys()), horizontal=True)
    cfg = PROVIDERS_CFG[provider]

    # ── المفتاح (أولوية secrets) ──
    saved_key = secret(cfg["key_name"])
    if saved_key:
        api_key = saved_key
        st.markdown('<div class="box box-ok">🔑 المفتاح مُحمَّل من secrets</div>', unsafe_allow_html=True)
    else:
        api_key = st.text_input(
            "🔑 مفتاح API",
            type="password",
            placeholder=cfg["placeholder"],
            help=f"[احصل على مفتاح]({cfg['url']})" if cfg["url"] else "",
        )
    api_ok = bool(api_key and len(api_key) > 5)

    # ── النموذج ──
    if cfg["models"]:
        model_label = st.selectbox("🤖 النموذج", list(cfg["models"].keys()))
        model_id = cfg["models"][model_label]
    else:
        model_id = st.text_input("🤖 اسم النموذج", value="gpt-3.5-turbo")

    # ── رابط مخصص ──
    base_url = ""
    if "Compatible" in provider:
        base_url = st.text_input(
            "🔗 رابط الخادم",
            value="https://api.openai.com",
            help="مثال: http://localhost:11434"
        )

    st.markdown("---")

    # ── أسلوب التوليد ──
    style_name = st.radio("🎨 أسلوب التوليد", list(STYLES.keys()), horizontal=True)
    style = STYLES[style_name]
    st.caption(style["desc"])

    # ── معلمات ──
    temperature = st.slider("🌡️ الحرارة", 0.0, 1.0, style["temp"], 0.05)
    num_clues = st.slider("🔢 عدد الأدلة", 1, 10, 3)
    max_tokens = st.slider("📏 الحد الأقصى للرموز", 64, 1024, 300, 32)

    st.markdown("---")

    # ── إحصائيات سريعة ──
    c1, c2 = st.columns(2)
    c1.metric("🔄 طلبات", st.session_state.api_calls)
    c2.metric("💡 أدلة", st.session_state.total_clues)

    st.markdown(f"""
    <div class="box box-info" style="font-size:.85rem">
        <b>المزوّد:</b> {provider}<br>
        <b>النموذج:</b> {model_id}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 9. العنوان
# ══════════════════════════════════════════════
st.markdown('<h1 class="main-title">🧩 مولّد أدلة الكلمات المتقاطعة العربية</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">أداة ذكية متعددة المزودين — اقتراح تلقائي للفئة — تقييم جودة الأدلة — سجل كامل</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 10. عرض الأدلة (مكوّن مشترك)
# ══════════════════════════════════════════════
def render_clues(result, keyword, text):
    """عرض الأدلة المُولَّدة بتنسيق جميل."""
    if not result["success"]:
        st.markdown(f'<div class="box box-err">❌ {result["error"]}</div>', unsafe_allow_html=True)
        return
    st.markdown(f"⏱️ الزمن: **{result['time']}** ثانية")
    st.markdown("---")
    for i, d in enumerate(result["details"], 1):
        sc = d["score"]
        col = score_color(sc)
        st.markdown(f"""
        <div class="box box-clue">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                <span><strong>الدليل {i}:</strong> {d['text']}</span>
                <span>{diff_badge(d['difficulty'])}</span>
            </div>
            <div style="margin-top:6px;font-size:.82rem;color:#666">
                جودة: {sc}/100
                <div class="score-bar" style="width:{sc}%;background:{col}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # أزرار إضافية
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("⭐ حفظ في المفضلة", key=f"fav_{time.time()}"):
            st.session_state.favorites.append({
                "keyword": keyword,
                "clues": result["clues"],
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.success("✅ تمت الإضافة إلى المفضلة")
    with col_b:
        st.download_button(
            "📋 نسخ كنص",
            data="\n".join(f"{i+1}. {c}" for i, c in enumerate(result["clues"])),
            file_name=f"clues_{keyword}.txt",
            mime="text/plain",
            key=f"dl_{time.time()}",
        )
    with col_c:
        with st.expander("🔍 الاستجابة الكاملة"):
            st.code(result["raw"], language=None)


# ══════════════════════════════════════════════
# 11. التبويبات
# ══════════════════════════════════════════════
tab_manual, tab_csv, tab_compare, tab_history = st.tabs(
    ["📝 إدخال يدوي", "📁 ملف CSV", "🔄 مقارنة نماذج", "📜 السجل والتحليلات"]
)


# ──────────────────────────────────────────────
# تبويب 1: إدخال يدوي
# ──────────────────────────────────────────────
with tab_manual:
    st.markdown('<p class="rtl"><strong>📄 النص المرجعي</strong></p>', unsafe_allow_html=True)
    input_text = st.text_area(
        "النص",
        height=180,
        placeholder="الصق النص العربي هنا…",
        label_visibility="collapsed",
        key="m_text"
    )

    mc1, mc2, mc3 = st.columns([2, 2, 1])
    with mc1:
        input_kw = st.text_input(
            "🔑 الكلمة المفتاحية",
            placeholder="مثال: القاهرة",
            label_visibility="collapsed",
            key="m_kw"
        )
    with mc2:
        suggested = suggest_category(input_text) if input_text else CATEGORIES[0]
        idx = CATEGORIES.index(suggested) if suggested in CATEGORIES else 0
        input_cat = st.selectbox("📂 الفئة", CATEGORIES, index=idx, key="m_cat")
    with mc3:
        if input_text:
            st.markdown(f'<span class="badge badge-easy">اقتراح: {suggested}</span>', unsafe_allow_html=True)

    # تحقق من المدخلات
    if input_text and input_kw:
        warns, infos = validate_inputs(input_text, input_kw)
        for w in warns:
            st.warning(f"⚠️ {w}")
        for inf in infos:
            st.caption(inf)

    gen_btn = st.button(
        "🚀 توليد الأدلة",
        use_container_width=True,
        type="primary",
        disabled=not api_ok,
        key="gen1"
    )

    if not api_ok:
        st.info("🔑 أدخل مفتاح API في الشريط الجانبي أو أضفه في ملف secrets")

    if gen_btn and input_text.strip() and input_kw.strip():
        with st.spinner(f"⏳ التوليد عبر {provider}…"):
            res = generate(
                input_text,
                input_kw,
                input_cat,
                provider,
                api_key,
                model_id,
                temperature,
                max_tokens,
                num_clues,
                style["addon"],
                base_url,
            )
            render_clues(res, input_kw, input_text)
            if res["success"]:
                push_history({
                    "keyword": input_kw,
                    "category": input_cat,
                    "clues": res["clues"],
                    "model": model_id,
                    "provider": provider,
                    "style": style_name,
                    "time": res["time"],
                })


# ──────────────────────────────────────────────
# تبويب 2: ملف CSV
# ──────────────────────────────────────────────
with tab_csv:
    st.markdown('<p class="rtl"><strong>📁 معالجة دفعية عبر ملف CSV</strong></p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="box box-info">
        الأعمدة المطلوبة: <code>text</code> · <code>keyword</code> · <code>category</code>
    </div>
    """, unsafe_allow_html=True)

    sample = pd.DataFrame({
        "text": ["القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها وتقع على ضفاف نهر النيل."],
        "keyword": ["القاهرة"],
        "category": ["جغرافيا"],
    })
    st.download_button(
        "📥 تحميل نموذج CSV",
        sample.to_csv(index=False).encode("utf-8-sig"),
        "sample.csv",
        "text/csv",
    )

    upl = st.file_uploader("ارفع ملف CSV", type=["csv"], label_visibility="collapsed")
    if upl:
        try:
            df = pd.read_csv(upl)
            need = {"text", "keyword", "category"}
            if not need.issubset(df.columns):
                st.error(f"❌ أعمدة مفقودة: {need - set(df.columns)}")
            else:
                st.info(f"📊 {len(df)} صف")
                with st.expander("👁️ معاينة"):
                    st.dataframe(df.head(10), use_container_width=True)

                delay = st.slider("⏱️ تأخير بين الطلبات (ثانية)", 1, 20, 3, key="csv_delay")
                if st.button(
                    "🚀 بدء المعالجة",
                    use_container_width=True,
                    type="primary",
                    disabled=not api_ok,
                    key="csv_go"
                ):
                    rows_out = []
                    bar = st.progress(0)
                    status = st.empty()
                    for idx, row in df.iterrows():
                        bar.progress((idx + 1) / len(df))
                        status.markdown(f"⏳ **{idx+1}/{len(df)}** — {row['keyword']}")
                        res = generate(
                            str(row["text"]),
                            str(row["keyword"]),
                            str(row["category"]),
                            provider,
                            api_key,
                            model_id,
                            temperature,
                            max_tokens,
                            num_clues,
                            style["addon"],
                            base_url,
                        )
                        if res["success"]:
                            clues_txt = "\n".join(res["clues"])
                            rows_out.append({**row, "generated_clues": clues_txt, "status": "✅"})
                            push_history({
                                "keyword": row["keyword"],
                                "category": row["category"],
                                "clues": res["clues"],
                                "model": model_id,
                                "provider": provider,
                                "style": style_name,
                                "time": res["time"],
                            })
                        else:
                            rows_out.append({**row, "generated_clues": "", "status": f"❌ {res['error']}"})
                        if idx < len(df) - 1:
                            time.sleep(delay)

                    bar.progress(1.0)
                    status.success("✅ تمت المعالجة!")
                    out_df = pd.DataFrame(rows_out)
                    ok_n = sum(r["status"] == "✅" for r in rows_out)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("الإجمالي", len(rows_out))
                    c2.metric("نجاح ✅", ok_n)
                    c3.metric("أخطاء ❌", len(rows_out) - ok_n)
                    st.dataframe(out_df, use_container_width=True)

                    # تحميل بتنسيقات متعددة
                    dl1, dl2, dl3 = st.columns(3)
                    with dl1:
                        st.download_button(
                            "📥 CSV",
                            out_df.to_csv(index=False).encode("utf-8-sig"),
                            "results.csv",
                            "text/csv",
                            use_container_width=True,
                        )
                    with dl2:
                        st.download_button(
                            "📥 JSON",
                            out_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                            "results.json",
                            "application/json",
                            use_container_width=True,
                        )
                    with dl3:
                        try:
                            buf = BytesIO()
                            out_df.to_excel(buf, index=False, engine="openpyxl")
                            st.download_button(
                                "📥 Excel",
                                buf.getvalue(),
                                "results.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )
                        except Exception:
                            st.caption("📦 ثبّت openpyxl لتصدير Excel")
        except Exception as e:
            st.error(f"❌ خطأ: {e}")


# ──────────────────────────────────────────────
# تبويب 3: مقارنة نماذج
# ──────────────────────────────────────────────
with tab_compare:
    st.markdown('<p class="rtl"><strong>🔄 قارن نتائج نموذجين جنباً لجنب</strong></p>', unsafe_allow_html=True)
    cmp_text = st.text_area("النص", height=120, key="cmp_t", placeholder="النص المرجعي…")
    cc1, cc2 = st.columns(2)
    with cc1:
        cmp_kw = st.text_input("الكلمة المفتاحية", key="cmp_kw")
    with cc2:
        cmp_cat = st.selectbox("الفئة", CATEGORIES, key="cmp_cat")

    st.markdown("**اختر نموذجين للمقارنة:**")
    all_models = {}
    for p, c in PROVIDERS_CFG.items():
        for label, mid in c["models"].items():
            all_models[f"{p.split()[0]} / {label}"] = (p, mid)

    if all_models:
        model_keys = list(all_models.keys())
        cx1, cx2 = st.columns(2)
        with cx1:
            m1_label = st.selectbox("النموذج ①", model_keys, index=0, key="cmp_m1")
        with cx2:
            m2_label = st.selectbox("النموذج ②", model_keys, index=min(1, len(model_keys)-1), key="cmp_m2")

        if st.button("⚡ قارن", use_container_width=True, type="primary", disabled=not api_ok, key="cmp_go"):
            if cmp_text.strip() and cmp_kw.strip():
                col_l, col_r = st.columns(2)
                for col, m_label in [(col_l, m1_label), (col_r, m2_label)]:
                    prov, mid = all_models[m_label]
                    with col:
                        st.markdown(f"**{m_label}**")
                        with st.spinner("⏳"):
                            r = generate(
                                cmp_text,
                                cmp_kw,
                                cmp_cat,
                                prov,
                                api_key,
                                mid,
                                temperature,
                                max_tokens,
                                num_clues,
                                style["addon"],
                                base_url,
                            )
                            render_clues(r, cmp_kw, cmp_text)
            else:
                st.warning("أدخل النص والكلمة المفتاحية أولاً.")
    else:
        st.info("لا توجد نماذج للمقارنة.")


# ──────────────────────────────────────────────
# تبويب 4: السجل والتحليلات
# ──────────────────────────────────────────────
with tab_history:
    h = st.session_state.history
    f = st.session_state.favorites
    sub1, sub2 = st.tabs(["📜 السجل", "⭐ المفضلة"])

    # ── السجل ──
    with sub1:
        if not h:
            st.info("لا يوجد سجل بعد. ابدأ بتوليد بعض الأدلة!")
        else:
            filter_kw = st.text_input("🔍 بحث بالكلمة المفتاحية", key="hist_filter")
            filtered = [r for r in h if filter_kw.lower() in r.get("keyword", "").lower()] if filter_kw else h
            st.caption(f"عرض {len(filtered)} من {len(h)} سجل")
            for rec in filtered[:50]:
                clues_str = " | ".join(rec.get("clues", [])[:3])
                st.markdown(f"""
                <div class="box box-hist">
                    <b>{rec.get('keyword','')}</b> · {rec.get('category','')} · <span style="font-size:.8rem;color:#888">{rec.get('ts','')}</span><br>
                    <span style="font-size:.9rem">{clues_str}</span><br>
                    <span style="font-size:.78rem;color:#aaa">
                        {rec.get('provider','')} · {rec.get('model','').split('/')[-1]} · {rec.get('time','')}s
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── تحليلات ──
            st.markdown("---")
            st.markdown("### 📊 تحليلات")
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f'<div class="stat-card"><h2>{len(h)}</h2><p>طلب</p></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="stat-card"><h2>{st.session_state.total_clues}</h2><p>دليل</p></div>', unsafe_allow_html=True)
            with sc3:
                avg_t = round(sum(r.get("time", 0) for r in h) / max(len(h), 1), 1)
                st.markdown(f'<div class="stat-card"><h2>{avg_t}s</h2><p>متوسط الزمن</p></div>', unsafe_allow_html=True)
            with sc4:
                cats = pd.Series([r.get("category", "") for r in h]).value_counts()
                top_cat = cats.index[0] if len(cats) > 0 else "—"
                st.markdown(f'<div class="stat-card"><h2>{top_cat}</h2><p>أكثر فئة</p></div>', unsafe_allow_html=True)

            if len(h) >= 3:
                # رسم بياني — توزيع الفئات
                cat_counts = pd.DataFrame(
                    pd.Series([r.get("category", "") for r in h]).value_counts()
                ).reset_index()
                cat_counts.columns = ["الفئة", "العدد"]
                if PLOTLY:
                    fig = px.bar(cat_counts, x="الفئة", y="العدد", color="الفئة", title="توزيع الفئات")
                    fig.update_layout(showlegend=False, font=dict(family="Tajawal"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(cat_counts.set_index("الفئة"))

                # رسم بياني — الزمن عبر الطلبات
                times = [r.get("time", 0) for r in reversed(h)]
                if PLOTLY:
                    fig2 = px.line(y=times, title="زمن الاستجابة عبر الطلبات", labels={"x": "الطلب", "y": "ثانية"})
                    fig2.update_layout(font=dict(family="Tajawal"))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.line_chart(times)

            # تصدير السجل
            st.markdown("---")
            if st.button("🗑️ مسح السجل", key="clear_hist"):
                st.session_state.history = []
                st.session_state.api_calls = 0
                st.session_state.total_clues = 0
                st.rerun()
            if h:
                hist_df = pd.DataFrame(h)
                st.download_button(
                    "📥 تصدير السجل CSV",
                    hist_df.to_csv(index=False).encode("utf-8-sig"),
                    "history.csv",
                    "text/csv",
                    use_container_width=True,
                )

    # ── المفضلة ──
    with sub2:
        if not f:
            st.info("⭐ لم تحفظ أي أدلة في المفضلة بعد.")
        else:
            for fav in f:
                clues_str = " | ".join(fav.get("clues", []))
                st.markdown(f"""
                <div class="box box-clue">
                    <b>⭐ {fav.get('keyword','')}</b>
                    <span style="font-size:.8rem;color:#888"> — {fav.get('ts','')}</span><br>
                    {clues_str}
                </div>
                """, unsafe_allow_html=True)
            if st.button("🗑️ مسح المفضلة", key="clear_fav"):
                st.session_state.favorites = []
                st.rerun()


# ══════════════════════════════════════════════
# 12. التذييل
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#95a5a6;padding:18px">
    🧩 مولّد أدلة الكلمات المتقاطعة العربية — نسخة متقدمة<br>
    <span style="font-size:.82rem">Streamlit · HuggingFace · Mistral · OpenAI-Compatible</span>
</div>
""", unsafe_allow_html=True)
