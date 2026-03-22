import streamlit as st
import pandas as pd
import re
import requests
import time
import json
import io
from huggingface_hub import InferenceClient

# محاولة استيراد mistralai (إذا كان مثبتاً)
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    st.warning("⚠️ مكتبة mistralai غير مثبتة. لن تعمل خيارات Mistral.")

# =============================================
# إعدادات الصفحة
# =============================================
st.set_page_config(
    page_title="مولّد أدلة الكلمات المتقاطعة العربية",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CSS للعربية
# =============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
* {
    font-family: 'Tajawal', sans-serif;
}
.main-title {
    text-align: center;
    color: #1a5276;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    direction: rtl;
}
.sub-title {
    text-align: center;
    color: #5d6d7e;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    direction: rtl;
}
.rtl-text {
    direction: rtl;
    text-align: right;
    font-size: 1.05rem;
    line-height: 1.8;
}
.result-box {
    background: #eaf2f8;
    border-right: 5px solid #2980b9;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
    direction: rtl;
    text-align: right;
}
.clue-item {
    background: #d5f5e3;
    border-right: 4px solid #27ae60;
    padding: 12px 18px;
    border-radius: 6px;
    margin: 8px 0;
    direction: rtl;
    text-align: right;
    font-size: 1.1rem;
}
.error-box {
    background: #fadbd8;
    border-right: 5px solid #e74c3c;
    padding: 15px;
    border-radius: 8px;
    direction: rtl;
    text-align: right;
}
.info-box {
    background: #fef9e7;
    border-right: 5px solid #f39c12;
    padding: 15px;
    border-radius: 8px;
    direction: rtl;
    text-align: right;
    margin: 10px 0;
}
.success-box {
    background: #d5f5e3;
    border-right: 5px solid #27ae60;
    padding: 15px;
    border-radius: 8px;
    direction: rtl;
    text-align: right;
    margin: 10px 0;
}
.stTextArea textarea, .stTextInput input {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif;
    font-size: 1rem;
}
div[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# الثوابت
# =============================================
# نماذج Hugging Face (conversational)
HF_MODELS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3"
}
# نموذج Mistral عبر API الخاص بهم (إذا أردت استخدامه بدلاً من Hugging Face)
MISTRAL_MODEL = "mistral-large-latest"  # أو أي نموذج متاح لديك

SYSTEM_PROMPT = (
    "أنت مساعد خبير في إنشاء أدلة الكلمات المتقاطعة باللغة العربية. "
    "سأعطيك نصًا وكلمة مفتاحية وفئة، وعليك توليد أدلة مناسبة للكلمة المفتاحية بناءً على النص والفئة. "
    "قدم الأدلة في سطور منفصلة، كل دليل في سطر واحد بالصيغة: "
    "الدليل 1: ...\nالدليل 2: ...\nالدليل 3: ...\n"
    "تأكد من أن الأدلة مختصرة وواضحة، وتتعلق مباشرة بالكلمة المفتاحية."
)
USER_INSTRUCTION = (
    "أنشئ أدلة كلمات متقاطعة عربية للكلمة المفتاحية المحددة، "
    "باستخدام النص المقدم مع التركيز على الفئة المشار إليها."
)

CATEGORIES = [
    "دين", "تاريخ", "جغرافيا", "علوم", "أدب", "رياضة", "فن", "سياسة", "اقتصاد", "تكنولوجيا", "طب", "ثقافة عامة", "أخرى"
]

# =============================================
# الدوال المساعدة
# =============================================
def format_messages(text: str, keyword: str, category: str) -> list:
    """تنسيق المدخلات إلى قائمة الرسائل المطلوبة للنماذج."""
    user_content = (
        f"{USER_INSTRUCTION}\n\n"
        f"النص: {text}\n\n"
        f"الكلمة المفتاحية: {keyword}\n\n"
        f"الفئة: {category}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

def extract_clues(generated_text: str, max_clues: int = 3) -> list:
    """استخراج الأدلة من النص المُولّد مع معالجة الأنماط المختلفة."""
    if not generated_text:
        return []
    
    # أولاً: محاولة استخراج بنمط "الدليل X: ..."
    patterns = [
        r'(?:الدليل\s*)(\d+)\s*:\s*(.+?)(?=(?:الدليل\s*\d+\s*:)|$)',   # الدليل 1: ...
        r'(?:CLUE\s*)(\d+)\s*:\s*(.+?)(?=(?:CLUE\s*\d+\s*:)|$)',       # CLUE 1: ...
        r'(?:Clue\s*)(\d+)\s*:\s*(.+?)(?=(?:Clue\s*\d+\s*:)|$)',       # Clue 1: ...
        r'(\d+)\s*[\.\-\)]\s*(.+?)(?=\d+\s*[\.\-\)]|$)',               # 1. ... أو 1- ...
    ]
    
    clues = []
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    # النمط الأول يعطي (الرقم, النص)
                    clue_text = match[1].strip()
                else:
                    clue_text = match.strip()
                clues.append(clue_text)
            break
    
    # إذا لم نجد أدلة بالأنماط، نقسم حسب الأسطر
    if not clues:
        lines = [l.strip() for l in generated_text.split('\n') if l.strip()]
        clues = lines
    
    # تنظيف الأدلة من الأقواس المربعة والشرطات الزائدة
    cleaned = []
    for clue in clues:
        # إزالة الأقواس المربعة ومحتواها
        clue = re.sub(r'\[[^\]]*\]', '', clue)
        # إزالة الشرطات الزائدة في البداية والنهاية
        clue = re.sub(r'^-+|-+$', '', clue)
        # استبدال الشرطات المتعددة بفواصل
        clue = re.sub(r'\s*-\s*', ' - ', clue)
        # إزالة المسافات الزائدة
        clue = re.sub(r'\s+', ' ', clue).strip()
        if clue:
            cleaned.append(clue)
    
    # إذا كان عدد الأدلة أقل من المطلوب، قد نضيف بعض الأسطر الإضافية
    return cleaned[:max_clues] if cleaned else clues[:max_clues]

# =============================================
# دوال الاتصال بـ Hugging Face API
# =============================================
def call_hf_api(
    messages: list,
    api_key: str,
    model_name: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    max_retries: int = 3
) -> dict:
    """استدعاء Hugging Face Inference API باستخدام chat_completion."""
    client = InferenceClient(
        model=model_name,
        token=api_key,
        timeout=120
    )
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95
            )
            if response and response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                return {"success": True, "generated_text": generated_text}
            else:
                return {"success": False, "error": "❌ لم يتم الحصول على رد من النموذج."}
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "loading" in error_msg.lower():
                if attempt < max_retries - 1:
                    st.warning(f"⏳ النموذج قيد التحميل، إعادة المحاولة ({attempt+1}/{max_retries})...")
                    time.sleep(15)
                    continue
                else:
                    return {"success": False, "error": "❌ النموذج لا يزال قيد التحميل، حاول لاحقاً."}
            elif "401" in error_msg or "Unauthorized" in error_msg:
                return {"success": False, "error": "❌ مفتاح API غير صالح أو النموذج يتطلب قبول الشروط."}
            elif "429" in error_msg:
                return {"success": False, "error": "❌ تم تجاوز حد الطلبات. حاول بعد دقيقة."}
            else:
                return {"success": False, "error": f"❌ خطأ: {error_msg}"}
    return {"success": False, "error": "❌ فشلت جميع المحاولات."}

# =============================================
# دوال الاتصال بـ Mistral API
# =============================================
def call_mistral_api(
    messages: list,
    api_key: str,
    model_name: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    max_retries: int = 3
) -> dict:
    """استدعاء Mistral API."""
    if not MISTRAL_AVAILABLE:
        return {"success": False, "error": "❌ مكتبة mistralai غير مثبتة."}
    client = Mistral(api_key=api_key)
    for attempt in range(max_retries):
        try:
            # تحويل الرسائل إلى تنسيق Mistral (نفس التنسيق)
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=0.95
            )
            if response and response.choices:
                generated_text = response.choices[0].message.content
                return {"success": True, "generated_text": generated_text}
            else:
                return {"success": False, "error": "❌ لم يتم الحصول على رد من Mistral."}
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                return {"success": False, "error": "❌ تم تجاوز حد الطلبات. حاول بعد دقيقة."}
            elif "401" in error_msg or "Unauthorized" in error_msg:
                return {"success": False, "error": "❌ مفتاح Mistral API غير صالح."}
            else:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ خطأ في Mistral API، إعادة المحاولة ({attempt+1}/{max_retries})...")
                    time.sleep(3)
                    continue
                else:
                    return {"success": False, "error": f"❌ خطأ من Mistral: {error_msg}"}
    return {"success": False, "error": "❌ فشلت جميع المحاولات مع Mistral."}

# =============================================
# الدالة الرئيسية للتوليد
# =============================================
def generate_clues(
    text: str,
    keyword: str,
    category: str,
    provider: str,
    api_key: str,
    model_name: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    num_clues: int = 3
) -> dict:
    """توليد الأدلة باستخدام المزود المختار."""
    messages = format_messages(text, keyword, category)
    
    if provider == "Hugging Face":
        api_result = call_hf_api(messages, api_key, model_name, temperature, max_new_tokens)
    elif provider == "Mistral":
        api_result = call_mistral_api(messages, api_key, model_name, temperature, max_new_tokens)
    else:
        return {"success": False, "error": "مزود غير معروف.", "clues": [], "raw_response": ""}
    
    if not api_result["success"]:
        return {
            "success": False,
            "error": api_result["error"],
            "clues": [],
            "raw_response": ""
        }
    
    raw = api_result["generated_text"]
    clues = extract_clues(raw, max_clues=num_clues)
    
    return {
        "success": True,
        "clues": clues,
        "raw_response": raw,
        "full_response": raw
    }

# =============================================
# الشريط الجانبي
# =============================================
with st.sidebar:
    st.markdown('<h2 style="text-align:right;color:#1a5276;">⚙️ الإعدادات</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    # اختيار المزود
    provider = st.radio(
        "اختر مزود الخدمة",
        options=["Hugging Face", "Mistral"],
        index=0,
        help="Hugging Face يتطلب مفتاح HF_xxxx، Mistral يتطلب مفتاح Mistral API."
    )
    
    # مفتاح API المناسب
    if provider == "Hugging Face":
        # حاول قراءة المفتاح من secrets أو من إدخال المستخدم
        default_hf_key = st.secrets.get("HF_API_KEY", "") if hasattr(st, 'secrets') else ""
        api_key = st.text_input(
            "مفتاح Hugging Face API",
            type="password",
            value=default_hf_key,
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
            help="احصل عليه من: https://huggingface.co/settings/tokens"
        )
        # اختيار النموذج
        model_choice = st.selectbox(
            "اختر النموذج",
            options=list(HF_MODELS.keys()),
            index=0
        )
        model_name = HF_MODELS[model_choice]
    else:  # Mistral
        default_mistral_key = st.secrets.get("MISTRAL_API_KEY", "") if hasattr(st, 'secrets') else ""
        api_key = st.text_input(
            "مفتاح Mistral API",
            type="password",
            value=default_mistral_key,
            placeholder="your-mistral-api-key",
            help="احصل عليه من: https://console.mistral.ai/api-keys/"
        )
        # نموذج Mistral (يمكن تخصيصه)
        model_name = st.text_input(
            "اسم النموذج (Mistral)",
            value="mistral-large-latest",
            help="مثال: mistral-large-latest, open-mistral-7b, etc."
        )
    
    st.markdown("---")
    
    # درجة الحرارة
    st.markdown('<p class="rtl-text"><strong>🌡️ درجة الحرارة</strong></p>', unsafe_allow_html=True)
    temperature = st.slider(
        "Temperature", 0.0, 1.0, 0.1, 0.05,
        label_visibility="collapsed",
        help="أقل = أكثر دقة، أعلى = أكثر إبداعاً"
    )
    
    # عدد الأدلة
    st.markdown('<p class="rtl-text"><strong>🔢 عدد الأدلة</strong></p>', unsafe_allow_html=True)
    num_clues = st.slider("عدد الأدلة", 1, 5, 3, label_visibility="collapsed")
    
    # الحد الأقصى للرموز
    st.markdown('<p class="rtl-text"><strong>📏 الحد الأقصى للرموز</strong></p>', unsafe_allow_html=True)
    max_tokens = st.slider("Max tokens", 64, 512, 256, 32, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
        <p><strong>📦 المزود:</strong> {provider}</p>
        <p><strong>🤖 النموذج:</strong> {model_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # التحقق من صحة المفتاح (بسيط)
    api_valid = bool(api_key and len(api_key) > 5)

# =============================================
# العنوان الرئيسي
# =============================================
st.markdown('<h1 class="main-title">🧩 مولّد أدلة الكلمات المتقاطعة العربية</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">أداة ذكية لإنشاء أدلة كلمات متقاطعة عربية عبر API — مع دعم Hugging Face و Mistral</p>', unsafe_allow_html=True)
st.markdown("---")

# =============================================
# التبويبات
# =============================================
tab1, tab2 = st.tabs(["📝 إدخال يدوي", "📁 رفع ملف CSV"])

# =============================================
# تبويب 1: إدخال يدوي
# =============================================
with tab1:
    st.markdown('<h3 class="rtl-text">📝 أدخل البيانات لتوليد الأدلة</h3>', unsafe_allow_html=True)
    
    st.markdown('<p class="rtl-text"><strong>📄 النص المرجعي:</strong></p>', unsafe_allow_html=True)
    input_text = st.text_area("النص", height=200, placeholder="أدخل النص العربي هنا...", label_visibility="collapsed", key="manual_text")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="rtl-text"><strong>🔑 الكلمة المفتاحية:</strong></p>', unsafe_allow_html=True)
        input_keyword = st.text_input("الكلمة المفتاحية", placeholder="أدخل الكلمة المفتاحية...", label_visibility="collapsed", key="manual_keyword")
    with col2:
        st.markdown('<p class="rtl-text"><strong>📂 الفئة:</strong></p>', unsafe_allow_html=True)
        input_category = st.selectbox("الفئة", options=CATEGORIES, label_visibility="collapsed", key="manual_category")
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 توليد الأدلة", use_container_width=True, type="primary", disabled=not api_valid, key="gen_manual")
    
    if not api_valid:
        st.markdown("""
        <div class="info-box">
            <p>🔑 أدخل مفتاح API المناسب للمزود المختار في الشريط الجانبي.</p>
        </div>
        """, unsafe_allow_html=True)
    
    if generate_btn:
        if not input_text.strip():
            st.warning("⚠️ يرجى إدخال النص المرجعي")
        elif not input_keyword.strip():
            st.warning("⚠️ يرجى إدخال الكلمة المفتاحية")
        else:
            with st.spinner(f"⏳ جارٍ توليد الأدلة عبر {provider}..."):
                result = generate_clues(
                    text=input_text,
                    keyword=input_keyword,
                    category=input_category,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    num_clues=num_clues
                )
                
                if result["success"]:
                    st.markdown("---")
                    st.markdown('<h3 class="rtl-text">✨ الأدلة المُولَّدة:</h3>', unsafe_allow_html=True)
                    if result["clues"]:
                        for i, clue in enumerate(result["clues"], 1):
                            st.markdown(
                                f'<div class="clue-item"><strong>الدليل {i}:</strong> {clue}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            f'<div class="result-box">{result["raw_response"]}</div>',
                            unsafe_allow_html=True
                        )
                    with st.expander("🔍 عرض الاستجابة الكاملة"):
                        st.code(result["raw_response"], language=None)
                else:
                    st.markdown(
                        f'<div class="error-box">{result["error"]}</div>',
                        unsafe_allow_html=True
                    )

# =============================================
# تبويب 2: رفع CSV (نفس الكود السابق مع إضافة المعلمات الجديدة)
# =============================================
with tab2:
    st.markdown('<h3 class="rtl-text">📁 معالجة دفعية عبر ملف CSV</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>📋 تنسيق الملف المطلوب:</strong></p>
        <ul style="direction:rtl; text-align:right;">
            <li><code>text</code> — النص المرجعي</li>
            <li><code>keyword</code> — الكلمة المفتاحية</li>
            <li><code>category</code> — الفئة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    sample_df = pd.DataFrame({
        'text': ['القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. تقع على ضفاف نهر النيل وتعد من أكبر المدن في أفريقيا.'],
        'keyword': ['القاهرة'],
        'category': ['جغرافيا']
    })
    st.download_button("📥 تحميل نموذج CSV", data=sample_df.to_csv(index=False).encode('utf-8-sig'), file_name="sample_input.csv", mime="text/csv")
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required = {'text', 'keyword', 'category'}
            if not required.issubset(set(df.columns)):
                missing = required - set(df.columns)
                st.error(f"❌ أعمدة مفقودة: {', '.join(missing)}")
            else:
                st.markdown(f'<div class="info-box">📊 عدد الصفوف: <strong>{len(df)}</strong></div>', unsafe_allow_html=True)
                with st.expander("👁️ معاينة البيانات"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if len(df) > 20:
                    st.markdown("""
                    <div class="info-box">⚠️ عدد الصفوف كبير. قد يستغرق الأمر وقتاً طويلاً وقد تتجاوز حد طلبات API المجاني.</div>
                    """, unsafe_allow_html=True)
                
                st.markdown('<p class="rtl-text"><strong>⏱️ التأخير بين الطلبات (ثانية):</strong></p>', unsafe_allow_html=True)
                delay = st.slider("delay", 1, 15, 3, label_visibility="collapsed", help="لتجنّب تجاوز حد الطلبات")
                
                process_btn = st.button("🚀 بدء المعالجة الدفعية", use_container_width=True, type="primary", disabled=not api_valid, key="process_csv")
                
                if process_btn:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for index, row in df.iterrows():
                        progress = (index + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.markdown(
                            f'<p class="rtl-text" style="color:#e67e22;font-weight:bold;">⏳ معالجة الصف {index + 1} من {len(df)}...</p>',
                            unsafe_allow_html=True
                        )
                        
                        try:
                            result = generate_clues(
                                text=str(row['text']),
                                keyword=str(row['keyword']),
                                category=str(row['category']),
                                provider=provider,
                                api_key=api_key,
                                model_name=model_name,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                num_clues=num_clues
                            )
                            if result["success"]:
                                clues_text = '\n'.join(result['clues']) if result['clues'] else result['raw_response']
                                status = 'نجاح ✅'
                            else:
                                clues_text = ''
                                status = f'خطأ ❌: {result["error"]}'
                            results.append({
                                'text': row['text'],
                                'keyword': row['keyword'],
                                'category': row['category'],
                                'generated_clues': clues_text,
                                'status': status
                            })
                        except Exception as e:
                            results.append({
                                'text': row['text'],
                                'keyword': row['keyword'],
                                'category': row['category'],
                                'generated_clues': '',
                                'status': f'خطأ ❌: {str(e)}'
                            })
                        
                        if index < len(df) - 1:
                            time.sleep(delay)
                    
                    progress_bar.progress(1.0)
                    status_text.markdown('<p class="rtl-text" style="color:#27ae60;font-weight:bold;">✅ تمت المعالجة!</p>', unsafe_allow_html=True)
                    
                    result_df = pd.DataFrame(results)
                    st.markdown("---")
                    st.markdown('<h3 class="rtl-text">📊 النتائج:</h3>', unsafe_allow_html=True)
                    success_count = sum(1 for r in results if 'نجاح' in r['status'])
                    error_count = len(results) - success_count
                    c1, c2, c3 = st.columns(3)
                    c1.metric("الإجمالي", len(results))
                    c2.metric("نجاح ✅", success_count)
                    c3.metric("أخطاء ❌", error_count)
                    
                    st.dataframe(result_df, use_container_width=True)
                    
                    with st.expander("📋 عرض النتائج بالتفصيل"):
                        for res in results:
                            st.markdown(
                                f'<div class="result-box">'
                                f'<p><strong>الكلمة:</strong> {res["keyword"]}</p>'
                                f'<p><strong>الفئة:</strong> {res["category"]}</p>'
                                f'<p><strong>الأدلة:</strong><br>{res["generated_clues"]}</p>'
                                f'<p><strong>الحالة:</strong> {res["status"]}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    
                    st.download_button("📥 تحميل النتائج CSV", data=result_df.to_csv(index=False).encode('utf-8-sig'), file_name="generated_clues_output.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"❌ خطأ في قراءة الملف: {str(e)}")

# =============================================
# التذييل
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; padding:20px;">
    <p>🧩 مولّد أدلة الكلمات المتقاطعة العربية — يدعم Hugging Face و Mistral</p>
    <p style="font-size:0.85rem;"> مبني باستخدام Streamlit و Hugging Face / Mistral API</p>
</div>
""", unsafe_allow_html=True)
