import streamlit as st
import pandas as pd
import re
import requests
import time
import json
import io

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
MODEL_NAME = "Kamyar-zeinalipour/Llama3-8B-Ar-Text-to-Cross"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"
SYSTEM_PROMPT = (
    "You are an invaluable assistant who creates Arabic crossword clues based on the "
    "provided Arabic text, keyword, and specific category."
)
USER_INSTRUCTION = (
    "Create Arabic crossword clues for a specified keyword in Arabic, "
    "using the provided text and focusing on the indicated category."
)
CATEGORIES = [
    "دين", "تاريخ", "جغرافيا", "علوم", "أدب", "رياضة", "فن", "سياسة", "اقتصاد", "تكنولوجيا", "طب", "ثقافة عامة", "أخرى"
]

# =============================================
# الدوال المساعدة
# =============================================
def format_prompt(text: str, keyword: str, category: str) -> str:
    """تنسيق المدخلات إلى الشكل المطلوب للنموذج."""
    user_message = (
        f"{USER_INSTRUCTION}\n\n"
        f"TEXT: {text}\n\n"
        f"KEYWORD: {keyword}\n\n"
        f"CATEGORY: {category}"
    )
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted

def extract_response(text: str) -> str:
    """استخراج رد المساعد من النص المُولّد."""
    if not text:
        return ""
    try:
        # محاولة الاستخراج عبر الفاصل
        parts = text.split('<|end_header_id|>')
        if len(parts) >= 4:
            response_part = parts[3]
        elif len(parts) >= 3:
            response_part = parts[2]
        else:
            response_part = text
        # تنظيف
        for tag in ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>']:
            response_part = response_part.replace(tag, '')
        return response_part.strip()
    except Exception:
        return text.strip()

def extract_clues(generated_text: str, max_clues: int = 3) -> list:
    """استخراج الأدلة من النص المُولّد."""
    if not generated_text:
        return []
    patterns = [
        r'(CLUE\s*\d+\s*:\s*.+?)(?=CLUE\s*\d+\s*:|$)',
        r'(Clue\s*\d+\s*:\s*.+?)(?=Clue\s*\d+\s*:|$)',
        r'(\d+\s*[\.\-\)]\s*.+?)(?=\d+\s*[\.\-\)]|$)',
    ]
    clues = []
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
        if matches:
            clues = [m.strip() for m in matches if m.strip()]
            break
    if not clues:
        lines = [l.strip() for l in generated_text.split('\n') if l.strip()]
        clues = lines
    return clues[:max_clues]

def call_hf_api(
    prompt: str,
    api_key: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    max_retries: int = 3
) -> dict:
    """استدعاء Hugging Face Inference API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "do_sample": temperature > 0,
            "return_full_text": True
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            # ── النموذج قيد التحميل ──
            if response.status_code == 503:
                data = response.json()
                wait_time = data.get("estimated_time", 30)
                st.warning(
                    f"⏳ النموذج قيد التحميل... الانتظار {int(wait_time)} ثانية "
                    f"(المحاولة {attempt + 1}/{max_retries})"
                )
                time.sleep(min(wait_time, 60))
                continue

            # ── تجاوز الحصة ──
            if response.status_code == 429:
                st.warning(
                    f"⏳ تم تجاوز حد الطلبات، الانتظار 10 ثوانٍ... "
                    f"(المحاولة {attempt + 1}/{max_retries})"
                )
                time.sleep(10)
                continue

            # ── خطأ مصادقة ──
            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "❌ مفتاح API غير صالح. تأكّد من صلاحيته."
                }

            # ── أخطاء أخرى ──
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"❌ خطأ من الخادم ({response.status_code}): "
                             f"{response.text[:300]}"
                }

            # ── نجاح ──
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated = result.get("generated_text", "")
            else:
                generated = str(result)

            return {"success": True, "generated_text": generated}

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(
                    f"⏳ انتهت مهلة الطلب، إعادة المحاولة... "
                    f"({attempt + 1}/{max_retries})"
                )
                time.sleep(5)
            else:
                return {
                    "success": False,
                    "error": "❌ انتهت مهلة الاتصال بالخادم بعد عدة محاولات."
                }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "❌ فشل الاتصال بالخادم. تحقّق من اتصال الإنترنت."
            }
        except Exception as e:
            return {"success": False, "error": f"❌ خطأ غير متوقع: {str(e)}"}

    return {
        "success": False,
        "error": "❌ فشلت جميع المحاولات. حاول مرة أخرى لاحقاً."
    }

def generate_clues(
    text: str,
    keyword: str,
    category: str,
    api_key: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    num_clues: int = 3
) -> dict:
    """توليد الأدلة باستخدام API."""
    prompt = format_prompt(text, keyword, category)
    api_result = call_hf_api(prompt, api_key, temperature, max_new_tokens)

    if not api_result["success"]:
        return {
            "success": False,
            "error": api_result["error"],
            "clues": [],
            "raw_response": ""
        }

    raw = api_result["generated_text"]
    extracted = extract_response(raw)
    clues = extract_clues(extracted, max_clues=num_clues)

    return {
        "success": True,
        "clues": clues,
        "raw_response": extracted,
        "full_response": raw
    }

def verify_api_key(api_key: str) -> bool:
    """التحقق السريع من صلاحية مفتاح API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        return r.status_code == 200
    except Exception:
        return False

# =============================================
# الشريط الجانبي
# =============================================
with st.sidebar:
    st.markdown(
        '<h2 style="text-align:right;color:#1a5276;">⚙️ الإعدادات</h2>',
        unsafe_allow_html=True
    )

    # ── مفتاح API ──
    st.markdown("---")
    st.markdown(
        '<p class="rtl-text"><strong>🔑 مفتاح Hugging Face API</strong></p>',
        unsafe_allow_html=True
    )
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
        help="احصل عليه من: https://huggingface.co/settings/tokens"
    )

    # التحقق من المفتاح
    api_valid = False
    if api_key:
        if api_key.startswith("hf_") and len(api_key) > 10:
            api_valid = True
            st.markdown(
                '<div class="success-box">✅ تم إدخال المفتاح</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="error-box">❌ صيغة المفتاح غير صحيحة</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown("""
        <div class="info-box">
            <p>📌 تحتاج مفتاح API من Hugging Face</p>
            <p style="font-size:0.85rem;">
                <a href="https://huggingface.co/settings/tokens" target="_blank">
                اضغط هنا للحصول على مفتاح
                </a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── درجة الحرارة ──
    st.markdown(
        '<p class="rtl-text"><strong>🌡️ درجة الحرارة</strong></p>',
        unsafe_allow_html=True
    )
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.1, 0.05,
        label_visibility="collapsed",
        help="أقل = أكثر دقة، أعلى = أكثر إبداعاً"
    )

    # ── عدد الأدلة ──
    st.markdown(
        '<p class="rtl-text"><strong>🔢 عدد الأدلة</strong></p>',
        unsafe_allow_html=True
    )
    num_clues = st.slider(
        "عدد الأدلة",
        1, 5, 3,
        label_visibility="collapsed"
    )

    # ── الحد الأقصى للرموز ──
    st.markdown(
        '<p class="rtl-text"><strong>📏 الحد الأقصى للرموز</strong></p>',
        unsafe_allow_html=True
    )
    max_tokens = st.slider(
        "Max tokens",
        64, 512, 256, 32,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div class="info-box">
        <p><strong>📦 النموذج:</strong></p>
        <p style="font-size:0.85rem;">Llama3-8B-Ar-Text-to-Cross</p>
        <p><strong>🌐 الوضع:</strong> API (Hugging Face)</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# العنوان الرئيسي
# =============================================
st.markdown(
    '<h1 class="main-title">🧩 مولّد أدلة الكلمات المتقاطعة العربية</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-title">أداة ذكية لإنشاء أدلة كلمات متقاطعة عربية عبر API — '
    'بدون تحميل النموذج محلياً</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# =============================================
# التبويبات
# =============================================
tab1, tab2 = st.tabs(["📝 إدخال يدوي", "📁 رفع ملف CSV"])

# =============================================
# تبويب 1: إدخال يدوي
# =============================================
with tab1:
    st.markdown(
        '<h3 class="rtl-text">📝 أدخل البيانات لتوليد الأدلة</h3>',
        unsafe_allow_html=True
    )

    # النص
    st.markdown(
        '<p class="rtl-text"><strong>📄 النص المرجعي:</strong></p>',
        unsafe_allow_html=True
    )
    input_text = st.text_area(
        "النص",
        height=200,
        placeholder="أدخل النص العربي هنا...",
        label_visibility="collapsed",
        key="manual_text"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<p class="rtl-text"><strong>🔑 الكلمة المفتاحية:</strong></p>',
            unsafe_allow_html=True
        )
        input_keyword = st.text_input(
            "الكلمة المفتاحية",
            placeholder="أدخل الكلمة المفتاحية...",
            label_visibility="collapsed",
            key="manual_keyword"
        )
    with col2:
        st.markdown(
            '<p class="rtl-text"><strong>📂 الفئة:</strong></p>',
            unsafe_allow_html=True
        )
        input_category = st.selectbox(
            "الفئة",
            options=CATEGORIES,
            label_visibility="collapsed",
            key="manual_category"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(
        "🚀 توليد الأدلة",
        use_container_width=True,
        type="primary",
        disabled=not api_valid,
        key="gen_manual"
    )

    if not api_valid and not api_key:
        st.markdown("""
        <div class="info-box">
            <p>🔑 أدخل مفتاح API في الشريط الجانبي للبدء</p>
        </div>
        """, unsafe_allow_html=True)

    if generate_btn:
        if not input_text.strip():
            st.warning("⚠️ يرجى إدخال النص المرجعي")
        elif not input_keyword.strip():
            st.warning("⚠️ يرجى إدخال الكلمة المفتاحية")
        else:
            with st.spinner("⏳ جارٍ توليد الأدلة عبر API..."):
                result = generate_clues(
                    text=input_text,
                    keyword=input_keyword,
                    category=input_category,
                    api_key=api_key,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    num_clues=num_clues
                )

                if result["success"]:
                    st.markdown("---")
                    st.markdown(
                        '<h3 class="rtl-text">✨ الأدلة المُولَّدة:</h3>',
                        unsafe_allow_html=True
                    )
                    if result["clues"]:
                        for i, clue in enumerate(result["clues"], 1):
                            st.markdown(
                                f'<div class="clue-item">'
                                f'<strong>الدليل {i}:</strong> {clue}'
                                f'</div>',
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
# تبويب 2: رفع CSV
# =============================================
with tab2:
    st.markdown(
        '<h3 class="rtl-text">📁 معالجة دفعية عبر ملف CSV</h3>',
        unsafe_allow_html=True
    )
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

    # نموذج CSV
    sample_df = pd.DataFrame({
        'text': [
            'القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. '
            'تقع على ضفاف نهر النيل وتعد من أكبر المدن في أفريقيا.'
        ],
        'keyword': ['القاهرة'],
        'category': ['جغرافيا']
    })
    st.download_button(
        "📥 تحميل نموذج CSV",
        data=sample_df.to_csv(index=False).encode('utf-8-sig'),
        file_name="sample_input.csv",
        mime="text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "اختر ملف CSV",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required = {'text', 'keyword', 'category'}
            if not required.issubset(set(df.columns)):
                missing = required - set(df.columns)
                st.error(f"❌ أعمدة مفقودة: {', '.join(missing)}")
            else:
                st.markdown(
                    f'<div class="info-box">'
                    f'📊 عدد الصفوف: <strong>{len(df)}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                with st.expander("👁️ معاينة البيانات"):
                    st.dataframe(df.head(10), use_container_width=True)

                # تحذير عند وجود عدد كبير
                if len(df) > 20:
                    st.markdown("""
                    <div class="info-box">
                        ⚠️ عدد الصفوف كبير. قد يستغرق الأمر وقتاً طويلاً وقد تتجاوز حد طلبات API المجاني.
                    </div>
                    """, unsafe_allow_html=True)

                # تأخير بين الطلبات
                st.markdown(
                    '<p class="rtl-text"><strong>⏱️ التأخير بين الطلبات (ثانية):</strong></p>',
                    unsafe_allow_html=True
                )
                delay = st.slider(
                    "delay",
                    1, 15, 3,
                    label_visibility="collapsed",
                    help="لتجنّب تجاوز حد الطلبات"
                )

                process_btn = st.button(
                    "🚀 بدء المعالجة الدفعية",
                    use_container_width=True,
                    type="primary",
                    disabled=not api_valid,
                    key="process_csv"
                )

                if process_btn:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for index, row in df.iterrows():
                        progress = (index + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.markdown(
                            f'<p class="rtl-text" style="color:#e67e22;font-weight:bold;">'
                            f'⏳ معالجة الصف {index + 1} من {len(df)}...'
                            f'</p>',
                            unsafe_allow_html=True
                        )

                        try:
                            result = generate_clues(
                                text=str(row['text']),
                                keyword=str(row['keyword']),
                                category=str(row['category']),
                                api_key=api_key,
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

                        # تأخير بين الطلبات
                        if index < len(df) - 1:
                            time.sleep(delay)

                    progress_bar.progress(1.0)
                    status_text.markdown(
                        '<p class="rtl-text" style="color:#27ae60;font-weight:bold;">'
                        '✅ تمت المعالجة!</p>',
                        unsafe_allow_html=True
                    )

                    # ── عرض النتائج ──
                    result_df = pd.DataFrame(results)
                    st.markdown("---")
                    st.markdown(
                        '<h3 class="rtl-text">📊 النتائج:</h3>',
                        unsafe_allow_html=True
                    )
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

                    # تحميل النتائج
                    st.download_button(
                        "📥 تحميل النتائج CSV",
                        data=result_df.to_csv(index=False).encode('utf-8-sig'),
                        file_name="generated_clues_output.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"❌ خطأ في قراءة الملف: {str(e)}")

# =============================================
# التذييل
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; padding:20px;">
    <p>🧩 مولّد أدلة الكلمات المتقاطعة العربية — نسخة API</p>
    <p style="font-size:0.85rem;"> مبني باستخدام Streamlit و Hugging Face Inference API </p>
</div>
""", unsafe_allow_html=True)
