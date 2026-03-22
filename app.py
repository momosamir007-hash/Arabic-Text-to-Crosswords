import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
# تنسيق CSS للعربية (RTL)
# =============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
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
    font-size: 1.1rem;
    line-height: 1.8;
}
.result-box {
    background-color: #eaf2f8;
    border-right: 5px solid #2980b9;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
    direction: rtl;
    text-align: right;
}
.clue-item {
    background-color: #d5f5e3;
    border-right: 4px solid #27ae60;
    padding: 12px 18px;
    border-radius: 6px;
    margin: 8px 0;
    direction: rtl;
    text-align: right;
    font-size: 1.1rem;
}
.error-box {
    background-color: #fadbd8;
    border-right: 5px solid #e74c3c;
    padding: 15px;
    border-radius: 8px;
    direction: rtl;
    text-align: right;
}
.info-box {
    background-color: #fef9e7;
    border-right: 5px solid #f39c12;
    padding: 15px;
    border-radius: 8px;
    direction: rtl;
    text-align: right;
    margin: 10px 0;
}
.stTextArea textarea {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif;
    font-size: 1rem;
}
.stTextInput input {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif;
    font-size: 1rem;
}
.sidebar .sidebar-content {
    direction: rtl;
}
div[data-testid="stSidebar"] {
    direction: rtl;
    text-align: right;
}
.status-running {
    color: #e67e22;
    font-weight: bold;
}
.status-done {
    color: #27ae60;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# الثوابت
# =============================================
MODEL_NAME = "Kamyar-zeinalipour/Llama3-8B-Ar-Text-to-Cross"
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
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted_prompt

def extract_response(text: str) -> str:
    """استخراج رد المساعد من النص المُولّد."""
    try:
        parts = text.split('<|end_header_id|>')
        if len(parts) >= 4:
            response_part = parts[3]
        elif len(parts) >= 3:
            response_part = parts[2]
        else:
            return text
        # تنظيف النص
        response_part = response_part.split('<|eot_id|>')[0]
        response_part = response_part.split('<|end_of_text|>')[0]
        response_part = response_part.strip()
        return response_part
    except (IndexError, AttributeError):
        return text

def extract_clues(generated_text: str, max_clues: int = 3) -> list:
    """استخراج الأدلة من النص المُولّد."""
    if not generated_text:
        return []
    # البحث عن الأدلة بعدة أنماط
    patterns = [
        r'(CLUE\s*\d+\s*:\s*.+?)(?=CLUE\s*\d+|$)',
        r'(Clue\s*\d+\s*:\s*.+?)(?=Clue\s*\d+|$)',
        r'(\d+\s*[\.\-\)]\s*.+?)(?=\d+\s*[\.\-\)]|$)',
    ]
    clues = []
    for pattern in patterns:
        matches = re.findall(pattern, generated_text, re.DOTALL | re.IGNORECASE)
        if matches:
            clues = [m.strip() for m in matches if m.strip()]
            break
    # إذا لم يتم العثور على أدلة بالأنماط، نقسم بالأسطر
    if not clues:
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        clues = lines
    return clues[:max_clues]

@st.cache_resource(show_spinner=False)
def load_model():
    """تحميل النموذج والمُرمِّز مع التخزين المؤقت."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device

def generate_clues(
    text: str,
    keyword: str,
    category: str,
    model,
    tokenizer,
    device: str,
    temperature: float = 0.1,
    max_new_tokens: int = 256,
    num_clues: int = 3
) -> dict:
    """توليد أدلة الكلمات المتقاطعة."""
    prompt = format_prompt(text, keyword, category)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_k=50,
            top_p=0.95,
            do_sample=True if temperature > 0 else False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        extracted = extract_response(full_response)
        clues = extract_clues(extracted, max_clues=num_clues)
    return {
        "clues": clues,
        "raw_response": extracted,
        "full_response": full_response
    }

# =============================================
# الشريط الجانبي - الإعدادات
# =============================================
with st.sidebar:
    st.markdown('<h2 style="text-align:right; color:#1a5276;">⚙️ الإعدادات</h2>', unsafe_allow_html=True)
    st.markdown("---")
    # درجة الحرارة
    st.markdown('<p class="rtl-text"><strong>🌡️ درجة الحرارة (Temperature)</strong></p>', unsafe_allow_html=True)
    temperature = st.slider(
        "درجة الحرارة",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        label_visibility="collapsed",
        help="قيمة أقل = نتائج أكثر تحديداً، قيمة أعلى = نتائج أكثر تنوعاً"
    )
    # عدد الأدلة
    st.markdown('<p class="rtl-text"><strong>🔢 عدد الأدلة المطلوبة</strong></p>', unsafe_allow_html=True)
    num_clues = st.slider(
        "عدد الأدلة",
        min_value=1,
        max_value=5,
        value=3,
        label_visibility="collapsed"
    )
    # الحد الأقصى للرموز الجديدة
    st.markdown('<p class="rtl-text"><strong>📏 الحد الأقصى للرموز الجديدة</strong></p>', unsafe_allow_html=True)
    max_tokens = st.slider(
        "الحد الأقصى للرموز",
        min_value=64,
        max_value=512,
        value=256,
        step=32,
        label_visibility="collapsed"
    )
    st.markdown("---")
    # معلومات النموذج
    st.markdown('<p class="rtl-text"><strong>📦 معلومات النموذج</strong></p>', unsafe_allow_html=True)
    device_info = "GPU 🟢" if torch.cuda.is_available() else "CPU 🟡"
    st.markdown(f"""
    <div class="info-box">
        <p><strong>النموذج:</strong> Llama3-8B-Ar-Text-to-Cross</p>
        <p><strong>الجهاز:</strong> {device_info}</p>
    </div>
    """, unsafe_allow_html=True)
    if not torch.cuda.is_available():
        st.markdown("""
        <div class="error-box">
            <p>⚠️ <strong>تحذير:</strong> لا يوجد GPU متاح. سيعمل النموذج على CPU وقد يكون بطيئاً جداً.</p>
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
    '<p class="sub-title">أداة ذكية لإنشاء أدلة كلمات متقاطعة عربية '
    'باستخدام نموذج Llama3 المُدرَّب</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# =============================================
# تحميل النموذج
# =============================================
model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.device = None

# زر تحميل النموذج
col_load1, col_load2, col_load3 = st.columns([1, 2, 1])
with col_load2:
    if st.session_state.model is None:
        st.markdown("""
        <div class="info-box">
            <p>📥 يجب تحميل النموذج أولاً قبل البدء في التوليد</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📥 تحميل النموذج", use_container_width=True, type="primary"):
            with st.spinner("⏳ جارٍ تحميل النموذج... قد يستغرق بضع دقائق"):
                try:
                    model, tokenizer, device = load_model()
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.device = device
                    st.success("✅ تم تحميل النموذج بنجاح!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
    else:
        st.markdown("""
        <div class="clue-item">
            <p>✅ النموذج جاهز للاستخدام</p>
        </div>
        """, unsafe_allow_html=True)
        model_loaded = True

st.markdown("---")

# =============================================
# التبويبات الرئيسية
# =============================================
tab1, tab2 = st.tabs(["📝 إدخال يدوي", "📁 رفع ملف CSV"])

# =============================================
# التبويب الأول: الإدخال اليدوي
# =============================================
with tab1:
    st.markdown(
        '<h3 class="rtl-text">📝 أدخل البيانات لتوليد أدلة الكلمات المتقاطعة</h3>',
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
    # الكلمة المفتاحية والفئة
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
    # زر التوليد
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(
        "🚀 توليد الأدلة",
        use_container_width=True,
        type="primary",
        disabled=not model_loaded,
        key="generate_manual"
    )
    if generate_btn:
        if not input_text.strip():
            st.warning("⚠️ يرجى إدخال النص المرجعي")
        elif not input_keyword.strip():
            st.warning("⚠️ يرجى إدخال الكلمة المفتاحية")
        else:
            with st.spinner("⏳ جارٍ توليد الأدلة..."):
                try:
                    result = generate_clues(
                        text=input_text,
                        keyword=input_keyword,
                        category=input_category,
                        model=st.session_state.model,
                        tokenizer=st.session_state.tokenizer,
                        device=st.session_state.device,
                        temperature=temperature,
                        max_new_tokens=max_tokens,
                        num_clues=num_clues
                    )
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
                    # عرض الاستجابة الكاملة في قسم قابل للطي
                    with st.expander("🔍 عرض الاستجابة الكاملة"):
                        st.code(result["raw_response"], language=None)
                except Exception as e:
                    st.markdown(
                        f'<div class="error-box">❌ حدث خطأ: {str(e)}</div>',
                        unsafe_allow_html=True
                    )

# =============================================
# التبويب الثاني: رفع ملف CSV
# =============================================
with tab2:
    st.markdown(
        '<h3 class="rtl-text">📁 معالجة دفعية عبر ملف CSV</h3>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box">
        <p><strong>📋 تنسيق ملف CSV المطلوب:</strong></p>
        <p>يجب أن يحتوي الملف على الأعمدة التالية:</p>
        <ul style="direction:rtl; text-align:right;">
            <li><code>text</code> - النص المرجعي</li>
            <li><code>keyword</code> - الكلمة المفتاحية</li>
            <li><code>category</code> - الفئة</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # زر تحميل نموذج CSV
    sample_df = pd.DataFrame({
        'text': [
            'القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها. '
            'تقع على ضفاف نهر النيل وتعد من أكبر المدن في أفريقيا والشرق الأوسط.'
        ],
        'keyword': ['القاهرة'],
        'category': ['جغرافيا']
    })
    csv_sample = sample_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 تحميل نموذج CSV",
        data=csv_sample,
        file_name="sample_input.csv",
        mime="text/csv",
        key="download_sample"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # رفع الملف
    uploaded_file = st.file_uploader(
        "اختر ملف CSV",
        type=["csv"],
        label_visibility="collapsed",
        key="csv_upload"
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # التحقق من الأعمدة
            required_cols = {'text', 'keyword', 'category'}
            if not required_cols.issubset(set(df.columns)):
                missing = required_cols - set(df.columns)
                st.error(f"❌ الأعمدة التالية مفقودة: {', '.join(missing)}")
            else:
                st.markdown(
                    f'<div class="info-box">'
                    f'<p>📊 عدد الصفوف: <strong>{len(df)}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # عرض معاينة
                with st.expander("👁️ معاينة البيانات"):
                    st.dataframe(df.head(10), use_container_width=True)
                # زر بدء المعالجة
                process_btn = st.button(
                    "🚀 بدء المعالجة الدفعية",
                    use_container_width=True,
                    type="primary",
                    disabled=not model_loaded,
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
                            f'<p class="rtl-text status-running">'
                            f'⏳ جارٍ معالجة الصف {index + 1} من {len(df)}...'
                            f'</p>',
                            unsafe_allow_html=True
                        )
                        try:
                            result = generate_clues(
                                text=str(row['text']),
                                keyword=str(row['keyword']),
                                category=str(row['category']),
                                model=st.session_state.model,
                                tokenizer=st.session_state.tokenizer,
                                device=st.session_state.device,
                                temperature=temperature,
                                max_new_tokens=max_tokens,
                                num_clues=num_clues
                            )
                            clues_text = '\n'.join(result['clues']) if result['clues'] else result['raw_response']
                            results.append({
                                'text': row['text'],
                                'keyword': row['keyword'],
                                'category': row['category'],
                                'generated_clues': clues_text,
                                'status': 'نجاح ✅'
                            })
                        except Exception as e:
                            results.append({
                                'text': row['text'],
                                'keyword': row['keyword'],
                                'category': row['category'],
                                'generated_clues': '',
                                'status': f'خطأ ❌: {str(e)}'
                            })
                    progress_bar.progress(1.0)
                    status_text.markdown(
                        '<p class="rtl-text status-done">✅ تمت المعالجة بنجاح!</p>',
                        unsafe_allow_html=True
                    )
                    # عرض النتائج
                    result_df = pd.DataFrame(results)
                    st.markdown("---")
                    st.markdown(
                        '<h3 class="rtl-text">📊 النتائج:</h3>',
                        unsafe_allow_html=True
                    )
                    # إحصائيات
                    success_count = sum(1 for r in results if 'نجاح' in r['status'])
                    error_count = len(results) - success_count
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        st.metric("إجمالي الصفوف", len(results))
                    with col_s2:
                        st.metric("نجاح ✅", success_count)
                    with col_s3:
                        st.metric("أخطاء ❌", error_count)
                    # عرض جدول النتائج
                    st.dataframe(result_df, use_container_width=True)
                    # عرض كل نتيجة بالتفصيل
                    with st.expander("📋 عرض النتائج بالتفصيل"):
                        for i, res in enumerate(results):
                            st.markdown(
                                f'<div class="result-box">'
                                f'<p><strong>الكلمة المفتاحية:</strong> {res["keyword"]}</p>'
                                f'<p><strong>الفئة:</strong> {res["category"]}</p>'
                                f'<p><strong>الأدلة:</strong></p>'
                                f'<p>{res["generated_clues"]}</p>'
                                f'<p><strong>الحالة:</strong> {res["status"]}</p>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    # زر تحميل النتائج
                    output_csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 تحميل النتائج كملف CSV",
                        data=output_csv,
                        file_name="generated_clues_output.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_results"
                    )
        except Exception as e:
            st.error(f"❌ خطأ في قراءة الملف: {str(e)}")

# =============================================
# التذييل
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; padding:20px;">
    <p>🧩 مولّد أدلة الكلمات المتقاطعة العربية</p>
    <p style="font-size:0.85rem;"> مبني باستخدام Streamlit و نموذج Llama3-8B-Ar-Text-to-Cross </p>
</div>
""", unsafe_allow_html=True)
