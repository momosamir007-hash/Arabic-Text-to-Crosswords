import streamlit as st
import pandas as pd
import requests
import re

# إعداد الصفحة
st.set_page_config(page_title="مُولّد تلميحات الكلمات المتقاطعة", page_icon="🧩", layout="wide")

st.title("🧩 مُولّد تلميحات الكلمات المتقاطعة باللغة العربية")
st.markdown("""
هذا التطبيق يقوم بتوليد تلميحات لألعاب الكلمات المتقاطعة باللغة العربية عبر الاتصال بواجهة برمجة تطبيقات Hugging Face (API) للنموذج `Llama3-8B-Ar-Text-to-Cross`.
""")

# --- إعدادات الـ API ---
# الرابط الجديد لخوادم Hugging Face
API_URL = "https://router.huggingface.co/hf-inference/models/Kamyar-zeinalipour/Llama3-8B-Ar-Text-to-Cross"

# جلب التوكن من إعدادات Streamlit
hf_token = st.secrets["HF_TOKEN"]
headers = {"Authorization": f"Bearer {hf_token}"}

# --- الدوال الأساسية ---
simple_prompt = (
    'Create Arabic crossword clues for a specified keyword in Arabic, '
    'using the provided text and focusing on the indicated category.'
)

def format_row(text, keyword, category):
    user_message = (
        f"{simple_prompt}\n\n"
        f"TEXT: {text}\n\n"
        f"KEYWORD: {keyword}\n\n"
        f"CATEGORY: {category}"
    )

    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an invaluable assistant who creates Arabic crossword clues based on the "
        f"provided Arabic text, keyword, and specific category.\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return formatted_prompt

def get_first_three_clues(generated_text):
    if not generated_text: return "لم يتم توليد تلميحات."
    # البحث عن الأسطر التي تبدأ بـ CLUE
    clues = re.findall(r'(CLUE\d+:.*)', generated_text)
    first_three_clues = clues[:3]
    return '\n'.join(first_three_clues) if first_three_clues else generated_text

def get_code_completion(prompt, temperature):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": temperature,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False # لمنع إعادة إرسال السؤال مع الإجابة
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # 1. التحقق من نجاح الطلب (كود 200)
        if response.status_code == 200:
            response_data = response.json()
            # استخراج النص المولد
            if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
                return response_data[0]["generated_text"]
            else:
                return f"❌ استجابة غير متوقعة من الخادم: {response_data}"
                
        # 2. التعامل مع حالة (النموذج قيد التحميل - كود 503 عادةً)
        elif response.status_code == 503:
            try:
                error_info = response.json()
                wait_time = error_info.get("estimated_time", 20)
                return f"⏳ النموذج قيد التحميل الآن على خوادم Hugging Face (يحتاج حوالي {wait_time:.1f} ثانية للاستيقاظ). يرجى الانتظار قليلاً ثم المحاولة مرة أخرى."
            except:
                return f"⏳ النموذج قيد التحميل، يرجى المحاولة بعد قليل. (رمز الخطأ: {response.status_code})"
                
        # 3. التعامل مع أخطاء الصلاحيات أو الأخطاء الأخرى
        else:
            error_message = response.text
            return f"❌ خطأ من الخادم (كود {response.status_code}): \n{error_message}"
            
    except requests.exceptions.RequestException as e:
        return f"❌ حدث خطأ في الاتصال بالشبكة: {e}"
    except ValueError as e:
         return f"❌ فشل في قراءة استجابة الخادم: {e}\nمحتوى الاستجابة: {response.text}"

def process_single_entry(text, keyword, category, temperature):
    prompt = format_row(text, keyword, category)
    response_text = get_code_completion(prompt, temperature)
    
    # التحقق مما إذا كانت النتيجة رسالة خطأ أو انتظار
    if response_text.startswith("❌") or response_text.startswith("⏳"):
        return response_text
        
    return get_first_three_clues(response_text)

# --- إعدادات الواجهة الجانبية ---
st.sidebar.header("إعدادات التوليد ⚙️")
temperature = st.sidebar.slider("درجة الحرارة (Temperature)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, help="قيمة أقل تعني إجابات دقيقة، وقيمة أعلى تعني إجابات أكثر تنوعاً.")

# --- تقسيم الواجهة إلى قسمين ---
tab1, tab2 = st.tabs(["📝 إدخال يدوي مباشر", "📁 معالجة ملف CSV"])

# القسم الأول: إدخال يدوي
with tab1:
    st.subheader("تجربة سريعة لنص واحد")
    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_area("النص (Text)", "يعتبر النيل أطول نهر في العالم.")
    with col2:
        input_keyword = st.text_input("الكلمة المفتاحية (Keyword)", "نيل")
        input_category = st.text_input("التصنيف بالإنجليزية (Category)", "Geography")
    
    if st.button("توليد التلميح 🪄", type="primary"):
        if input_text and input_keyword and input_category:
            with st.spinner("جاري الاتصال بخوادم Hugging Face..."):
                result = process_single_entry(input_text, input_keyword, input_category, temperature)
                if result.startswith("❌"):
                    st.error(result)
                elif result.startswith("⏳"):
                    st.warning(result)
                else:
                    st.success("تم التوليد بنجاح!")
                    st.text_area("النتيجة:", result, height=150)
        else:
            st.warning("يرجى ملء جميع الحقول أولاً.")

# القسم الثاني: معالجة ملفات CSV
with tab2:
    st.subheader("معالجة بيانات متعددة عبر ملف CSV")
    st.markdown("يجب أن يحتوي الملف على الأعمدة التالية: `text`, `keyword`, `category`")
    
    uploaded_file = st.file_uploader("ارفع ملف CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("معاينة البيانات المرفوعة:")
        st.dataframe(df.head())
        
        if st.button("بدء معالجة الملف 🚀"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            outputs = []
            total_rows = len(df)
            
            for index, row in df.iterrows():
                status_text.text(f"جاري معالجة الصف {index + 1} من {total_rows}...")
                
                result = process_single_entry(row['text'], row['keyword'], row['category'], temperature)
                outputs.append({
                    'text': row['text'],
                    'keyword': row['keyword'],
                    'category': row['category'],
                    'Generated Arabic Crossword Clue': result
                })
                
                progress_bar.progress((index + 1) / total_rows)
            
            status_text.text("تمت المعالجة بنجاح! ✅")
            
            # عرض النتائج
            output_df = pd.DataFrame(outputs)
            st.dataframe(output_df)
            
            # توفير زر للتحميل
            csv_data = output_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 تحميل النتائج (CSV)",
                data=csv_data,
                file_name='crossword_clues_output.csv',
                mime='text/csv',
            )
