import streamlit as st
import pandas as pd
import requests
import re
import time

# --- 1. إعدادات الصفحة والواجهة ---
st.set_page_config(page_title="مُولّد تلميحات الكلمات المتقاطعة", page_icon="🧩", layout="wide")

st.title("🧩 مُولّد تلميحات الكلمات المتقاطعة باللغة العربية")
st.markdown("""
هذا التطبيق يستخدم الذكاء الاصطناعي لتوليد تلميحات ذكية للألغاز بناءً على سياق نصي.
""")

# --- 2. إعدادات الـ API والنموذج ---
# اخترنا Qwen2.5-7B لأنه الأفضل حالياً للغة العربية ويدعم التشغيل المجاني
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

# التحقق من وجود التوكن في Secrets
if "HF_TOKEN" not in st.secrets:
    st.error("❌ خطأ: لم يتم العثور على HF_TOKEN في إعدادات Secrets. يرجى إضافته أولاً.")
    st.stop()

headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# --- 3. الدوال الأساسية ---

def get_ai_response(prompt, temperature):
    """إرسال الطلب إلى خوادم Hugging Face ومعالجة الرد"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0:
                return res_json[0].get("generated_text", "").strip()
            return str(res_json)
            
        elif response.status_code == 503:
            return "⏳_LOADING_" # إشارة بأن النموذج يستيقظ
            
        elif response.status_code == 401:
            return "❌_TOKEN_ERROR_" # مشكلة في التوكن
            
        else:
            return f"❌ خطأ {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"❌ فشل الاتصال: {str(e)}"

def clean_output(text):
    """تنظيف النص المستخرج لضمان ظهور التلميح فقط"""
    # إزالة أي كلمات توضيحية قد يضيفها النموذج
    text = re.sub(r'^(التلميح:|تلميح:|Clue:)', '', text, flags=re.IGNORECASE).strip()
    return text

# --- 4. إعدادات الواجهة الجانبية ---
st.sidebar.header("الإعدادات ⚙️")
temp = st.sidebar.slider("درجة الإبداع (Temperature)", 0.1, 1.0, 0.4)
st.sidebar.info("درجة منخفضة تعني إجابات دقيقة ومباشرة.")

# --- 5. تبويبات التطبيق ---
tab1, tab2 = st.tabs(["📝 توليد يدوي", "📁 معالجة ملف CSV"])

with tab1:
    st.subheader("توليد تلميح لنص واحد")
    c1, c2 = st.columns(2)
    with c1:
        u_text = st.text_area("النص السياقي", "الجزائر هي أكبر دولة في أفريقيا من حيث المساحة.")
    with c2:
        u_kw = st.text_input("الكلمة المستهدفة", "الجزائر")
        u_cat = st.text_input("التصنيف (اختياري)", "جغرافيا")

    if st.button("توليد التلميح الآن 🪄"):
        if u_text and u_kw:
            # صياغة الأمر للموديل
            prompt = f"<|im_start|>user\nأعطني تلميحاً واحداً فقط وبسيطاً للغز كلمات متقاطعة للكلمة التالية: '{u_kw}'.\nاستعن بهذا النص: '{u_text}'.\nالتصنيف: '{u_cat}'.\nاجعل التلميح قصيراً ومباشراً.<|im_end|>\n<|im_start|>assistant\nالتلميح:"
            
            with st.spinner("جاري التوليد..."):
                result = get_ai_response(prompt, temp)
                
                if result == "⏳_LOADING_":
                    st.warning("⏳ النموذج قيد التحميل على خوادم Hugging Face.. انتظر 10 ثواني واضغط الزر مجدداً.")
                elif result == "❌_TOKEN_ERROR_":
                    st.error("❌ مشكلة في التوكن! تأكد من صحة HF_TOKEN في إعدادات Secrets.")
                elif result.startswith("❌"):
                    st.error(result)
                else:
                    st.success("تم التوليد بنجاح!")
                    st.subheader("التلميح المقترح:")
                    st.code(clean_output(result))
        else:
            st.warning("يرجى إدخال النص والكلمة أولاً.")

with tab2:
    st.subheader("معالجة بيانات من ملف CSV")
    file = st.file_uploader("ارفع الملف (يجب أن يحتوي على أعمدة: text, keyword, category)", type=['csv'])
    
    if file:
        df = pd.read_csv(file)
        st.write("معاينة البيانات:")
        st.dataframe(df.head())
        
        if st.button("ابدأ المعالجة الجماعية 🚀"):
            results = []
            progress = st.progress(0)
            
            for i, row in df.iterrows():
                p = f"<|im_start|>user\nأعطني تلميحاً للكلمة: '{row['keyword']}' بناءً على: '{row['text']}'.<|im_end|>\n<|im_start|>assistant\nالتلميح:"
                res = get_ai_response(p, temp)
                results.append(clean_output(res))
                progress.progress((i + 1) / len(df))
                time.sleep(0.5) # تجنب تجاوز حدود الـ API
                
            df['Generated Clue'] = results
            st.success("اكتملت المعالجة!")
            st.dataframe(df)
            
            # زر التحميل
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("تحميل النتائج 📥", csv, "results.csv", "text/csv")
