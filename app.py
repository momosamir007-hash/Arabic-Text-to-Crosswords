import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re

# إعداد الصفحة
st.set_page_config(page_title="مُولّد تلميحات الكلمات المتقاطعة", page_icon="🧩", layout="wide")

st.title("🧩 مُولّد تلميحات الكلمات المتقاطعة باللغة العربية")
st.markdown("""
هذا التطبيق يقوم بتوليد تلميحات لألعاب الكلمات المتقاطعة باللغة العربية بناءً على نصوص، كلمات مفتاحية، وتصنيفات محددة، باستخدام نموذج الذكاء الاصطناعي `Llama3-8B-Ar-Text-to-Cross`.
""")

# --- إعداد النموذج ---
@st.cache_resource
def load_model():
    model_name = "Kamyar-zeinalipour/Llama3-8B-Ar-Text-to-Cross"
    
    # التحقق من توفر معالج رسوميات (GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # استخدام torch_dtype=torch.float16 لتقليل استهلاك الذاكرة إذا كان هناك GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    return tokenizer, model, device

with st.spinner("جاري تحميل نموذج الذكاء الاصطناعي... (قد يستغرق بعض الوقت في المرة الأولى)"):
    tokenizer, model, device = load_model()


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

def extract_text(text):
    try:
        if text.count('<|end_header_id|>\n\n') > 1:
            response_part = text.split('<|end_header_id|>\n\n')[2]
            assistant_response = response_part.split('<|end_of_text|>')[0]
            assistant_response = assistant_response.replace('<|eot_id|><|start_header_id|>assistant', '')
            return assistant_response.strip()
    except IndexError:
        pass
    return None

def get_first_three_clues(generated_text):
    if not generated_text: return "لم يتم توليد تلميحات."
    clues = re.findall(r'(CLUE\d+:.*)', generated_text)
    first_three_clues = clues[:3]
    return '\n'.join(first_three_clues) if first_three_clues else generated_text

def get_code_completion(prompt, temperature):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def process_single_entry(text, keyword, category, temperature):
    prompt = format_row(text, keyword, category)
    response = get_code_completion(prompt, temperature)
    generated_text = extract_text(response)
    return get_first_three_clues(generated_text)

# --- إعدادات الواجهة الجانبية ---
st.sidebar.header("إعدادات التوليد ⚙️")
temperature = st.sidebar.slider("درجة الحرارة (Temperature)", min_value=0.1, max_value=1.0, value=0.1, step=0.1, help="قيمة أقل تعني إجابات أكثر دقة، وقيمة أعلى تعني إجابات أكثر إبداعاً وتنوعاً.")

# --- تقسيم الواجهة إلى قسمين (Tabs) ---
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
            with st.spinner("جاري المعالجة..."):
                result = process_single_entry(input_text, input_keyword, input_category, temperature)
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
                
                try:
                    result = process_single_entry(row['text'], row['keyword'], row['category'], temperature)
                    outputs.append({
                        'text': row['text'],
                        'keyword': row['keyword'],
                        'category': row['category'],
                        'Generated Arabic Crossword Clue': result
                    })
                except Exception as e:
                    outputs.append({
                        'text': row['text'],
                        'keyword': row['keyword'],
                        'category': row['category'],
                        'Generated Arabic Crossword Clue': f"خطأ: {str(e)}"
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
