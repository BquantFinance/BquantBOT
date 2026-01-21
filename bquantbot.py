import streamlit as st
from groq import Groq
import json
import re

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="BQuant", page_icon="⚡", layout="centered")

GROQ_API_KEY = "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"

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    .stApp { background: #0a0a0f; }
    .block-container { padding: 1rem !important; max-width: 800px !important; }
    
    .header { text-align: center; padding: 1rem 0; }
    .logo { font-size: 2rem; font-weight: 700; color: #818cf8; }
    .tagline { color: #555; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; }
    
    .stButton > button {
        background: #1a1a2e !important;
        border: 1px solid #333 !important;
        color: #aaa !important;
        border-radius: 20px !important;
    }
    .stButton > button:hover {
        background: #2a2a4e !important;
        color: white !important;
    }
    
    .stChatMessage { background: #111 !important; border-radius: 10px !important; }
    [data-testid="stChatInput"] > div { background: #151520 !important; border: 1px solid #333 !important; border-radius: 15px !important; }
    [data-testid="stChatInput"] input { color: white !important; }
    [data-testid="stChatInput"] button { background: #6366f1 !important; border-radius: 10px !important; }
    
    .source-tag { background: #1a1a2e; color: #818cf8; padding: 4px 8px; border-radius: 5px; font-size: 0.7rem; margin-top: 8px; display: inline-block; }
    .footer { text-align: center; color: #444; font-size: 0.65rem; padding: 1rem; }
    .footer a { color: #666; }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA
# ============================================
@st.cache_data
def load_letters():
    try:
        with open("berkshire_letters.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)

# ============================================
# SEARCH
# ============================================
def search(query, letters):
    q = query.lower()
    stopwords = {'que','el','la','los','de','en','a','por','para','con','sobre','es','buffett','warren','carta','dice','qué','cómo','hola'}
    keywords = [w for w in re.findall(r'\w+', q) if w not in stopwords and len(w) > 2]
    
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    target_year = year_match.group(1) if year_match else None
    
    results = []
    for year, data in letters.items():
        if target_year and year != target_year:
            continue
        for para in data.get('text', '').split('\n\n'):
            if len(para) < 100:
                continue
            score = sum(para.lower().count(k) for k in keywords)
            if score > 0 or target_year:
                results.append({'year': year, 'text': para[:1500], 'score': score or 0.1})
    
    if not results:
        for y in sorted(letters.keys(), reverse=True)[:2]:
            for p in letters[y].get('text', '').split('\n\n')[:2]:
                if len(p) > 150:
                    results.append({'year': y, 'text': p[:1500], 'score': 0})
    
    return sorted(results, key=lambda x: -x['score'])[:5]

# ============================================
# GENERATE
# ============================================
def generate(query, letters, client):
    chunks = search(query, letters)
    context = "\n\n".join([f"[{c['year']}]: {c['text']}" for c in chunks])
    sources = list(dict.fromkeys([c['year'] for c in chunks]))
    
    messages = [
        {
            "role": "system",
            "content": """Eres un asistente experto en las cartas anuales de Warren Buffett (1977-2024).
            
REGLAS:
- Responde basándote SOLO en el contexto proporcionado
- Cita el año cuando menciones algo específico: "En [año], Buffett..."
- Responde en español
- Sé conciso (150-200 palabras)
- No inventes información que no esté en el contexto"""
        },
        {
            "role": "user", 
            "content": f"""CONTEXTO DE LAS CARTAS:
{context}

PREGUNTA: {query}"""
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content, sources
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# APP
# ============================================
letters = load_letters()
client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown("""
<div class="header">
    <div class="logo">⚡ BQuant</div>
    <div class="tagline">Berkshire Letters AI</div>
</div>
""", unsafe_allow_html=True)

# Welcome
if not st.session_state.messages:
    st.markdown("<p style='text-align:center; color:#666; margin: 1rem 0;'>Pregunta sobre 47 años de cartas de Buffett</p>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    suggestions = ["Filosofía inversión", "Crisis 2008", "Inflación", "Carta 2023", "Buena empresa", "El oro"]
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": f"¿Qué dice Buffett sobre {s.lower()}?"})
                st.rerun()

# Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.write(msg["content"])
        if msg.get("sources"):
            st.markdown(f'<div class="source-tag">📚 {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# Input
if letters:
    if prompt := st.chat_input("Pregunta sobre Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Pensando..."):
                response, sources = generate(prompt, letters, client)
            st.write(response)
            if sources:
                st.markdown(f'<div class="source-tag">📚 {", ".join(sources)}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
else:
    st.error("No se encontró berkshire_letters.json")

# Footer
st.markdown('<div class="footer">No es asesoramiento financiero · <a href="https://bquantfinance.com">BQuant</a> · <a href="https://twitter.com/Gsnchez">@Gsnchez</a></div>', unsafe_allow_html=True)
