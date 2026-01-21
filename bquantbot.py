import streamlit as st
from google import genai
from google.genai import types
import json
import re

# ============================================
# CONFIGURACIÓN
# ============================================
st.set_page_config(
    page_title="BQuant Chatbot",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

GEMINI_API_KEY = "AIzaSyBuxu0jsV6t0hVBVmksD6LBJhKPu8VjPOY"

# ============================================
# ESTILOS CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    #MainMenu, footer, header, .stDeployButton {display: none !important;}
    
    .stApp {
        background: #08080c;
        background-image: radial-gradient(ellipse at top, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
    }
    
    .block-container { padding: 1rem !important; max-width: 800px !important; }
    
    .header { text-align: center; padding: 0.75rem 0; }
    .logo {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tagline { color: rgba(255,255,255,0.35); font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; }
    
    .badges { display: flex; justify-content: center; gap: 0.4rem; margin-top: 0.5rem; }
    .badge { padding: 3px 10px; border-radius: 15px; font-size: 0.65rem; }
    .badge-online { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2); color: #22c55e; }
    .badge-data { background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.2); color: #818cf8; }
    
    .welcome { text-align: center; padding: 1rem 0; }
    .welcome-title { font-size: 1.3rem; font-weight: 600; color: white; }
    .welcome-sub { color: rgba(255,255,255,0.4); font-size: 0.85rem; }
    
    .stButton > button {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.7) !important;
        border-radius: 20px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.8rem !important;
    }
    .stButton > button:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.4) !important;
        color: white !important;
    }
    
    .stChatMessage {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: rgba(20,20,30,0.95) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] input { color: white !important; }
    [data-testid="stChatInput"] button { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; border-radius: 12px !important; }
    
    .source-tag {
        display: inline-block;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        color: #a5b4fc;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
    }
    
    .footer { text-align: center; padding: 0.75rem; color: rgba(255,255,255,0.2); font-size: 0.6rem; }
    .footer a { color: rgba(99,102,241,0.5); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR DATOS
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
    return genai.Client(api_key=GEMINI_API_KEY)

# ============================================
# BÚSQUEDA
# ============================================
def search(query, letters, n=5):
    q = query.lower()
    stop = {'que','qué','como','cómo','el','la','los','las','de','en','a','por','para','con','sobre',
            'es','son','the','of','in','to','for','buffett','warren','berkshire','carta','dice','año'}
    
    kw = [w for w in re.findall(r'\w+', q) if w not in stop and len(w) > 2]
    year = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    year = year.group(1) if year else None
    
    results = []
    for y, data in letters.items():
        if year and y != year:
            continue
        text = data.get('text', '')
        for p in text.split('\n\n'):
            if len(p) < 100:
                continue
            score = sum(p.lower().count(k) for k in kw)
            if score > 0 or year:
                results.append({'year': y, 'text': p[:1500], 'score': score or 0.1})
    
    if not results:
        for y in sorted(letters.keys(), reverse=True)[:2]:
            text = letters[y].get('text', '')
            for p in text.split('\n\n')[:2]:
                if len(p) > 150:
                    results.append({'year': y, 'text': p[:1500], 'score': 0})
    
    return sorted(results, key=lambda x: -x['score'])[:n]

# ============================================
# RESPUESTA
# ============================================
def ask(query, letters, client):
    chunks = search(query, letters)
    context = "\n\n".join([f"[{c['year']}]: {c['text']}" for c in chunks])
    sources = list(dict.fromkeys([c['year'] for c in chunks]))
    
    prompt = f"""Eres un asistente experto en las cartas de Warren Buffett (1977-2024).

CONTEXTO:
{context}

REGLAS:
- Usa los datos del contexto
- Cita el año cuando sea relevante
- Responde en español
- Sé conciso (150-250 palabras)

PREGUNTA: {query}

RESPUESTA:"""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )
        return response.text, sources
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# APP
# ============================================
letters = load_letters()
client = get_client()

if "msgs" not in st.session_state:
    st.session_state.msgs = []

# Header
st.markdown("""
<div class="header">
    <div class="logo">⚡ BQuant</div>
    <div class="tagline">Berkshire Letters AI</div>
    <div class="badges">
        <span class="badge badge-online">● Online</span>
        <span class="badge badge-data">📚 48 cartas</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome
if not st.session_state.msgs:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">¿Qué quieres saber?</div>
        <div class="welcome-sub">47 años de sabiduría de Buffett</div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, s in enumerate(["Filosofía inversión", "Crisis 2008", "Inflación", "Carta 2023", "Buena empresa", "El oro"]):
        with cols[i % 3]:
            if st.button(s, key=f"s{i}", use_container_width=True):
                st.session_state.msgs.append({"role": "user", "content": f"¿Qué dice Buffett sobre {s.lower()}?"})
                st.rerun()

# Messages
for m in st.session_state.msgs:
    with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "⚡"):
        st.write(m["content"])
        if m.get("sources"):
            st.markdown(f'<div class="source-tag">📚 {", ".join(m["sources"])}</div>', unsafe_allow_html=True)

# Input
if letters and client:
    if prompt := st.chat_input("Pregunta sobre Buffett..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Pensando..."):
                resp, src = ask(prompt, letters, client)
            st.write(resp)
            if src:
                st.markdown(f'<div class="source-tag">📚 {", ".join(src)}</div>', unsafe_allow_html=True)
        
        st.session_state.msgs.append({"role": "assistant", "content": resp, "sources": src})

# Footer
st.markdown('<div class="footer">No es asesoramiento financiero · <a href="https://bquantfinance.com">BQuant</a> · <a href="https://twitter.com/Gsnchez">@Gsnchez</a></div>', unsafe_allow_html=True)
