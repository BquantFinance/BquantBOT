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
# CSS PREMIUM
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    .stApp {
        background: #050508;
        background-image: 
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.18) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at center, rgba(6, 182, 212, 0.06) 0%, transparent 70%);
        min-height: 100vh;
    }
    
    .block-container {
        padding: 2rem 1.5rem !important;
        max-width: 850px !important;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 1.5rem 0 2rem 0;
    }
    
    .logo {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-online {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
    }
    
    .badge-data {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 12px #22c55e;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }
    
    /* Welcome */
    .welcome {
        text-align: center;
        padding: 2.5rem 0;
    }
    
    .welcome-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .welcome-sub {
        color: rgba(255, 255, 255, 0.45);
        font-size: 1rem;
    }
    
    /* Suggestion buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.75) !important;
        border-radius: 14px !important;
        padding: 0.8rem 1.3rem !important;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 18px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stChatMessageContent"] {
        color: rgba(255, 255, 255, 0.92) !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
    }
    
    [data-testid="stChatMessageContent"] p {
        color: rgba(255, 255, 255, 0.92) !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] > div {
        background: rgba(12, 12, 20, 0.95) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 18px !important;
        box-shadow: 
            0 0 50px rgba(99, 102, 241, 0.1),
            0 8px 30px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 
            0 0 70px rgba(99, 102, 241, 0.15),
            0 8px 40px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stChatInput"] input {
        color: white !important;
        font-size: 1rem !important;
        padding: 0.2rem 0 !important;
    }
    
    [data-testid="stChatInput"] input::placeholder {
        color: rgba(255, 255, 255, 0.35) !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Source tag */
    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.78rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: rgba(255, 255, 255, 0.25);
        font-size: 0.75rem;
    }
    
    .footer a {
        color: rgba(99, 102, 241, 0.6);
        text-decoration: none;
        transition: color 0.2s;
    }
    
    .footer a:hover {
        color: rgba(99, 102, 241, 1);
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .logo { font-size: 2.3rem; }
        .welcome-title { font-size: 1.3rem; }
        .block-container { padding: 1rem !important; }
    }
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
    stopwords = {'que','el','la','los','de','en','a','por','para','con','sobre','es','buffett','warren','carta','dice','qué','cómo','hola','hello','hi'}
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
- Sé conciso pero completo (150-250 palabras)
- No inventes información que no esté en el contexto
- Sé directo y profesional"""
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
            max_tokens=600,
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
    <div class="badges">
        <span class="badge badge-online"><span class="pulse-dot"></span> Online</span>
        <span class="badge badge-data">📚 48 cartas · 1977-2024</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">¿Qué quieres saber de Buffett?</div>
        <div class="welcome-sub">47 años de sabiduría inversora a tu alcance</div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    suggestions = ["Filosofía de inversión", "Crisis de 2008", "Sobre la inflación", "Carta de 2023", "Buena empresa", "Opinión del oro"]
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
            st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# Input
if letters:
    if prompt := st.chat_input("Pregunta sobre las cartas de Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Buscando en las cartas..."):
                response, sources = generate(prompt, letters, client)
            st.write(response)
            if sources:
                st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(sources)}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
else:
    st.error("No se encontró berkshire_letters.json")

# Footer
st.markdown("""
<div class="footer">
    <a href="https://bquantfinance.com">BQuant Finance</a> · <a href="https://twitter.com/Gsnchez">@Gsnchez</a>
</div>
""", unsafe_allow_html=True)
