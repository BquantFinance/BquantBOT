import streamlit as st
import google.generativeai as genai
import json
import re

# ============================================
# CONFIGURACIÓN
# ============================================
st.set_page_config(
    page_title="BQuant Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Key
GEMINI_API_KEY = "AIzaSyBuxu0jsV6t0hVBVmksD6LBJhKPu8VjPOY"

# ============================================
# ESTILOS CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: #08080c;
        background-image: 
            radial-gradient(ellipse at top, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse at bottom, rgba(139, 92, 246, 0.08) 0%, transparent 50%);
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 180px !important;
        max-width: 850px !important;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    
    .logo {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.8rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
    }
    
    .badge-online {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #22c55e;
    }
    
    .badge-data {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #818cf8;
    }
    
    .pulse-dot {
        width: 6px;
        height: 6px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* Welcome section */
    .welcome {
        text-align: center;
        padding: 3rem 0;
    }
    
    .welcome-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .welcome-sub {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1rem;
        margin-bottom: 2.5rem;
    }
    
    /* Suggestion chips */
    .suggestions {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.6rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Streamlit button override for suggestions */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 25px !important;
        padding: 0.6rem 1.2rem !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        transition: all 0.25s ease !important;
        white-space: nowrap !important;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stChatMessageContent"] {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    
    /* Chat input - GRANDE Y CENTRADO */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 750px !important;
        padding: 1.5rem 2rem 2rem 2rem !important;
        background: linear-gradient(to top, #08080c 60%, transparent) !important;
        z-index: 999 !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: rgba(20, 20, 30, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 20px !important;
        padding: 0.4rem !important;
        box-shadow: 
            0 0 40px rgba(99, 102, 241, 0.1),
            0 4px 20px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(20px) !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 
            0 0 60px rgba(99, 102, 241, 0.15),
            0 4px 30px rgba(0, 0, 0, 0.4) !important;
    }
    
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: white !important;
        font-size: 1.05rem !important;
        padding: 1rem 1.2rem !important;
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.35) !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin: 0.3rem !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Source tag */
    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        margin-top: 0.75rem;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 90px;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
        color: rgba(255, 255, 255, 0.2);
        font-size: 0.65rem;
        z-index: 998;
    }
    
    .footer a {
        color: rgba(99, 102, 241, 0.5);
        text-decoration: none;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #6366f1 transparent transparent transparent !important;
    }
    
    @media (max-width: 768px) {
        .logo { font-size: 2rem; }
        .welcome-title { font-size: 1.4rem; }
        [data-testid="stChatInput"] {
            padding: 1rem !important;
            max-width: 95% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR DATOS Y MODELO
# ============================================
@st.cache_data
def load_berkshire_letters():
    try:
        with open("berkshire_letters.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def init_gemini():
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    return None

# ============================================
# BÚSQUEDA
# ============================================
def search_letters(query: str, letters: dict, max_chunks: int = 5, chunk_size: int = 2000) -> list:
    query_lower = query.lower()
    
    stopwords = {'que', 'qué', 'como', 'cómo', 'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 
                 'a', 'por', 'para', 'con', 'sobre', 'es', 'son', 'fue', 'were', 'was', 'the', 'a', 'an',
                 'of', 'in', 'to', 'for', 'on', 'with', 'about', 'is', 'are', 'what', 'how', 'when', 'why',
                 'buffett', 'warren', 'berkshire', 'piensa', 'dijo', 'dice', 'said', 'think', 'thinks',
                 'carta', 'letter', 'año', 'year', 'resume', 'resumen', 'summarize'}
    
    keywords = [w for w in re.findall(r'\w+', query_lower) if w not in stopwords and len(w) > 2]
    
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    specific_year = year_match.group(1) if year_match else None
    
    results = []
    
    for year, letter_data in letters.items():
        if specific_year and year != specific_year:
            continue
            
        text = letter_data.get('text', '')
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        
        for para in paragraphs:
            para_lower = para.lower()
            score = 0
            matched_keywords = []
            
            for kw in keywords:
                count = para_lower.count(kw)
                if count > 0:
                    score += count
                    matched_keywords.append(kw)
            
            if score > 0 or specific_year:
                results.append({
                    'year': year,
                    'text': para[:chunk_size],
                    'score': score if score > 0 else 0.1,
                })
    
    if not results:
        recent_years = sorted(letters.keys(), reverse=True)[:3]
        for year in recent_years:
            text = letters[year].get('text', '')
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 200]
            for para in paragraphs[:2]:
                results.append({'year': year, 'text': para[:chunk_size], 'score': 0})
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_chunks]

# ============================================
# RESPUESTA
# ============================================
def get_response(query: str, letters: dict, model, messages_history: list) -> tuple:
    relevant_chunks = search_letters(query, letters)
    
    context = ""
    sources = []
    
    if relevant_chunks:
        context = "FRAGMENTOS DE LAS CARTAS DE WARREN BUFFETT:\n\n"
        for chunk in relevant_chunks:
            context += f"[Carta {chunk['year']}]:\n{chunk['text']}\n\n---\n\n"
            if chunk['year'] not in sources:
                sources.append(chunk['year'])
    
    system = """Eres el asistente de BQuant Finance especializado en las cartas anuales de Warren Buffett (1977-2024).

INSTRUCCIONES:
- Responde basándote en los fragmentos proporcionados
- Cita el año cuando sea relevante: "En su carta de [año]..."
- Si no hay fragmentos relevantes, usa conocimiento general pero acláralo
- Sé conciso (150-300 palabras)
- Responde en el idioma del usuario
- No inventes citas textuales

ESTILO: Profesional, directo, con ejemplos concretos."""

    full_prompt = system + "\n\n" + context
    full_prompt += "\nCONVERSACIÓN:\n"
    for msg in messages_history[-4:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        full_prompt += f"{role}: {msg['content']}\n\n"
    full_prompt += f"Usuario: {query}\n\nAsistente:"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text, sources
    except Exception as e:
        return f"Error: {str(e)}", []

# ============================================
# INIT
# ============================================
letters = load_berkshire_letters()
model = init_gemini()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="header">
    <div class="logo">⚡ BQuantChatBot</div>
    <div class="tagline">Berkshire Letters AI</div>
    <div class="badges">
        <span class="badge badge-online"><span class="pulse-dot"></span> Online</span>
        <span class="badge badge-data">📚 48 cartas · 1977-2024</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# WELCOME + SUGGESTIONS (solo si no hay mensajes)
# ============================================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">¿Qué quieres saber de Buffett?</div>
        <div class="welcome-sub">Pregunta lo que quieras sobre 47 años de sabiduría inversora</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sugerencias como botones
    suggestions = [
        "¿Qué piensa sobre la inflación?",
        "¿Cuál es su filosofía de inversión?",
        "¿Qué dijo sobre la crisis de 2008?",
        "Resume la carta de 2023",
        "¿Qué opina del oro?",
        "¿Cómo define una buena empresa?",
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()

# ============================================
# MENSAJES
# ============================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# ============================================
# INPUT
# ============================================
if letters and model:
    if prompt := st.chat_input("Escribe tu pregunta sobre las cartas de Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Buscando..."):
                response, sources = get_response(prompt, letters, model, st.session_state.messages)
            st.write(response)
            if sources:
                st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(sources)}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
elif not letters:
    st.error("⚠️ No se encontró berkshire_letters.json")
else:
    st.error("⚠️ API key no configurada")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <a href="https://bquantfinance.com">BQuant Finance</a> · 
    <a href="https://twitter.com/Gsnchez">@Gsnchez</a>
</div>
""", unsafe_allow_html=True)
