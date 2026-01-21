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
# ESTILOS CSS PREMIUM
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
        background: #050508;
        background-image: 
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 100px !important;
        max-width: 900px !important;
    }
    
    /* Header */
    .logo-container {
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .logo {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    
    .status-container {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.8rem;
        color: #22c55e;
    }
    
    .data-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.8rem;
        color: #818cf8;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }
    
    .stat-icon { font-size: 1.25rem; margin-bottom: 0.25rem; }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Suggestions */
    .suggestions-title {
        text-align: center;
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: rgba(99, 102, 241, 0.2) !important;
    }
    
    .stChatMessage p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Chat input */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 0.25rem !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: rgba(99, 102, 241, 0.5) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15) !important;
    }
    
    .stChatInput input {
        color: white !important;
        font-size: 0.95rem !important;
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.3) !important;
    }
    
    .stChatInput button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border-radius: 12px !important;
    }
    
    /* Source tag */
    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255, 255, 255, 0.25);
        font-size: 0.7rem;
    }
    
    .footer a {
        color: rgba(99, 102, 241, 0.6);
        text-decoration: none;
    }
    
    @media (max-width: 768px) {
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
        .logo { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR DATOS
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
# BÚSQUEDA EN LAS CARTAS
# ============================================
def search_letters(query: str, letters: dict, max_chunks: int = 5, chunk_size: int = 2000) -> list:
    query_lower = query.lower()
    
    stopwords = {'que', 'qué', 'como', 'cómo', 'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 
                 'a', 'por', 'para', 'con', 'sobre', 'es', 'son', 'fue', 'were', 'was', 'the', 'a', 'an',
                 'of', 'in', 'to', 'for', 'on', 'with', 'about', 'is', 'are', 'what', 'how', 'when', 'why',
                 'buffett', 'warren', 'berkshire', 'piensa', 'dijo', 'dice', 'said', 'think', 'thinks',
                 'carta', 'letter', 'año', 'year'}
    
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
                    'keywords': matched_keywords
                })
    
    if not results and not specific_year:
        recent_years = sorted(letters.keys(), reverse=True)[:3]
        for year in recent_years:
            text = letters[year].get('text', '')
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 200]
            for para in paragraphs[:2]:
                results.append({
                    'year': year,
                    'text': para[:chunk_size],
                    'score': 0,
                    'keywords': []
                })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:max_chunks]

# ============================================
# GENERAR RESPUESTA
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
- Sé conciso (150-250 palabras)
- Responde en el idioma del usuario
- No inventes citas textuales

ESTILO: Profesional, directo, con ejemplos concretos."""

    full_prompt = system + "\n\n"
    
    if context:
        full_prompt += context
    
    full_prompt += "CONVERSACIÓN RECIENTE:\n"
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
# INICIALIZAR
# ============================================
letters = load_berkshire_letters()
model = init_gemini()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="logo-container">
    <div class="logo">⚡ BQuantChatBot</div>
    <div class="tagline">Berkshire Letters AI</div>
    <div class="status-container">
        <div class="status-pill">
            <div class="status-dot"></div>
            Online
        </div>
        <div class="data-pill">
            📚 48 cartas · 1977-2024
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PANTALLA INICIAL
# ============================================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon">📜</div>
            <div class="stat-value">48</div>
            <div class="stat-label">Cartas</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📅</div>
            <div class="stat-value">47</div>
            <div class="stat-label">Años</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📝</div>
            <div class="stat-value">3.4M</div>
            <div class="stat-label">Caracteres</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">RAG</div>
            <div class="stat-label">Búsqueda</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="suggestions-title">Prueba a preguntar</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    suggestions = [
        "¿Qué piensa Buffett sobre la inflación?",
        "¿Cuál es su filosofía de inversión?",
        "¿Qué dijo sobre la crisis de 2008?",
        "Resume la carta de 2023"
    ]
    
    for col, text in zip([col1, col2, col3, col4], suggestions):
        with col:
            if st.button(text, key=text, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": text})
                st.rerun()

# ============================================
# MENSAJES (usando st.chat_message nativo)
# ============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# ============================================
# INPUT
# ============================================
if letters and model:
    if prompt := st.chat_input("Pregunta sobre las cartas de Warren Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Buscando en las cartas..."):
                response, sources = get_response(prompt, letters, model, st.session_state.messages)
            st.write(response)
            if sources:
                st.markdown(f'<div class="source-tag">📚 Cartas: {", ".join(sources)}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })
elif not letters:
    st.error("⚠️ No se encontró `berkshire_letters.json`")
else:
    st.error("⚠️ Configura GEMINI_API_KEY en los secrets")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>Datos de las cartas oficiales de Berkshire Hathaway · No es asesoramiento financiero</p>
    <p style="margin-top: 0.5rem;">
        © 2026 <a href="https://bquantfinance.com" target="_blank">BQuant Finance</a> · 
        <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a>
    </p>
</div>
""", unsafe_allow_html=True)
