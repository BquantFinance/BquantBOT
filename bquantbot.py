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

# API Key hardcodeada (CAMBIAR EN PRODUCCIÓN)
GEMINI_API_KEY = "AIzaSyBuxu0jsV6t0hVBVmksD6LBJhKPu8VjPOY"

# ============================================
# ESTILOS CSS PREMIUM
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp {
        background: #050508;
        background-image: 
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at center, rgba(6, 182, 212, 0.05) 0%, transparent 70%);
        min-height: 100vh;
    }
    
    .logo-container {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    
    .logo {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        color: rgba(255, 255, 255, 0.5);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .status-container {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 1rem;
        flex-wrap: wrap;
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
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(0.95); }
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 2rem 0;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(99, 102, 241, 0.2);
        transform: translateY(-4px);
    }
    
    .stat-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.7) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.75rem;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .suggestions-title {
        text-align: center;
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    
    .message-row {
        display: flex;
        margin: 1rem 0;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .message-row.user { justify-content: flex-end; }
    .message-row.assistant { justify-content: flex-start; }
    
    .message-bubble {
        max-width: 80%;
        padding: 1rem 1.25rem;
        border-radius: 20px;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    .message-bubble.user {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-bottom-right-radius: 6px;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    
    .message-bubble.assistant {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.9);
        border-bottom-left-radius: 6px;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        margin: 0 0.75rem;
        flex-shrink: 0;
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        order: 1;
    }
    
    .avatar.assistant {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .source-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        margin-top: 0.5rem;
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
        color: white;
    }
    
    .stChatInput > div {
        max-width: 900px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
    }
    
    .stChatInput input {
        color: white !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 5rem 0;
        color: rgba(255, 255, 255, 0.25);
        font-size: 0.75rem;
    }
    
    .footer a {
        color: rgba(99, 102, 241, 0.6);
        text-decoration: none;
    }
    
    @media (max-width: 768px) {
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
        .logo { font-size: 2.5rem; }
        .message-bubble { max-width: 90%; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR DATOS
# ============================================
@st.cache_data
def load_berkshire_letters():
    try:
        with open("data/berkshire_letters.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def init_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-1.5-flash')

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
        text_lower = text.lower()
        
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
    <div class="logo">⚡ BQuant</div>
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
# MENSAJES
# ============================================
for msg in st.session_state.messages:
    role_class = msg["role"]
    avatar = "👤" if role_class == "user" else "⚡"
    
    content = msg["content"]
    sources_html = ""
    if role_class == "assistant" and "sources" in msg and msg["sources"]:
        sources_html = f'<div class="source-tag">📚 Cartas: {", ".join(msg["sources"])}</div>'
    
    st.markdown(f"""
    <div class="message-row {role_class}">
        <div class="avatar {role_class}">{avatar}</div>
        <div class="message-bubble {role_class}">
            {content}
            {sources_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# INPUT
# ============================================
if letters:
    if prompt := st.chat_input("Pregunta sobre las cartas de Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        st.markdown(f"""
        <div class="message-row user">
            <div class="avatar user">👤</div>
            <div class="message-bubble user">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Buscando en las cartas..."):
            response, sources = get_response(prompt, letters, model, st.session_state.messages)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })
        st.rerun()
else:
    st.error("⚠️ No se encontró `data/berkshire_letters.json`. Coloca el archivo en la carpeta `data/`.")

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>Datos extraídos de las cartas oficiales de Berkshire Hathaway</p>
    <p style="margin-top: 0.5rem;">
        © 2026 <a href="https://bquantfinance.com" target="_blank">BQuant Finance</a> · 
        <a href="https://twitter.com/Gsnchez" target="_blank">@Gsnchez</a>
    </p>
</div>
""", unsafe_allow_html=True)
