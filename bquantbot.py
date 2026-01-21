import streamlit as st
from groq import Groq
import json
import re

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="BQuant", page_icon="⚡", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# ============================================
# CSS PREMIUM
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Space Grotesk', sans-serif; }
    
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
    section[data-testid="stSidebar"], .main, .main .block-container {
        background: #050508 !important;
        background-image: 
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.18) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(139, 92, 246, 0.12) 0%, transparent 50%) !important;
    }
    
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }
    
    .block-container {
        padding: 1.5rem 1.5rem 0 1.5rem !important;
        max-width: 850px !important;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
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
        font-size: 0.75rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.4rem;
    }
    
    .badges {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 50px;
        font-size: 0.72rem;
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
        width: 7px;
        height: 7px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 10px #22c55e;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }
    
    /* Welcome */
    .welcome {
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.4rem;
    }
    
    .welcome-sub {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.95rem;
    }
    
    /* Suggestion buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.75) !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.2rem !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 1rem 1.2rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    [data-testid="stChatMessageContent"] {
        color: rgba(255, 255, 255, 0.92) !important;
        font-size: 0.92rem !important;
        line-height: 1.65 !important;
    }
    
    [data-testid="stChatMessageContent"] p {
        color: rgba(255, 255, 255, 0.92) !important;
    }
    
    /* Chat input */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    .stBottom,
    .stChatInput,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: #0a0a12 !important;
        border: 1.5px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 20px !important;
        padding: 0.4rem !important;
        box-shadow: 0 0 50px rgba(99, 102, 241, 0.12) !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: rgba(99, 102, 241, 0.7) !important;
        box-shadow: 0 0 70px rgba(99, 102, 241, 0.18) !important;
    }
    
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: white !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.1rem !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    
    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
        margin: 0.35rem !important;
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
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-top: 0.75rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem 0;
        color: rgba(255, 255, 255, 0.25);
        font-size: 0.72rem;
    }
    
    .footer a {
        color: rgba(99, 102, 241, 0.6);
        text-decoration: none;
    }
    
    .footer a:hover {
        color: rgba(99, 102, 241, 1);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .logo { font-size: 2.2rem; }
        .welcome-title { font-size: 1.25rem; }
        .block-container { padding: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA & CLIENT
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
# TRANSLATE QUERY TO ENGLISH
# ============================================
def translate_to_english(query, client):
    """Traduce la query a inglés para buscar en las cartas"""
    
    # Detectar si es un saludo simple
    greetings = ['hola', 'hello', 'hi', 'buenas', 'buenos dias', 'buenas tardes', 'hey']
    if query.lower().strip() in greetings or len(query.split()) <= 2 and not any(c.isdigit() for c in query):
        # Verificar si tiene contenido sustancial
        if not any(word in query.lower() for word in ['qué', 'cómo', 'cuál', 'cuándo', 'dónde', 'por qué', 'what', 'how', 'when', 'why']):
            return None  # Es solo un saludo
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Extract search keywords in English from the user's question about Warren Buffett's letters.
Return ONLY the English keywords, nothing else. No explanations.

Examples:
- "¿Qué piensa Buffett sobre la inflación?" → "inflation"
- "¿Qué dijo sobre la crisis de 2008?" → "crisis 2008 financial crash"
- "¿Cuál es su filosofía de inversión?" → "investment philosophy approach"
- "Resume la carta de 1983" → "1983"
- "¿Qué opina del oro?" → "gold"
- "¿Cómo evalúa una empresa?" → "evaluate company business value"
- "Háblame de los seguros" → "insurance"
- "¿Qué son los moats?" → "moat competitive advantage"
"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=50,
        )
        keywords = response.choices[0].message.content.strip()
        return keywords if keywords else None
    except:
        return None

# ============================================
# SEARCH
# ============================================
def chunk_text(text, chunk_size=1500):
    """Divide texto en chunks manejables"""
    # Primero intentar dividir por doble salto de línea
    if text.count('\n\n') > 5:
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if paragraphs:
            return paragraphs
    
    # Si no hay dobles saltos, dividir por salto simple y agrupar
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        current_chunk.append(line)
        current_length += len(line)
        
        if current_length >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def search(query_keywords, letters):
    """Busca párrafos relevantes usando keywords en inglés"""
    
    if not query_keywords:
        return []
    
    # Extraer keywords y año
    keywords = [w.lower() for w in re.findall(r'\w+', query_keywords) if len(w) > 2]
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query_keywords)
    target_year = year_match.group(1) if year_match else None
    
    if not keywords and not target_year:
        return []
    
    results = []
    
    for year, data in letters.items():
        if target_year and year != target_year:
            continue
            
        text = data.get('text', '')
        chunks = chunk_text(text)
        
        for chunk in chunks:
            if len(chunk) < 80:
                continue
                
            chunk_lower = chunk.lower()
            score = 0
            
            for kw in keywords:
                count = chunk_lower.count(kw)
                if count > 0:
                    score += count * 2
            
            if score > 0 or target_year:
                results.append({
                    'year': year, 
                    'text': chunk[:2000], 
                    'score': score if score > 0 else 0.1
                })
    
    results.sort(key=lambda x: -x['score'])
    return results[:5]

# ============================================
# GENERATE RESPONSE
# ============================================
def generate(query, letters, client):
    # 1. Traducir query a keywords en inglés
    english_keywords = translate_to_english(query, client)
    
    # 2. Buscar en las cartas
    if english_keywords:
        chunks = search(english_keywords, letters)
    else:
        chunks = []
    
    context = "\n\n".join([f"[{c['year']}]: {c['text']}" for c in chunks])
    sources = list(dict.fromkeys([c['year'] for c in chunks]))
    
    # 3. Generar respuesta
    if not chunks:
        # Es un saludo o no hay resultados
        system_msg = """Eres el asistente de BQuant especializado en las cartas anuales de Warren Buffett (1977-2024). 
Tienes acceso a 48 cartas que abarcan 47 años de sabiduría inversora.
Responde de forma amigable y breve en español. Invita al usuario a preguntar sobre temas como inversión, empresas, crisis financieras, inflación, seguros, o cualquier año específico."""
        user_msg = query
    else:
        system_msg = """Eres un asistente experto en las cartas anuales de Warren Buffett (1977-2024).

REGLAS:
- Responde basándote SOLO en el contexto proporcionado
- Cita el año cuando menciones algo específico: "En [año], Buffett..."
- Responde en español
- Sé conciso pero completo (200-300 palabras)
- No inventes información
- Sé directo y profesional"""
        user_msg = f"""CONTEXTO DE LAS CARTAS:
{context}

PREGUNTA DEL USUARIO: {query}"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=700,
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
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Header
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

# Welcome + Suggestions
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <div class="welcome-title">¿Qué quieres saber de Buffett?</div>
        <div class="welcome-sub">47 años de sabiduría inversora a tu alcance</div>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    suggestions = ["Filosofía de inversión", "Crisis de 2008", "Sobre la inflación", "Carta de 1983", "Buena empresa", "Opinión del oro"]
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = f"¿Qué dice Buffett sobre {s.lower()}?"
                st.rerun()

# Procesar query pendiente
if st.session_state.pending_query and letters:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.spinner("Buscando en las cartas..."):
        response, sources = generate(query, letters, client)
    
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
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
