
import streamlit as st
from groq import Groq
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import re
import math
from collections import Counter

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="BQuant · Buffett", page_icon="⚡", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================
# CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@400;500;600&display=swap');
    
    :root {
        --bg: #0a0a0a;
        --border: #222;
        --accent: #d4a853;
        --text: #eee;
        --muted: #777;
    }
    
    * { font-family: 'DM Sans', -apple-system, sans-serif; }
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    
    html, body, .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] > div,
    .main, 
    .block-container {
        background: var(--bg) !important;
        background-color: var(--bg) !important;
    }
    
    .block-container { 
        padding: 2rem 1.5rem 8rem !important; 
        max-width: 780px !important; 
    }
    
    [data-testid="stBottom"] {
        background: var(--bg) !important;
        border-top: 1px solid var(--border) !important;
        padding: 1rem 0 1.5rem 0 !important;
    }
    
    [data-testid="stBottomBlockContainer"] {
        background: var(--bg) !important;
        max-width: 780px !important;
        margin: 0 auto !important;
        padding: 0 1.5rem !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: #111 !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 18px !important;
        min-height: 60px !important;
    }
    
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    
    [data-testid="stChatInput"] input, 
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        font-size: 1.05rem !important;
        padding: 1.1rem 1.3rem !important;
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] input::placeholder { color: var(--muted) !important; }
    
    [data-testid="stChatInput"] button {
        background: var(--accent) !important;
        border: none !important;
        border-radius: 14px !important;
        min-width: 48px !important;
        min-height: 48px !important;
        margin: 6px 8px 6px 0 !important;
    }
    
    .header {
        text-align: center;
        padding: 1.5rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .logo {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 2.2rem;
        color: var(--text);
        letter-spacing: -1px;
    }
    .logo span { color: var(--accent); }
    .tagline {
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-top: 6px;
    }
    .meta {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 1rem;
        font-size: 0.78rem;
        color: var(--muted);
    }
    .meta-dot {
        display: inline-block;
        width: 6px; height: 6px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 50% { opacity: 0.4; } }
    
    .welcome {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
    }
    .welcome h1 {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.8rem;
        font-weight: 400;
        color: var(--text);
        line-height: 1.35;
        margin: 0 0 0.4rem;
    }
    .welcome p {
        color: var(--muted);
        font-size: 0.95rem;
    }
    
    .stButton > button {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--muted) !important;
        border-radius: 50px !important;
        padding: 0.55rem 1.1rem !important;
        font-size: 0.82rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: rgba(212,168,83,0.08) !important;
    }
    
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 1rem 0 !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }
    [data-testid="stChatMessageContent"], 
    [data-testid="stChatMessageContent"] p {
        color: var(--text) !important;
        font-size: 0.92rem !important;
        line-height: 1.7 !important;
    }
    
    .sources {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 0.8rem;
        padding-top: 0.6rem;
        border-top: 1px dashed var(--border);
        align-items: center;
    }
    .src {
        background: #111;
        border: 1px solid var(--border);
        color: var(--muted);
        padding: 3px 10px;
        border-radius: 5px;
        font-size: 0.7rem;
        font-weight: 500;
    }
    .src b { color: var(--accent); }
    
    .confidence {
        font-size: 0.68rem;
        padding: 3px 10px;
        border-radius: 5px;
        margin-left: auto;
    }
    .confidence.low {
        background: rgba(255, 152, 0, 0.15);
        border: 1px solid rgba(255, 152, 0, 0.3);
        color: #ff9800;
    }
    .confidence.medium {
        background: rgba(255, 235, 59, 0.15);
        border: 1px solid rgba(255, 235, 59, 0.3);
        color: #fdd835;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 1.5rem;
        border-top: 1px solid var(--border);
        font-size: 0.72rem;
        color: var(--muted);
    }
    .footer a { color: var(--muted); text-decoration: none; }
    .footer a:hover { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD RESOURCES
# ============================================
@st.cache_resource
def load_index():
    try:
        return faiss.read_index(INDEX_FILE)
    except:
        return None

@st.cache_resource
def load_chunks():
    try:
        with open(CHUNKS_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def build_bm25_index(_chunks):
    """Construye índice BM25 para búsqueda keyword"""
    return BM25(_chunks)


# ============================================
# BM25 IMPLEMENTATION
# ============================================
class BM25:
    """Implementación simple de BM25 para búsqueda keyword"""
    
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_count = len(chunks)
        
        # Tokenizar documentos
        self.doc_tokens = [self._tokenize(c['text']) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.doc_count if self.doc_count > 0 else 0
        
        # Calcular IDF
        self.idf = {}
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
    
    def _tokenize(self, text):
        """Tokeniza texto en palabras"""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]{2,}\b', text)
        return tokens
    
    def search(self, query, top_k=20):
        """Busca documentos más relevantes"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = 0
            doc_len = self.doc_lens[idx]
            token_freq = Counter(doc_tokens)
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                tf = token_freq.get(token, 0)
                idf = self.idf[token]
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
            
            if score > 0:
                scores.append((idx, score))
        
        # Ordenar por score
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ============================================
# QUERY ENHANCEMENT
# ============================================
def enhance_query(query: str, client) -> str:
    """Traduce y expande la query a inglés"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are a search query optimizer for Warren Buffett's annual letters to shareholders.

Your task: Convert the user's question into optimal English search terms.

Rules:
1. Translate to English if not already
2. Extract key concepts and add synonyms Buffett would use
3. Return ONLY the search terms, no explanations
4. Keep it concise (max 10-15 words)

Examples:
- "¿Qué opina de la diversificación?" → "diversification concentrated portfolio focus circle of competence"
- "¿Cómo valora una empresa?" → "valuation intrinsic value earnings owner earnings cash flow"
- "¿Qué son los moats?" → "moat competitive advantage durable economic franchise"
- "Crisis de 2008" → "2008 financial crisis panic fear recession"
- "¿Por qué compró See's Candies?" → "See's Candies acquisition purchase brand pricing power"
- "¿Qué piensa del oro?" → "gold investment store of value unproductive asset"
- "Recompra de acciones" → "share repurchase buyback intrinsic value per share"
- "¿Qué dijo sobre los bancos?" → "banks banking financial institutions leverage"
- "Filosofía de inversión" → "investment philosophy value investing long-term patience margin of safety"
"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=50
        )
        enhanced = response.choices[0].message.content.strip()
        return enhanced if enhanced else query
    except Exception as e:
        if "rate" in str(e).lower():
            raise e  # Propagar para manejar arriba
        return query


# ============================================
# RESOLVE FOLLOW-UP QUERY
# ============================================
def resolve_followup(query: str, conversation_history: list, client) -> str:
    """
    Resuelve referencias anafóricas en preguntas de seguimiento.
    "¿Y qué más dijo sobre eso?" → "¿Qué más dijo sobre la diversificación?"
    """
    # Si no hay historial o la query es completa, devolverla tal cual
    if not conversation_history or len(conversation_history) < 2:
        return query
    
    # Detectar si es un follow-up (pregunta corta con pronombres/referencias)
    followup_indicators = [
        r'\beso\b', r'\beste tema\b', r'\bal respecto\b', r'\bsobre eso\b',
        r'\by qué\b', r'\bpor qué\b.*\?$', r'\bcuándo\b', r'\bdónde\b',
        r'\bmás\b', r'\botro\b', r'\botra\b', r'\btambién\b',
        r'^y ', r'^pero ', r'^entonces ',
        r'\bél\b', r'\bella\b', r'\blo\b', r'\bla\b', r'\bles\b'
    ]
    
    is_followup = any(re.search(p, query.lower()) for p in followup_indicators)
    is_short = len(query.split()) < 8
    
    if not (is_followup or is_short):
        return query
    
    # Obtener contexto de la conversación reciente
    recent_context = []
    for msg in conversation_history[-4:]:  # Últimos 2 intercambios
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg["content"][:300]  # Truncar para no exceder contexto
        recent_context.append(f"{role}: {content}")
    
    context_str = "\n".join(recent_context)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Tu tarea es reformular preguntas de seguimiento para que sean autocontenidas.

Si la pregunta actual hace referencia a temas de la conversación anterior (usando "eso", "este tema", "él", pronombres, etc.), reescríbela para que sea clara sin contexto.

Si la pregunta ya es clara y autocontenida, devuélvela sin cambios.

Responde SOLO con la pregunta reformulada, nada más.

Ejemplos:
- Contexto: hablando de diversificación → "¿Y qué más dijo?" → "¿Qué más dijo Buffett sobre la diversificación?"
- Contexto: hablando del oro → "¿Desde cuándo piensa así?" → "¿Desde cuándo piensa Buffett que el oro no es buena inversión?"
- Contexto: hablando de See's → "¿Cuánto pagó?" → "¿Cuánto pagó Buffett por See's Candies?"
- Sin contexto relevante → "¿Qué opina del oro?" → "¿Qué opina del oro?" (sin cambios)
"""
                },
                {
                    "role": "user", 
                    "content": f"CONVERSACIÓN RECIENTE:\n{context_str}\n\nNUEVA PREGUNTA: {query}"
                }
            ],
            temperature=0,
            max_tokens=100
        )
        resolved = response.choices[0].message.content.strip()
        return resolved if resolved else query
    except:
        return query


# ============================================
# HYBRID SEARCH (BM25 + EMBEDDINGS)
# ============================================
def hybrid_search(
    query: str, 
    index, 
    chunks: list, 
    model, 
    bm25: BM25,
    client,
    top_k: int = 8,
    semantic_weight: float = 0.7
) -> list[dict]:
    """
    Búsqueda híbrida combinando embeddings y BM25.
    - Embeddings: bueno para semántica y sinónimos
    - BM25: bueno para nombres propios y términos exactos
    """
    
    # Extraer año si se menciona
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    year_filter = year_match.group(1) if year_match else None
    
    # Detectar si es año + concepto
    query_without_year = re.sub(r'\b(19[7-9]\d|20[0-2]\d)\b', '', query).strip()
    is_year_plus_concept = year_filter and len(query_without_year.split()) > 3
    
    if is_year_plus_concept:
        base_year = int(year_filter)
        valid_years = {str(y) for y in [base_year, base_year + 1, base_year + 2] if 1977 <= y <= 2024}
    else:
        valid_years = {year_filter} if year_filter else None
    
    # 1. Búsqueda semántica
    enhanced_query = enhance_query(query, client)
    q_emb = model.encode([enhanced_query], normalize_embeddings=True, convert_to_numpy=True)
    
    retrieve_k = 30  # Recuperar más para luego combinar
    sem_scores, sem_indices = index.search(q_emb.astype('float32'), min(retrieve_k, len(chunks)))
    
    # Normalizar scores semánticos a [0, 1]
    sem_results = {}
    max_sem = max(sem_scores[0]) if sem_scores[0].max() > 0 else 1
    for score, idx in zip(sem_scores[0], sem_indices[0]):
        if idx >= 0:
            sem_results[idx] = score / max_sem
    
    # 2. Búsqueda BM25
    bm25_results_raw = bm25.search(enhanced_query, top_k=retrieve_k)
    
    # Normalizar scores BM25 a [0, 1]
    bm25_results = {}
    max_bm25 = bm25_results_raw[0][1] if bm25_results_raw else 1
    for idx, score in bm25_results_raw:
        bm25_results[idx] = score / max_bm25 if max_bm25 > 0 else 0
    
    # 3. Combinar scores (Reciprocal Rank Fusion alternativo: weighted sum)
    all_indices = set(sem_results.keys()) | set(bm25_results.keys())
    combined_scores = {}
    
    for idx in all_indices:
        sem_score = sem_results.get(idx, 0)
        bm25_score = bm25_results.get(idx, 0)
        combined_scores[idx] = (semantic_weight * sem_score) + ((1 - semantic_weight) * bm25_score)
    
    # 4. Ordenar y filtrar
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: -combined_scores[x])
    
    results = []
    for idx in sorted_indices:
        chunk = chunks[idx].copy()
        chunk['score'] = combined_scores[idx]
        chunk['sem_score'] = sem_results.get(idx, 0)
        chunk['bm25_score'] = bm25_results.get(idx, 0)
        
        # Filtrar por años válidos
        if valid_years and chunk['year'] not in valid_years:
            continue
        
        # Diversificar: máximo 3 chunks por año
        year_count = sum(1 for r in results if r['year'] == chunk['year'])
        if year_count >= 3:
            continue
        
        results.append(chunk)
        if len(results) >= top_k * 2:  # Recuperar más para reranking
            break
    
    return results


# ============================================
# RERANKING WITH LLM
# ============================================
def rerank_with_llm(query: str, chunks: list[dict], client, top_k: int = 8) -> list[dict]:
    """
    Reordena chunks usando el LLM para evaluar relevancia real.
    """
    if len(chunks) <= top_k:
        return chunks
    
    # Preparar chunks para evaluación
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:16]):  # Máximo 16 para no exceder contexto
        preview = chunk['text'][:400].replace('\n', ' ')
        chunk_summaries.append(f"[{i}] ({chunk['year']}): {preview}...")
    
    chunks_text = "\n\n".join(chunk_summaries)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"""Eres un experto evaluando relevancia de documentos.

Dada una pregunta y una lista de fragmentos de las cartas de Warren Buffett, ordena los {top_k} fragmentos MÁS RELEVANTES para responder la pregunta.

Responde SOLO con los números de los fragmentos ordenados de más a menos relevante, separados por comas.
Ejemplo: 3,7,1,5,2,8,4,6

Criterios de relevancia:
- Responde directamente la pregunta
- Contiene información específica (no genérica)
- Incluye datos, ejemplos o citas de Buffett
"""
                },
                {
                    "role": "user",
                    "content": f"PREGUNTA: {query}\n\nFRAGMENTOS:\n{chunks_text}"
                }
            ],
            temperature=0,
            max_tokens=50
        )
        
        # Parsear respuesta
        ranking_str = response.choices[0].message.content.strip()
        ranking = [int(x.strip()) for x in re.findall(r'\d+', ranking_str)]
        
        # Reordenar chunks según ranking
        reranked = []
        seen = set()
        for idx in ranking:
            if idx < len(chunks) and idx not in seen:
                reranked.append(chunks[idx])
                seen.add(idx)
        
        # Añadir cualquier chunk faltante al final
        for i, chunk in enumerate(chunks):
            if i not in seen and len(reranked) < top_k:
                reranked.append(chunk)
        
        return reranked[:top_k]
    
    except:
        # Si falla, devolver los primeros top_k
        return chunks[:top_k]


# ============================================
# CONFIDENCE DETECTION
# ============================================
def calculate_confidence(results: list[dict], query: str) -> tuple[str, float]:
    """Calcula nivel de confianza"""
    if not results:
        return "none", 0.0
    
    scores = [r['score'] for r in results]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    if max_score >= 0.5 and avg_score >= 0.35:
        return "high", max_score
    elif max_score >= 0.30 and avg_score >= 0.20:
        return "medium", max_score
    else:
        return "low", max_score


# ============================================
# GENERATE RESPONSE WITH QUOTES
# ============================================
def generate_response(
    query: str, 
    results: list[dict], 
    client,
    conversation_history: list = None
) -> tuple[str, list, str]:
    """
    Genera respuesta con citas exactas y contexto conversacional.
    """
    confidence_level, confidence_score = calculate_confidence(results, query)
    
    if not results or confidence_level == "none":
        system = """Eres el asistente de BQuant sobre las cartas de Warren Buffett (1977-2024).
Responde amigablemente y sugiere temas: inversión, adquisiciones, crisis, seguros, o años específicos."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
        confidence_level = "none"
    
    elif confidence_level == "low":
        sorted_results = sorted(results, key=lambda x: x['year'])
        context = "\n\n---\n\n".join([f"[CARTA {r['year']}]\n{r['text']}" for r in sorted_results])
        
        system = """Eres un experto en las cartas anuales de Warren Buffett (1977-2024).

⚠️ ALERTA: Los fragmentos recuperados tienen BAJA RELEVANCIA con la pregunta.

INSTRUCCIONES CRÍTICAS:
1. Si el contexto NO contiene información directamente relevante, DEBES decir:
   "No encontré información específica sobre [tema] en las cartas de Buffett."
2. NO inventes información, fechas, ni citas
3. NO menciones entrevistas ni fuentes externas
4. Si hay información tangencialmente relacionada, puedes mencionarla aclarando que no es respuesta directa
5. Responde en español

Es mejor admitir que no tienes la información que inventar una respuesta."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXTO (baja relevancia):\n{context}\n\n---\n\nPREGUNTA: {query}"}
        ]
    
    else:
        sorted_results = sorted(results, key=lambda x: x['year'])
        context = "\n\n---\n\n".join([f"[CARTA {r['year']}]\n{r['text']}" for r in sorted_results])
        
        system = """Eres un experto en las cartas anuales de Warren Buffett (1977-2024).

INSTRUCCIONES:
1. Sintetiza la información de TODAS las cartas proporcionadas
2. Cita el año: "En [año], Buffett explicó que..."
3. **INCLUYE CITAS TEXTUALES**: Cuando encuentres frases memorables o específicas de Buffett, cítalas entre comillas: «frase exacta»
4. Si hay evolución en su pensamiento, menciónalo
5. Sé directo - Buffett tiene opiniones fuertes, no las suavices
6. Si la información no está en el contexto, dilo
7. Responde en español, máximo 300 palabras
8. NUNCA inventes información

FORMATO DE CITAS:
- Usa «comillas latinas» para citas textuales de Buffett
- Ejemplo: En 1993, Buffett afirmó que «la diversificación es protección contra la ignorancia»."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXTO DE LAS CARTAS:\n{context}\n\n---\n\nPREGUNTA: {query}"}
        ]
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2 if confidence_level == "low" else 0.3,
            max_tokens=800
        )
        return resp.choices[0].message.content, results, confidence_level
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            raise e  # Propagar para manejar en search_and_generate
        return f"Error: {e}", [], "error"


# ============================================
# MAIN SEARCH PIPELINE
# ============================================
def search_and_generate(
    query: str,
    index,
    chunks: list,
    model,
    bm25: BM25,
    client,
    conversation_history: list = None
) -> tuple[str, list, str]:
    """
    Pipeline completo:
    1. Resolver follow-ups
    2. Búsqueda híbrida
    3. Reranking
    4. Generación con citas
    """
    
    try:
        # 1. Resolver referencias anafóricas
        resolved_query = resolve_followup(query, conversation_history or [], client)
        
        # 2. Búsqueda híbrida
        candidates = hybrid_search(resolved_query, index, chunks, model, bm25, client, top_k=16)
        
        # 3. Reranking con LLM
        if len(candidates) > 8:
            reranked = rerank_with_llm(resolved_query, candidates, client, top_k=8)
        else:
            reranked = candidates
        
        # 4. Generar respuesta
        response, used_chunks, confidence = generate_response(
            resolved_query, 
            reranked, 
            client,
            conversation_history
        )
        
        return response, used_chunks, confidence
    
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            return "⏳ Demasiadas consultas. Intenta en unos segundos.", [], "error"
        return f"Error: {str(e)}", [], "error"


# ============================================
# UI HELPERS
# ============================================
def show_sources(results: list, confidence: str = "high"):
    if not results:
        return
    
    confidence_indicators = {
        "high": ("🟢", "Alta relevancia"),
        "medium": ("🟡", "Relevancia media"),
        "low": ("🟠", "Baja relevancia"),
        "none": ("🔴", "Sin resultados"),
        "error": ("❌", "Error")
    }
    indicator, label = confidence_indicators.get(confidence, ("⚪", ""))
    
    years = sorted(set(r["year"] for r in results))
    pills = ''.join([f'<span class="src"><b>{y}</b></span>' for y in years])
    
    confidence_html = ""
    if confidence in ["low", "medium"]:
        confidence_html = f'<span class="confidence {confidence}">{indicator} {label}</span>'
    
    st.markdown(f'''
    <div class="sources">
        {pills}
        {confidence_html}
    </div>
    ''', unsafe_allow_html=True)


# ============================================
# MAIN
# ============================================
def main():
    index = load_index()
    chunks = load_chunks()
    model = load_model()
    client = get_client()
    
    if not index or not chunks:
        st.error("⚠️ Ejecuta primero: `python build_index.py`")
        st.stop()
    
    # Construir índice BM25
    bm25 = build_bm25_index(chunks)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    
    # HEADER
    st.markdown("""
    <div class="header">
        <div class="logo">⚡ <span>BQuant</span>ChatBot</div>
        <div class="tagline">Berkshire Letters AI</div>
        <div class="meta">
            <span><span class="meta-dot"></span>Online</span>
            <span>📚 48 cartas</span>
            <span>1977–2024</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # WELCOME
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <h1>Pregunta sobre las cartas<br/>de Warren Buffett</h1>
            <p>47 años de filosofía inversora</p>
        </div>
        """, unsafe_allow_html=True)
        
        suggestions = [
            ("Filosofía", "¿Cuál es la filosofía de inversión de Buffett?"),
            ("Crisis 2008", "¿Qué dijo Buffett sobre la crisis de 2008?"),
            ("Carta 1983", "Resume la carta de 1983"),
            ("Oro", "¿Qué opina Buffett del oro?"),
            ("Moats", "¿Qué son los moats?"),
            ("Seguros", "¿Por qué son importantes los seguros para Berkshire?"),
        ]
        
        cols = st.columns(3)
        for i, (label, q) in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(label, key=f"s{i}", use_container_width=True):
                    st.session_state.pending = q
                    st.rerun()
    
    # PENDING
    if st.session_state.pending:
        q = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": q})
        
        response, used, confidence = search_and_generate(
            q, index, chunks, model, bm25, client,
            st.session_state.messages
        )
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "sources": used,
            "confidence": confidence
        })
        st.rerun()
    
    # MESSAGES
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
            st.write(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"], msg.get("confidence", "high"))
    
    # INPUT
    if prompt := st.chat_input("Pregunta sobre las cartas de Buffett..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Buscando..."):
                response, used, confidence = search_and_generate(
                    prompt, index, chunks, model, bm25, client,
                    st.session_state.messages
                )
            st.write(response)
            show_sources(used, confidence)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "sources": used,
            "confidence": confidence
        })
    
    # FOOTER
    st.markdown("""
    <div class="footer">
        <a href="https://bquantfinance.com">BQuant Finance</a> · 
        <a href="https://twitter.com/Gsnchez">@Gsnchez</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
