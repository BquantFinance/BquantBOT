import streamlit as st
from groq import Groq
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import re
import math
import json
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

# ============================================
# CONFIG
# ============================================
st.set_page_config(page_title="BQuant · Buffett + Insiders", page_icon="⚡", layout="centered")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
INSIDER_FILE = "insider_data_all.csv"
CONGRESS_FILE = "Congresstrading.csv"
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
        --accent-green: #22c55e;
        --accent-red: #ef4444;
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
    .src.insider { border-color: #3b82f6; }
    .src.insider b { color: #3b82f6; }
    .src.congress { border-color: #8b5cf6; }
    .src.congress b { color: #8b5cf6; }
    .src.democrat { border-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
    .src.democrat b { color: #3b82f6; }
    .src.republican { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }
    .src.republican b { color: #ef4444; }
    .src.buy { border-color: var(--accent-green); }
    .src.buy b { color: var(--accent-green); }
    .src.sell { border-color: var(--accent-red); }
    .src.sell b { color: var(--accent-red); }
    
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
    
    .data-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 8px;
    }
    .data-badge.buffett {
        background: rgba(212, 168, 83, 0.2);
        color: var(--accent);
        border: 1px solid rgba(212, 168, 83, 0.4);
    }
    .data-badge.insider {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .data-badge.congress {
        background: rgba(139, 92, 246, 0.2);
        color: #8b5cf6;
        border: 1px solid rgba(139, 92, 246, 0.4);
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

@st.cache_data
def load_insider_data():
    """Carga y preprocesa datos de insider trading (764K+ transacciones)"""
    try:
        # Trade Date es el índice
        df = pd.read_csv(INSIDER_FILE, index_col=0)
        
        # Convertir índice a datetime
        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index.name = 'Trade Date'
        
        # Reset index para tenerlo como columna (más fácil para filtros)
        df = df.reset_index()
        
        # Convertir Filing Date
        df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
        
        # Limpiar valores numéricos
        df['Value_C'] = pd.to_numeric(df['Value_C'], errors='coerce').fillna(0)
        df['Value_V'] = pd.to_numeric(df['Value_V'], errors='coerce').fillna(0)
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')
        
        # Asegurar que is_sp500 es booleano
        df['is_sp500'] = df['is_sp500'].astype(bool)
        
        return df
    except Exception as e:
        st.error(f"Error cargando insider data: {e}")
        return None


@st.cache_data
def load_congress_data():
    """Carga y preprocesa datos de trading del Congreso (93K+ transacciones)"""
    try:
        df = pd.read_csv(CONGRESS_FILE)
        
        # Convertir fechas
        df['Filed Date'] = pd.to_datetime(df['Filed Date'], errors='coerce')
        df['Purchase Date'] = pd.to_datetime(df['Purchase Date'], errors='coerce')
        
        # Limpiar valores numéricos
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Gain/Loss'] = pd.to_numeric(df['Gain/Loss'], errors='coerce').fillna(0)
        
        # Extraer año y mes de Filed Date
        df['year'] = df['Filed Date'].dt.year
        df['month'] = df['Filed Date'].dt.month
        df['quarter'] = df['Filed Date'].dt.quarter
        
        # Limpiar campos de texto
        df['Party'] = df['Party'].fillna('Unknown')
        df['Chamber'] = df['Chamber'].fillna('Unknown')
        df['Company'] = df['Company'].fillna('')
        df['Industry'] = df['Industry'].fillna('Unknown')
        df['Security'] = df['Security'].fillna('Stock')
        df['Amount Range'] = df['Amount Range'].fillna('')
        df['Symbol'] = df['Symbol'].fillna('')
        
        return df
    except Exception as e:
        st.error(f"Error cargando congress data: {e}")
        return None


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
# QUERY ROUTER
# ============================================
def route_query(query: str, client) -> str:
    """
    Determina si la consulta es sobre:
    - INSIDER: datos de insider trading del mercado US (764K+ transacciones)
    - CONGRESS: datos de trading del Congreso US (93K+ transacciones)
    - BUFFETT: cartas anuales de Warren Buffett
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Clasifica la consulta en UNA categoría:

CONGRESS: Preguntas sobre trading de políticos, congresistas, senadores, representantes,
          transacciones del Congreso, Pelosi, demócratas/republicanos comprando acciones,
          House, Senate, partidos políticos y sus inversiones.
          
          Palabras clave: congreso, congresista, senador, representante, político, Pelosi,
          demócrata, republicano, House, Senate, cámara, partido, políticos comprando/vendiendo

INSIDER: Preguntas sobre insider trading CORPORATIVO, transacciones de ejecutivos de empresas,
         CEOs, CFOs, directores de empresas comprando/vendiendo acciones de SUS propias empresas,
         movimientos de directivos corporativos (NO políticos).
         
         Palabras clave: insider, ejecutivo, directivo, CEO, CFO, director corporativo,
         Tim Cook, Elon Musk (transacciones corporativas), S&P 500 insiders

BUFFETT: Preguntas sobre la filosofía de inversión de Warren Buffett, cartas de Berkshire Hathaway,
         sabiduría inversora, value investing, qué piensa/dijo Buffett.
         
         Palabras clave: Buffett, Berkshire, carta, filosofía, value investing, moat

Responde SOLO con: CONGRESS, INSIDER o BUFFETT"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=10
        )
        route = response.choices[0].message.content.strip().upper()
        # Limpiar respuesta
        if "CONGRESS" in route:
            return "CONGRESS"
        elif "INSIDER" in route:
            return "INSIDER"
        elif "BUFFETT" in route:
            return "BUFFETT"
        return "BUFFETT"  # Default
    except Exception as e:
        return "BUFFETT"


# ============================================
# INSIDER DATA SEARCH
# ============================================
def extract_insider_params(query: str, client) -> dict:
    """Extrae parámetros estructurados de la consulta para filtrar datos de insider"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Extrae parámetros de búsqueda de la consulta sobre insider trading.
Responde SOLO con JSON válido, sin texto adicional:

{
    "symbols": ["AAPL", "MSFT"] o null,
    "insider_names": ["Tim Cook"] o null,
    "titles": ["CEO", "CFO", "Dir"] o null,
    "trade_type": "buy" o "sell" o "both" o null,
    "year": 2024 o null,
    "quarter": 1 o 2 o 3 o 4 o null,
    "month": 1-12 o null,
    "min_value": 1000000 o null,
    "is_sp500": true o false o null,
    "aggregation": "by_company" o "by_insider" o "by_month" o "by_year" o "top_buys" o "top_sells" o "summary" o "recent" o null,
    "limit": 10,
    "sort_by": "value" o "date" o "quantity" o null
}

Ejemplos:
- "¿Qué insiders compraron en Apple?" → {"symbols": ["AAPL"], "trade_type": "buy", "limit": 20}
- "Mayores ventas de 2024" → {"trade_type": "sell", "year": 2024, "aggregation": "top_sells", "limit": 10}
- "¿Tim Cook vendió acciones?" → {"insider_names": ["Tim Cook"], "trade_type": "sell"}
- "Resumen de insider trading Q4 2024" → {"year": 2024, "quarter": 4, "aggregation": "summary"}
- "CEOs que más compraron" → {"titles": ["CEO"], "trade_type": "buy", "aggregation": "by_insider", "sort_by": "value"}
- "Compras mayores a 10 millones" → {"trade_type": "buy", "min_value": 10000000, "aggregation": "top_buys"}
- "Evolución del insider trading por año" → {"aggregation": "by_year"}
- "¿Cómo han cambiado las compras en AAPL?" → {"symbols": ["AAPL"], "aggregation": "by_year"}
- "Insider trading en empresas del S&P 500" → {"is_sp500": true, "aggregation": "summary"}
- "Transacciones recientes" → {"aggregation": "recent", "limit": 20}
- "¿Qué pasó en noviembre 2024?" → {"year": 2024, "month": 11, "aggregation": "summary"}
- "Empresas fuera del S&P con más compras" → {"is_sp500": false, "trade_type": "buy", "aggregation": "by_company"}
- "Últimas compras de directores" → {"titles": ["Dir", "Director"], "trade_type": "buy", "aggregation": "recent"}"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        # Limpiar posibles backticks de markdown
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = re.sub(r'^```\s*', '', content)
        
        params = json.loads(content)
        return params
    except Exception as e:
        return {"limit": 20, "aggregation": "summary"}


def search_insider_data(query: str, df: pd.DataFrame, client) -> dict:
    """Busca y filtra datos de insider trading según la consulta (764K+ transacciones)"""
    
    params = extract_insider_params(query, client)
    filtered = df.copy()
    
    # Filtrar por S&P 500
    if params.get("is_sp500") is not None:
        filtered = filtered[filtered['is_sp500'] == params["is_sp500"]]
    
    # Filtrar por símbolos
    if params.get("symbols"):
        symbols = [s.upper() for s in params["symbols"]]
        filtered = filtered[filtered['Symbol'].isin(symbols)]
    
    # Filtrar por nombre de insider
    if params.get("insider_names"):
        pattern = '|'.join(params["insider_names"])
        filtered = filtered[filtered['Insider Name'].str.contains(pattern, case=False, na=False)]
    
    # Filtrar por título
    if params.get("titles"):
        pattern = '|'.join(params["titles"])
        filtered = filtered[filtered['Title'].str.contains(pattern, case=False, na=False)]
    
    # Filtrar por tipo de transacción
    if params.get("trade_type") == "buy":
        filtered = filtered[filtered['Trade Type'].str.contains('P', case=False, na=False)]
    elif params.get("trade_type") == "sell":
        filtered = filtered[filtered['Trade Type'].str.contains('S', case=False, na=False)]
    
    # Filtrar por año
    if params.get("year"):
        filtered = filtered[filtered['year'] == float(params["year"])]
    
    # Filtrar por trimestre
    if params.get("quarter"):
        filtered = filtered[filtered['quarter'] == float(params["quarter"])]
    
    # Filtrar por mes
    if params.get("month"):
        filtered = filtered[filtered['month'] == float(params["month"])]
    
    # Filtrar por valor mínimo
    if params.get("min_value"):
        min_val = params["min_value"]
        filtered = filtered[
            (filtered['Value_C'].abs() >= min_val) | 
            (filtered['Value_V'].abs() >= min_val)
        ]
    
    limit = params.get("limit", 20)
    aggregation = params.get("aggregation")
    
    # Realizar agregación según tipo
    if aggregation == "by_company":
        result = filtered.groupby('Symbol').agg({
            'Value_C': 'sum',
            'Value_V': 'sum',
            'Qty': 'sum',
            'Trade Date': 'count',
            'is_sp500': 'first'
        }).rename(columns={'Trade Date': 'Num_Trades'})
        result['Net_Value'] = result['Value_C'] + result['Value_V']
        result = result.sort_values('Net_Value', ascending=False).head(limit)
        result_type = "aggregated_by_company"
        
    elif aggregation == "by_insider":
        result = filtered.groupby(['Insider Name', 'Symbol']).agg({
            'Value_C': 'sum',
            'Value_V': 'sum',
            'Qty': 'sum',
            'Title': 'first',
            'Trade Date': 'count',
            'Price': 'mean',
            'is_sp500': 'first'
        }).rename(columns={'Trade Date': 'Num_Trades', 'Price': 'Avg_Price'})
        result['Net_Value'] = result['Value_C'] + result['Value_V']
        result = result.sort_values('Value_C', ascending=False).head(limit)
        result_type = "aggregated_by_insider"
        
    elif aggregation == "by_month":
        filtered['YearMonth'] = filtered['Trade Date'].dt.to_period('M')
        result = filtered.groupby('YearMonth').agg({
            'Value_C': 'sum',
            'Value_V': 'sum',
            'Qty': 'sum',
            'Symbol': 'count'
        }).rename(columns={'Symbol': 'Num_Trades'})
        result['Net_Value'] = result['Value_C'] + result['Value_V']
        result = result.sort_index(ascending=False).head(limit)
        result_type = "aggregated_by_month"
    
    elif aggregation == "by_year":
        result = filtered.groupby('year').agg({
            'Value_C': 'sum',
            'Value_V': 'sum',
            'Qty': 'sum',
            'Symbol': 'nunique',
            'Insider Name': 'nunique',
            'Trade Date': 'count'
        }).rename(columns={'Symbol': 'Unique_Companies', 'Insider Name': 'Unique_Insiders', 'Trade Date': 'Num_Trades'})
        result['Net_Value'] = result['Value_C'] + result['Value_V']
        result = result.sort_index(ascending=False).head(limit)
        result_type = "aggregated_by_year"
        
    elif aggregation == "top_buys":
        result = filtered[filtered['Value_C'] > 0].nlargest(limit, 'Value_C')[
            ['Trade Date', 'year', 'Symbol', 'Insider Name', 'Title', 'Price', 'Qty', 'Value_C', 'ΔOwn', 'Owned', 'SeñalC', 'is_sp500']
        ]
        result_type = "top_buys"
        
    elif aggregation == "top_sells":
        result = filtered[filtered['Value_V'] < 0].nsmallest(limit, 'Value_V')[
            ['Trade Date', 'year', 'Symbol', 'Insider Name', 'Title', 'Price', 'Qty', 'Value_V', 'ΔOwn', 'Owned', 'SeñalV', 'is_sp500']
        ]
        result_type = "top_sells"
    
    elif aggregation == "recent":
        result = filtered.sort_values('Trade Date', ascending=False).head(limit)[
            ['Trade Date', 'year', 'Symbol', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Value_C', 'Value_V', 'ΔOwn', 'is_sp500']
        ]
        result_type = "recent"
        
    elif aggregation == "summary":
        # Resumen general
        total_buys = filtered['Value_C'].sum()
        total_sells = filtered['Value_V'].sum()
        num_transactions = len(filtered)
        unique_companies = filtered['Symbol'].nunique()
        unique_insiders = filtered['Insider Name'].nunique()
        sp500_count = filtered[filtered['is_sp500'] == True]['Symbol'].nunique()
        non_sp500_count = filtered[filtered['is_sp500'] == False]['Symbol'].nunique()
        
        top_buyers = filtered.groupby('Symbol')['Value_C'].sum().nlargest(5)
        top_sellers = filtered.groupby('Symbol')['Value_V'].sum().nsmallest(5)
        
        # Top insiders compradores
        top_insider_buyers = filtered.groupby('Insider Name')['Value_C'].sum().nlargest(5)
        
        result = {
            'total_buys': total_buys,
            'total_sells': total_sells,
            'net_flow': total_buys + total_sells,
            'num_transactions': num_transactions,
            'unique_companies': unique_companies,
            'unique_insiders': unique_insiders,
            'sp500_companies': sp500_count,
            'non_sp500_companies': non_sp500_count,
            'top_buyers': top_buyers.to_dict(),
            'top_sellers': top_sellers.to_dict(),
            'top_insider_buyers': top_insider_buyers.to_dict()
        }
        result_type = "summary"
        
    else:
        # Sin agregación - devolver transacciones individuales
        sort_by = params.get("sort_by", "date")
        cols = ['Trade Date', 'Symbol', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Value_C', 'Value_V', 'ΔOwn', 'Owned', 'is_sp500']
        if sort_by == "value":
            result = filtered.nlargest(limit, 'Value_C')[cols]
        elif sort_by == "quantity":
            result = filtered.nlargest(limit, 'Qty')[cols]
        else:
            result = filtered.sort_values('Trade Date', ascending=False).head(limit)[cols]
        result_type = "transactions"
    
    return {
        "data": result,
        "params": params,
        "total_rows": len(filtered),
        "result_type": result_type
    }


def generate_insider_response(query: str, search_result: dict, client) -> tuple[str, list, str]:
    """Genera respuesta analítica sobre datos de insider trading (764K+ transacciones)"""
    
    result_type = search_result['result_type']
    data = search_result['data']
    total_rows = search_result['total_rows']
    
    # Formatear datos para el contexto
    if result_type == "summary":
        data_context = f"""RESUMEN DE INSIDER TRADING:
- Total compras: ${data['total_buys']:,.0f}
- Total ventas: ${data['total_sells']:,.0f}
- Flujo neto: ${data['net_flow']:,.0f}
- Número de transacciones: {data['num_transactions']:,}
- Empresas únicas: {data['unique_companies']}
- Insiders únicos: {data['unique_insiders']}
- Empresas S&P 500: {data.get('sp500_companies', 'N/A')}
- Empresas fuera S&P 500: {data.get('non_sp500_companies', 'N/A')}

TOP 5 EMPRESAS CON MÁS COMPRAS (por valor):
{json.dumps(data['top_buyers'], indent=2)}

TOP 5 EMPRESAS CON MÁS VENTAS (por valor):
{json.dumps(data['top_sellers'], indent=2)}

TOP 5 INSIDERS COMPRADORES:
{json.dumps(data.get('top_insider_buyers', {}), indent=2)}"""
    elif isinstance(data, pd.DataFrame):
        # Add year summary if available
        if 'year' in data.columns or 'Trade Date' in data.columns:
            try:
                if 'year' not in data.columns and 'Trade Date' in data.columns:
                    data['year'] = pd.to_datetime(data['Trade Date']).dt.year
                year_summary = data.groupby('year').agg({
                    'Value_C': 'sum',
                    'Value_V': 'sum'
                }).to_string()
                data_str = data.head(40).to_string() if len(data) > 40 else data.to_string()
                data_context = f"RESUMEN POR AÑO:\n{year_summary}\n\nDETALLE DE TRANSACCIONES:\n{data_str}"
                if len(data) > 40:
                    data_context += f"\n\n... y {len(data) - 40} registros más"
            except:
                if len(data) > 30:
                    data_context = data.head(30).to_string()
                    data_context += f"\n\n... y {len(data) - 30} registros más"
                else:
                    data_context = data.to_string()
        else:
            if len(data) > 30:
                data_context = data.head(30).to_string()
                data_context += f"\n\n... y {len(data) - 30} registros más"
            else:
                data_context = data.to_string()
    else:
        data_context = str(data)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Eres un analista experto en insider trading con acceso a más de 764,000 transacciones del mercado estadounidense.

INSTRUCCIONES:
1. Analiza los datos proporcionados y responde de forma clara y estructurada
2. Usa formato de moneda apropiado (millones con M, miles con K)
3. **INCLUYE DETALLES CLAVE**: precio, cantidad, % cambio en propiedad cuando estén disponibles
4. Destaca patrones importantes:
   - Concentración de compras/ventas en empresas específicas
   - Insiders notables (CEOs, CFOs) y sus movimientos
   - **CAMBIOS GRANDES EN PROPIEDAD** (ΔOwn alto = señal fuerte)
   - Si SeñalC=1 o SeñalV=1, son transacciones significativas
5. Responde en español
6. Sé directo y analítico
7. Máximo 450 palabras
8. NO inventes datos que no estén en el contexto

FORMATO SUGERIDO PARA TRANSACCIONES:
- **Tim Cook** (CEO) - AAPL: Vendió $50M @ $175/acción (10,000 acciones, -2% de su posición)
- **John Smith** (Dir) - MSFT: Compró $2M @ $420/acción (ΔOwn: +15% 🔥)

CAMPOS DISPONIBLES:
- Value_C/Value_V: Valor en $ de compras/ventas
- Price: Precio por acción
- Qty: Cantidad de acciones
- ΔOwn: Cambio % en propiedad del insider (importante!)
- Owned: Acciones totales después de la transacción
- SeñalC/SeñalV: 1 = transacción significativa, 0 = rutinaria
- Title: Cargo (CEO, CFO, Dir, 10% Owner, etc.)
- is_sp500: True si empresa está en S&P 500

INTERPRETACIÓN:
- Compras de insiders = generalmente señal positiva (usan su dinero)
- Ventas = comunes por compensación, no siempre negativas
- ΔOwn alto + compra = muy bullish (aumentan posición significativamente)
- CEO/CFO comprando = más informativo que director independiente

Datos desde 2022 hasta 2025."""
                },
                {
                    "role": "user",
                    "content": f"""DATOS DE INSIDER TRADING:
{data_context}

Total de transacciones en el filtro: {total_rows:,}
Tipo de resultado: {result_type}

PREGUNTA: {query}"""
                }
            ],
            temperature=0.3,
            max_tokens=900
        )
        
        answer = response.choices[0].message.content
        
        # Preparar fuentes para mostrar
        if isinstance(data, pd.DataFrame) and 'Symbol' in data.columns:
            symbols = data['Symbol'].unique().tolist()[:8]
        elif isinstance(data, pd.DataFrame) and data.index.names[0] == 'Symbol':
            symbols = data.index.get_level_values(0).unique().tolist()[:8]
        elif result_type == "summary":
            symbols = list(data['top_buyers'].keys())[:4] + list(data['top_sellers'].keys())[:4]
        else:
            symbols = []
        
        sources = [{
            "type": "insider",
            "symbols": symbols,
            "count": total_rows,
            "result_type": result_type
        }]
        
        confidence = "high" if total_rows > 0 else "none"
        
        return answer, sources, confidence
        
    except Exception as e:
        return f"Error generando respuesta: {e}", [], "error"


# ============================================
# CONGRESS TRADING - PARAMETER EXTRACTION
# ============================================
def extract_congress_params(query: str, client) -> dict:
    """Extrae parámetros estructurados de la consulta para filtrar datos del Congreso"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Extrae parámetros de búsqueda de la consulta sobre trading del Congreso US.
Responde SOLO con JSON válido, sin texto adicional:

{
    "symbols": ["AAPL", "MSFT"] o null,
    "politicians": ["Nancy Pelosi", "Dan Crenshaw"] o null,
    "party": "D" o "R" o null,
    "chamber": "House" o "Senate" o null,
    "action": "buy" o "sell" o null,
    "industry": "Technology" o "Health Care" o null,
    "year": 2024 o null,
    "quarter": 1 o 2 o 3 o 4 o null,
    "min_amount": 50000 o null,
    "aggregation": "by_politician" o "by_party" o "by_symbol" o "by_industry" o "by_year" o "top_buys" o "top_sells" o "summary" o "recent" o null,
    "limit": 10,
    "sort_by": "amount" o "date" o "gain_loss" o null
}

Ejemplos:
- "¿Qué compró Nancy Pelosi?" → {"politicians": ["Pelosi"], "action": "buy"}
- "Trading de republicanos en 2024" → {"party": "R", "year": 2024, "aggregation": "summary"}
- "¿Qué senadores compraron Apple?" → {"symbols": ["AAPL"], "chamber": "Senate", "action": "buy"}
- "Mayores compras del Congreso" → {"action": "buy", "aggregation": "top_buys", "limit": 10}
- "Demócratas vs Republicanos" → {"aggregation": "by_party"}
- "¿Qué políticos invierten en tecnología?" → {"industry": "Information Technology", "aggregation": "by_politician"}
- "Transacciones recientes del Congreso" → {"aggregation": "recent", "limit": 20}
- "¿Quién ganó más dinero?" → {"aggregation": "by_politician", "sort_by": "gain_loss"}
- "Compras mayores a $100,000" → {"action": "buy", "min_amount": 100000}
- "Evolución del trading del Congreso por año" → {"aggregation": "by_year"}
- "¿Cómo ha cambiado el trading de Pelosi?" → {"politicians": ["Pelosi"], "aggregation": "by_year"}"""
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        content = re.sub(r'^```\s*', '', content)
        
        params = json.loads(content)
        return params
    except Exception as e:
        return {"limit": 20, "aggregation": "summary"}


# ============================================
# CONGRESS TRADING - SEARCH
# ============================================
def search_congress_data(query: str, df: pd.DataFrame, client) -> dict:
    """Busca y filtra datos de trading del Congreso (93K+ transacciones)"""
    
    params = extract_congress_params(query, client)
    filtered = df.copy()
    
    # Filtrar por símbolos
    if params.get("symbols"):
        symbols = [s.upper() for s in params["symbols"]]
        filtered = filtered[filtered['Symbol'].isin(symbols)]
    
    # Filtrar por político
    if params.get("politicians"):
        pattern = '|'.join(params["politicians"])
        filtered = filtered[filtered['Politician'].str.contains(pattern, case=False, na=False)]
    
    # Filtrar por partido
    if params.get("party"):
        filtered = filtered[filtered['Party'] == params["party"]]
    
    # Filtrar por cámara
    if params.get("chamber"):
        filtered = filtered[filtered['Chamber'] == params["chamber"]]
    
    # Filtrar por acción (compra/venta)
    if params.get("action") == "buy":
        filtered = filtered[filtered['Action'].str.contains('Purchase', case=False, na=False)]
    elif params.get("action") == "sell":
        filtered = filtered[filtered['Action'].str.contains('Sale', case=False, na=False)]
    
    # Filtrar por industria
    if params.get("industry"):
        filtered = filtered[filtered['Industry'].str.contains(params["industry"], case=False, na=False)]
    
    # Filtrar por año
    if params.get("year"):
        filtered = filtered[filtered['year'] == params["year"]]
    
    # Filtrar por trimestre
    if params.get("quarter"):
        filtered = filtered[filtered['quarter'] == params["quarter"]]
    
    # Filtrar por monto mínimo
    if params.get("min_amount"):
        filtered = filtered[filtered['Amount'] >= params["min_amount"]]
    
    limit = params.get("limit", 20)
    aggregation = params.get("aggregation")
    
    # Realizar agregación según tipo
    if aggregation == "by_politician":
        result = filtered.groupby('Politician').agg({
            'Amount': 'sum',
            'Gain/Loss': 'mean',
            'Symbol': 'count',
            'Party': 'first',
            'Chamber': 'first'
        }).rename(columns={'Symbol': 'Num_Trades', 'Gain/Loss': 'Avg_Gain_Loss'})
        result = result.sort_values('Amount', ascending=False).head(limit)
        result_type = "aggregated_by_politician"
        
    elif aggregation == "by_party":
        result = filtered.groupby('Party').agg({
            'Amount': 'sum',
            'Gain/Loss': 'mean',
            'Symbol': 'count',
            'Politician': 'nunique'
        }).rename(columns={'Symbol': 'Num_Trades', 'Politician': 'Num_Politicians', 'Gain/Loss': 'Avg_Gain_Loss'})
        result_type = "aggregated_by_party"
        
    elif aggregation == "by_symbol":
        result = filtered.groupby('Symbol').agg({
            'Amount': 'sum',
            'Gain/Loss': 'mean',
            'Politician': 'count',
            'Industry': 'first'
        }).rename(columns={'Politician': 'Num_Trades', 'Gain/Loss': 'Avg_Gain_Loss'})
        result = result.sort_values('Amount', ascending=False).head(limit)
        result_type = "aggregated_by_symbol"
        
    elif aggregation == "by_industry":
        result = filtered.groupby('Industry').agg({
            'Amount': 'sum',
            'Gain/Loss': 'mean',
            'Symbol': 'nunique',
            'Politician': 'count'
        }).rename(columns={'Symbol': 'Unique_Stocks', 'Politician': 'Num_Trades', 'Gain/Loss': 'Avg_Gain_Loss'})
        result = result.sort_values('Amount', ascending=False).head(limit)
        result_type = "aggregated_by_industry"
    
    elif aggregation == "by_year":
        result = filtered.groupby('year').agg({
            'Amount': 'sum',
            'Gain/Loss': 'mean',
            'Symbol': 'nunique',
            'Politician': 'nunique',
            'Filed Date': 'count'
        }).rename(columns={'Symbol': 'Unique_Stocks', 'Politician': 'Unique_Politicians', 'Filed Date': 'Num_Trades', 'Gain/Loss': 'Avg_Gain_Loss'})
        result = result.sort_index(ascending=False).head(limit)
        result_type = "aggregated_by_year"
        
    elif aggregation == "top_buys":
        buys = filtered[filtered['Action'].str.contains('Purchase', case=False, na=False)]
        result = buys.nlargest(limit, 'Amount')[
            ['Purchase Date', 'year', 'Symbol', 'Company', 'Politician', 'Party', 'Chamber', 'Amount', 'Amount Range', 'Gain/Loss', 'Industry', 'Security']
        ]
        result_type = "top_buys"
        
    elif aggregation == "top_sells":
        sells = filtered[filtered['Action'].str.contains('Sale', case=False, na=False)]
        result = sells.nlargest(limit, 'Amount')[
            ['Purchase Date', 'year', 'Symbol', 'Company', 'Politician', 'Party', 'Chamber', 'Amount', 'Amount Range', 'Gain/Loss', 'Industry', 'Security']
        ]
        result_type = "top_sells"
    
    elif aggregation == "recent":
        result = filtered.sort_values('Filed Date', ascending=False).head(limit)[
            ['Filed Date', 'Purchase Date', 'year', 'Symbol', 'Company', 'Politician', 'Party', 'Chamber', 'Action', 'Amount', 'Amount Range', 'Industry', 'Security']
        ]
        result_type = "recent"
        
    elif aggregation == "summary":
        # Resumen general
        total_buys = filtered[filtered['Action'].str.contains('Purchase', case=False, na=False)]['Amount'].sum()
        total_sells = filtered[filtered['Action'].str.contains('Sale', case=False, na=False)]['Amount'].sum()
        num_transactions = len(filtered)
        unique_politicians = filtered['Politician'].nunique()
        unique_symbols = filtered['Symbol'].nunique()
        
        by_party = filtered.groupby('Party')['Amount'].sum().to_dict()
        by_chamber = filtered.groupby('Chamber')['Amount'].sum().to_dict()
        
        top_politicians = filtered.groupby('Politician')['Amount'].sum().nlargest(5).to_dict()
        top_stocks = filtered.groupby('Symbol')['Amount'].sum().nlargest(5).to_dict()
        top_industries = filtered.groupby('Industry')['Amount'].sum().nlargest(5).to_dict()
        
        avg_gain_loss = filtered['Gain/Loss'].mean()
        
        result = {
            'total_buys': total_buys,
            'total_sells': total_sells,
            'num_transactions': num_transactions,
            'unique_politicians': unique_politicians,
            'unique_symbols': unique_symbols,
            'by_party': by_party,
            'by_chamber': by_chamber,
            'top_politicians': top_politicians,
            'top_stocks': top_stocks,
            'top_industries': top_industries,
            'avg_gain_loss': avg_gain_loss
        }
        result_type = "summary"
        
    else:
        # Sin agregación - devolver transacciones individuales
        sort_by = params.get("sort_by", "date")
        cols = ['Purchase Date', 'year', 'Symbol', 'Company', 'Politician', 'Party', 'Chamber', 'Action', 'Amount', 'Amount Range', 'Gain/Loss', 'Industry', 'Security']
        if sort_by == "amount":
            result = filtered.nlargest(limit, 'Amount')[cols]
        elif sort_by == "gain_loss":
            result = filtered.nlargest(limit, 'Gain/Loss')[cols]
        else:
            result = filtered.sort_values('Filed Date', ascending=False).head(limit)[cols]
        result_type = "transactions"
    
    return {
        "data": result,
        "params": params,
        "total_rows": len(filtered),
        "result_type": result_type
    }


# ============================================
# CONGRESS TRADING - RESPONSE GENERATION
# ============================================
def generate_congress_response(query: str, search_result: dict, client) -> tuple[str, list, str]:
    """Genera respuesta analítica sobre trading del Congreso (93K+ transacciones)"""
    
    result_type = search_result['result_type']
    data = search_result['data']
    total_rows = search_result['total_rows']
    
    # Formatear datos para el contexto
    if result_type == "summary":
        data_context = f"""RESUMEN DE TRADING DEL CONGRESO:
- Total compras: ${data['total_buys']:,.0f}
- Total ventas: ${data['total_sells']:,.0f}
- Número de transacciones: {data['num_transactions']:,}
- Políticos únicos: {data['unique_politicians']}
- Acciones únicas: {data['unique_symbols']}
- Ganancia/Pérdida promedio: {data['avg_gain_loss']:.2f}%

POR PARTIDO:
{json.dumps(data['by_party'], indent=2)}

POR CÁMARA:
{json.dumps(data['by_chamber'], indent=2)}

TOP 5 POLÍTICOS (por volumen):
{json.dumps(data['top_politicians'], indent=2)}

TOP 5 ACCIONES MÁS OPERADAS:
{json.dumps(data['top_stocks'], indent=2)}

TOP 5 INDUSTRIAS:
{json.dumps(data['top_industries'], indent=2)}"""
    elif isinstance(data, pd.DataFrame):
        # Add year summary if available
        if 'year' in data.columns:
            year_summary = data.groupby('year').agg({
                'Amount': ['sum', 'count']
            }).to_string()
            data_str = data.head(40).to_string() if len(data) > 40 else data.to_string()
            data_context = f"RESUMEN POR AÑO:\n{year_summary}\n\nDETALLE DE TRANSACCIONES:\n{data_str}"
            if len(data) > 40:
                data_context += f"\n\n... y {len(data) - 40} registros más"
        else:
            if len(data) > 30:
                data_context = data.head(30).to_string()
                data_context += f"\n\n... y {len(data) - 30} registros más"
            else:
                data_context = data.to_string()
    else:
        data_context = str(data)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Eres un analista experto en trading de políticos del Congreso de Estados Unidos.

INSTRUCCIONES:
1. Analiza los datos proporcionados y responde de forma clara y estructurada
2. **ORGANIZA CRONOLÓGICAMENTE**: Cuando muestres transacciones de un político, agrupa por año (más reciente primero)
3. Usa formato de moneda apropiado (millones con M, miles con K)
4. **USA EL NOMBRE DE LA EMPRESA** cuando esté disponible: "NVDA (NVIDIA)" es mejor que solo "NVDA"
5. Destaca patrones importantes:
   - Evolución temporal de las inversiones
   - Sectores preferidos y cambios a lo largo del tiempo
   - Rendimiento (Gain/Loss) de las operaciones
   - Tipo de instrumento (Stock vs Options)
6. Sé objetivo y neutral políticamente
7. Responde en español
8. Máximo 450 palabras
9. NO inventes datos que no estén en el contexto

FORMATO SUGERIDO PARA TRANSACCIONES:
**2024:**
- NVDA (NVIDIA): $3M - Rendimiento: +36% (Stock)
- AVGO (Broadcom): $3M - Rendimiento: +80% (Stock)

**2023:**
- ...

CAMPOS DISPONIBLES:
- Symbol: Ticker de la acción (ej: AAPL)
- Company: Nombre completo (ej: Apple Inc)
- Amount: Valor estimado (punto medio del rango)
- Amount Range: Rango reportado original (ej: "$1,001 - $15,000")
- Gain/Loss: Rendimiento % de la operación
- Industry: Sector (Information Technology, Health Care, etc.)
- Security: Tipo de instrumento (Stock, ST=Stock, OP=Options)
- Purchase Date: Fecha real de la transacción
- Filed Date: Fecha de reporte al Congreso

Datos desde 2014 hasta 2025."""
                },
                {
                    "role": "user",
                    "content": f"""DATOS DE TRADING DEL CONGRESO:
{data_context}

Total de transacciones en el filtro: {total_rows:,}
Tipo de resultado: {result_type}

PREGUNTA: {query}"""
                }
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        # Preparar fuentes para mostrar
        symbols = []
        parties = []
        
        if isinstance(data, pd.DataFrame):
            # Check columns first
            if 'Symbol' in data.columns:
                symbols = data['Symbol'].unique().tolist()[:6]
            elif 'Politician' in data.columns:
                symbols = data['Politician'].unique().tolist()[:6]
            
            # Check for Party in columns or index
            if 'Party' in data.columns:
                parties = data['Party'].unique().tolist()
            elif data.index.name == 'Party' or (hasattr(data.index, 'names') and 'Party' in data.index.names):
                parties = data.index.get_level_values('Party').unique().tolist() if hasattr(data.index, 'get_level_values') else data.index.unique().tolist()
            
            # If index is Party (for aggregated_by_party), get from index
            if result_type == "aggregated_by_party":
                parties = data.index.tolist()
                
        elif result_type == "summary":
            symbols = list(data['top_stocks'].keys())[:6]
            parties = list(data['by_party'].keys())
        
        sources = [{
            "type": "congress",
            "symbols": symbols,
            "parties": parties,
            "count": total_rows,
            "result_type": result_type
        }]
        
        confidence = "high" if total_rows > 0 else "none"
        
        return answer, sources, confidence
        
    except Exception as e:
        return f"Error generando respuesta: {e}", [], "error"


# ============================================
# BUFFETT LETTERS - QUERY ENHANCEMENT
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
            raise e
        return query


# ============================================
# BUFFETT - RESOLVE FOLLOW-UP QUERY
# ============================================
def resolve_followup(query: str, conversation_history: list, client) -> str:
    """
    Resuelve referencias anafóricas en preguntas de seguimiento.
    """
    if not conversation_history or len(conversation_history) < 2:
        return query
    
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
    
    recent_context = []
    for msg in conversation_history[-4:]:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        content = msg["content"][:300]
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

Responde SOLO con la pregunta reformulada, nada más."""
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
# BUFFETT - HYBRID SEARCH (BM25 + EMBEDDINGS)
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
    """
    
    year_match = re.search(r'\b(19[7-9]\d|20[0-2]\d)\b', query)
    year_filter = year_match.group(1) if year_match else None
    
    query_without_year = re.sub(r'\b(19[7-9]\d|20[0-2]\d)\b', '', query).strip()
    is_year_plus_concept = year_filter and len(query_without_year.split()) > 3
    
    if is_year_plus_concept:
        base_year = int(year_filter)
        valid_years = {str(y) for y in [base_year, base_year + 1, base_year + 2] if 1977 <= y <= 2024}
    else:
        valid_years = {year_filter} if year_filter else None
    
    enhanced_query = enhance_query(query, client)
    q_emb = model.encode([enhanced_query], normalize_embeddings=True, convert_to_numpy=True)
    
    retrieve_k = 30
    sem_scores, sem_indices = index.search(q_emb.astype('float32'), min(retrieve_k, len(chunks)))
    
    sem_results = {}
    max_sem = max(sem_scores[0]) if sem_scores[0].max() > 0 else 1
    for score, idx in zip(sem_scores[0], sem_indices[0]):
        if idx >= 0:
            sem_results[idx] = score / max_sem
    
    bm25_results_raw = bm25.search(enhanced_query, top_k=retrieve_k)
    
    bm25_results = {}
    max_bm25 = bm25_results_raw[0][1] if bm25_results_raw else 1
    for idx, score in bm25_results_raw:
        bm25_results[idx] = score / max_bm25 if max_bm25 > 0 else 0
    
    all_indices = set(sem_results.keys()) | set(bm25_results.keys())
    combined_scores = {}
    
    for idx in all_indices:
        sem_score = sem_results.get(idx, 0)
        bm25_score = bm25_results.get(idx, 0)
        combined_scores[idx] = (semantic_weight * sem_score) + ((1 - semantic_weight) * bm25_score)
    
    sorted_indices = sorted(combined_scores.keys(), key=lambda x: -combined_scores[x])
    
    results = []
    for idx in sorted_indices:
        chunk = chunks[idx].copy()
        chunk['score'] = combined_scores[idx]
        chunk['sem_score'] = sem_results.get(idx, 0)
        chunk['bm25_score'] = bm25_results.get(idx, 0)
        
        if valid_years and chunk['year'] not in valid_years:
            continue
        
        year_count = sum(1 for r in results if r['year'] == chunk['year'])
        if year_count >= 3:
            continue
        
        results.append(chunk)
        if len(results) >= top_k * 2:
            break
    
    return results


# ============================================
# BUFFETT - RERANKING WITH LLM
# ============================================
def rerank_with_llm(query: str, chunks: list[dict], client, top_k: int = 8) -> list[dict]:
    """
    Reordena chunks usando el LLM para evaluar relevancia real.
    """
    if len(chunks) <= top_k:
        return chunks
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:16]):
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
        
        ranking_str = response.choices[0].message.content.strip()
        ranking = [int(x.strip()) for x in re.findall(r'\d+', ranking_str)]
        
        reranked = []
        seen = set()
        for idx in ranking:
            if idx < len(chunks) and idx not in seen:
                reranked.append(chunks[idx])
                seen.add(idx)
        
        for i, chunk in enumerate(chunks):
            if i not in seen and len(reranked) < top_k:
                reranked.append(chunk)
        
        return reranked[:top_k]
    
    except:
        return chunks[:top_k]


# ============================================
# BUFFETT - CONFIDENCE DETECTION
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
# BUFFETT - GENERATE RESPONSE WITH QUOTES
# ============================================
def generate_buffett_response(
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
            raise e
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
    insider_df: pd.DataFrame,
    congress_df: pd.DataFrame,
    conversation_history: list = None
) -> tuple[str, list, str, str]:
    """
    Pipeline completo con routing:
    1. Detectar tipo de consulta (Buffett vs Insider vs Congress)
    2. Routing a pipeline apropiado
    3. Generación de respuesta
    
    Returns: (response, sources, confidence, data_type)
    """
    
    try:
        # 1. Detectar tipo de consulta
        data_type = route_query(query, client)
        
        if data_type == "CONGRESS" and congress_df is not None:
            # Pipeline de Congress Trading
            search_result = search_congress_data(query, congress_df, client)
            response, sources, confidence = generate_congress_response(query, search_result, client)
            return response, sources, confidence, "CONGRESS"
        
        elif data_type == "INSIDER" and insider_df is not None:
            # Pipeline de Insider Trading
            search_result = search_insider_data(query, insider_df, client)
            response, sources, confidence = generate_insider_response(query, search_result, client)
            return response, sources, confidence, "INSIDER"
        
        else:
            # Pipeline de Buffett Letters
            resolved_query = resolve_followup(query, conversation_history or [], client)
            
            candidates = hybrid_search(resolved_query, index, chunks, model, bm25, client, top_k=16)
            
            if len(candidates) > 8:
                reranked = rerank_with_llm(resolved_query, candidates, client, top_k=8)
            else:
                reranked = candidates
            
            response, used_chunks, confidence = generate_buffett_response(
                resolved_query, 
                reranked, 
                client,
                conversation_history
            )
            
            return response, used_chunks, confidence, "BUFFETT"
    
    except Exception as e:
        if "rate" in str(e).lower() or "limit" in str(e).lower():
            return "⏳ Demasiadas consultas. Intenta en unos segundos.", [], "error", "ERROR"
        return f"Error: {str(e)}", [], "error", "ERROR"


# ============================================
# UI HELPERS
# ============================================
def show_sources(results: list, confidence: str = "high", data_type: str = "BUFFETT"):
    """Muestra las fuentes según el tipo de datos"""
    
    if data_type == "CONGRESS":
        # Fuentes de Congress trading
        if not results:
            return
        
        source = results[0] if results else {}
        symbols = source.get("symbols", [])
        parties = source.get("parties", [])
        count = source.get("count", 0)
        result_type = source.get("result_type", "")
        
        # Build pills HTML
        pills_list = []
        for s in symbols[:6]:
            pills_list.append(f'<span class="src congress"><b>{s}</b></span>')
        
        # Party pills
        for p in parties:
            if p == 'D':
                pills_list.append('<span class="src democrat"><b>Demócratas</b></span>')
            elif p == 'R':
                pills_list.append('<span class="src republican"><b>Republicanos</b></span>')
        
        # Count pill
        pills_list.append(f'<span class="src congress"><b>{count:,}</b> txns</span>')
        
        # Type label
        type_labels = {
            "summary": "📊 Resumen",
            "top_buys": "🟢 Top Compras",
            "top_sells": "🔴 Top Ventas",
            "aggregated_by_politician": "👤 Por Político",
            "aggregated_by_party": "🏛️ Por Partido",
            "aggregated_by_symbol": "📈 Por Acción",
            "aggregated_by_industry": "🏭 Por Industria",
            "aggregated_by_year": "📅 Por Año",
            "transactions": "📋 Transacciones",
            "recent": "🕐 Recientes"
        }
        if result_type in type_labels:
            pills_list.append(f'<span class="src">{type_labels[result_type]}</span>')
        
        pills_html = " ".join(pills_list)
        st.markdown(f'<div class="sources">{pills_html}</div>', unsafe_allow_html=True)
    
    elif data_type == "INSIDER":
        # Fuentes de insider trading
        if not results:
            return
        
        source = results[0] if results else {}
        symbols = source.get("symbols", [])
        count = source.get("count", 0)
        result_type = source.get("result_type", "")
        
        pills_list = []
        for s in symbols[:8]:
            pills_list.append(f'<span class="src insider"><b>{s}</b></span>')
        
        pills_list.append(f'<span class="src insider"><b>{count:,}</b> txns</span>')
        
        type_labels = {
            "summary": "📊 Resumen",
            "top_buys": "🟢 Top Compras",
            "top_sells": "🔴 Top Ventas",
            "aggregated_by_company": "📈 Por Empresa",
            "aggregated_by_insider": "👤 Por Insider",
            "aggregated_by_month": "📅 Por Mes",
            "aggregated_by_year": "📅 Por Año",
            "transactions": "📋 Transacciones",
            "recent": "🕐 Recientes"
        }
        if result_type in type_labels:
            pills_list.append(f'<span class="src">{type_labels[result_type]}</span>')
        
        pills_html = " ".join(pills_list)
        st.markdown(f'<div class="sources">{pills_html}</div>', unsafe_allow_html=True)
    
    else:
        # Fuentes de Buffett Letters (original)
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
        pills_list = [f'<span class="src"><b>{y}</b></span>' for y in years]
        
        if confidence in ["low", "medium"]:
            pills_list.append(f'<span class="confidence {confidence}">{indicator} {label}</span>')
        
        pills_html = " ".join(pills_list)
        st.markdown(f'<div class="sources">{pills_html}</div>', unsafe_allow_html=True)


def get_data_badge(data_type: str) -> str:
    """Retorna el badge HTML según el tipo de datos"""
    if data_type == "CONGRESS":
        return '<span class="data-badge congress">🏛️ Congress Trading</span>'
    elif data_type == "INSIDER":
        return '<span class="data-badge insider">📊 Insider Data</span>'
    else:
        return '<span class="data-badge buffett">📚 Buffett Letters</span>'


# ============================================
# MAIN
# ============================================
def main():
    # Cargar recursos
    index = load_index()
    chunks = load_chunks()
    model = load_model()
    client = get_client()
    insider_df = load_insider_data()
    congress_df = load_congress_data()
    
    # Verificar recursos mínimos
    buffett_available = index is not None and chunks is not None
    insider_available = insider_df is not None
    congress_available = congress_df is not None
    
    if not buffett_available and not insider_available and not congress_available:
        st.error("⚠️ No hay datos disponibles. Verifica los archivos de datos.")
        st.stop()
    
    # Construir índice BM25 si hay chunks
    bm25 = build_bm25_index(chunks) if chunks else None
    
    # Estado de sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    
    # HEADER
    if insider_df is not None:
        insider_count = f"{insider_df['Insider Name'].nunique():,}"
        insider_vol = f"${insider_df['Value_C'].sum() / 1e9:.1f}B"
    else:
        insider_count = "0"
        insider_vol = "$0"
    
    if congress_df is not None:
        congress_count = f"{congress_df['Politician'].nunique():,}"
    else:
        congress_count = "0"
    
    buffett_count = "48 cartas" if buffett_available else "N/A"
    
    st.markdown(f"""
    <div class="header">
        <div class="logo">⚡ <span>BQuant</span>ChatBot</div>
        <div class="tagline">Buffett · Insiders · Congress Trading AI</div>
        <div class="meta">
            <span><span class="meta-dot"></span>Online</span>
            <span>📚 {buffett_count}</span>
            <span>📊 {insider_count} insiders</span>
            <span>🏛️ {congress_count} políticos</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # WELCOME
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome">
            <h1>Pregunta sobre Buffett,<br/>Insiders o el Congreso</h1>
            <p>Filosofía inversora + 764K insiders + 93K transacciones del Congreso US</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sugerencias mixtas
        suggestions = [
            ("📚 Filosofía", "¿Cuál es la filosofía de inversión de Buffett?"),
            ("📊 Top Insiders", "¿Cuáles fueron las mayores compras de insiders en 2024?"),
            ("🏛️ Pelosi", "¿Qué acciones ha comprado Nancy Pelosi?"),
            ("📚 Moats", "¿Qué son los moats según Buffett?"),
            ("📊 CEOs", "¿Qué CEOs compraron más acciones de sus empresas?"),
            ("🏛️ Partidos", "Compara el trading de demócratas vs republicanos"),
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
        
        response, used, confidence, data_type = search_and_generate(
            q, index, chunks, model, bm25, client, insider_df, congress_df,
            st.session_state.messages
        )
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "sources": used,
            "confidence": confidence,
            "data_type": data_type
        })
        st.rerun()
    
    # MESSAGES
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
            if msg["role"] == "assistant" and msg.get("data_type"):
                st.markdown(get_data_badge(msg.get("data_type", "BUFFETT")), unsafe_allow_html=True)
            st.write(msg["content"])
            if msg.get("sources"):
                show_sources(msg["sources"], msg.get("confidence", "high"), msg.get("data_type", "BUFFETT"))
    
    # INPUT
    if prompt := st.chat_input("Pregunta sobre Buffett, insiders o el Congreso..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("Analizando..."):
                response, used, confidence, data_type = search_and_generate(
                    prompt, index, chunks, model, bm25, client, insider_df, congress_df,
                    st.session_state.messages
                )
            st.markdown(get_data_badge(data_type), unsafe_allow_html=True)
            st.write(response)
            show_sources(used, confidence, data_type)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "sources": used,
            "confidence": confidence,
            "data_type": data_type
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
