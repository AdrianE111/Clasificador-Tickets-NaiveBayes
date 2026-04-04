import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK (solo la primera vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Limpia, tokeniza y lematiza el texto."""
    if not isinstance(text, str):
        return []
    
    # Minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^a-z\s]', '', text)
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminación de stopwords, palabras cortas y lematización
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    return cleaned_tokens

def build_vocabulary(corpus_tokens):
    """Construye el vocabulario (Bag of Words)."""
    vocab = set()
    for tokens in corpus_tokens:
        vocab.update(tokens)
    return {word: idx for idx, word in enumerate(vocab)}

def load_and_preprocess_data(csv_path):
    """Carga el dataset y mapea a las 5 categorías del proyecto."""
    print(f"Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Columnas disponibles: {df.columns.tolist()}")
    
    # Detectar columnas automáticamente
    desc_col = None
    type_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'description' in col_lower or 'text' in col_lower or 'body' in col_lower:
            desc_col = col
        if 'type' in col_lower or 'category' in col_lower or 'issue' in col_lower:
            type_col = col
    
    if desc_col and type_col:
        df = df[[desc_col, type_col]].dropna()
        df.columns = ['Ticket Description', 'Ticket Type']
    else:
        # Si no encuentra, usar nombres comunes
        print("Buscando columnas alternativas...")
        possible_desc = ['Ticket Description', 'Description', 'Text', 'Body', 'Content']
        possible_type = ['Ticket Type', 'Type', 'Category', 'Issue Type', 'Class']
        
        for col in possible_desc:
            if col in df.columns:
                desc_col = col
                break
        for col in possible_type:
            if col in df.columns:
                type_col = col
                break
        
        if desc_col and type_col:
            df = df[[desc_col, type_col]].dropna()
            df.columns = ['Ticket Description', 'Ticket Type']
        else:
            raise Exception(f"No se encontraron columnas esperadas. Columnas: {df.columns.tolist()}")
    
    # Mapeo a las categorías en español
    category_map = {
        'Technical issue': 'Soporte Técnico',
        'Technical Issue': 'Soporte Técnico',
        'technical': 'Soporte Técnico',
        'Billing inquiry': 'Facturación',
        'Billing': 'Facturación',
        'billing': 'Facturación',
        'Product inquiry': 'Consulta General',
        'Product': 'Consulta General',
        'General': 'Consulta General',
        'Refund request': 'Queja',
        'Refund': 'Queja',
        'Complaint': 'Queja',
        'Cancellation request': 'Cancelación',
        'Cancellation': 'Cancelación',
        'Cancel': 'Cancelación'
    }
    
    # Aplicar mapeo (búsqueda parcial)
    def map_category(ticket_type):
        if pd.isna(ticket_type):
            return None
        ticket_type_str = str(ticket_type)
        for key, value in category_map.items():
            if key.lower() in ticket_type_str.lower():
                return value
        return None
    
    df['Category'] = df['Ticket Type'].apply(map_category)
    df = df.dropna(subset=['Category'])
    
    print(f"\nDistribución de categorías:")
    print(df['Category'].value_counts())
    
    print("\nProcesando tickets...")
    df['cleaned_tokens'] = df['Ticket Description'].apply(clean_text)
    
    # Filtrar tickets vacíos
    df = df[df['cleaned_tokens'].apply(len) > 0]
    
    vocab = build_vocabulary(df['cleaned_tokens'])
    print(f"Vocabulario construido: {len(vocab)} palabras únicas")
    
    return df, vocab

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'data', 'archive', 'customer_support_tickets.csv')
    
    try:
        df_procesado, vocabulario = load_and_preprocess_data(csv_file)
        print(f"\n✅ Éxito: {len(df_procesado)} tickets listos para entrenar.")
        print(f"📚 Vocabulario: {len(vocabulario)} palabras.")
        print(f"📊 Ejemplo de tokens: {df_procesado['cleaned_tokens'].iloc[0][:10]}...")
    except Exception as e:
        print(f"❌ Error: {e}")