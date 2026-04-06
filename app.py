from flask import Flask, render_template, request, jsonify
import os
import pickle
from collections import defaultdict
from preprocess import clean_text

app = Flask(__name__)

# Clase Naïve Bayes (versión simplificada para la web)
class NaiveBayesClassifier:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab_size = 0
        self.vocab = {}
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.classes = data['classes']
        self.priors = data['priors']
        self.word_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data['word_counts'].items():
            self.word_counts[k] = defaultdict(int, v)
        self.class_word_totals = defaultdict(int, data['class_word_totals'])
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
    
    def predict_proba(self, tokens):
        import math
        if not tokens:
            return None, {}
        
        best_class = None
        max_log_prob = -float('inf')
        log_probs = {}
        
        for c in self.classes:
            log_prob_sum = self.priors[c]
            
            for word in tokens:
                count_w_c = self.word_counts[c].get(word, 0)
                total_words_c = self.class_word_totals[c]
                prob_w_c = (count_w_c + 1) / (total_words_c + self.vocab_size)
                log_prob_sum += math.log(prob_w_c)
            
            log_probs[c] = log_prob_sum
            if log_prob_sum > max_log_prob:
                max_log_prob = log_prob_sum
                best_class = c
        
        # Normalizar
        exp_probs = {c: math.exp(lp - max_log_prob) for c, lp in log_probs.items()}
        total = sum(exp_probs.values())
        probs = {c: exp_probs[c] / total for c in exp_probs}
        
        return best_class, probs

# Cargar modelo al iniciar
modelo = NaiveBayesClassifier()
model_path = os.path.join(os.path.dirname(__file__), 'modelo_entrenado.pkl')

def get_color_for_category(category):
    colors = {
        'Soporte Técnico': '#FF6B6B',
        'Facturación': '#4ECDC4',
        'Consulta General': '#45B7D1',
        'Queja': '#FFA07A',
        'Cancelación': '#98D8C8'
    }
    return colors.get(category, '#667eea')

if os.path.exists(model_path):
    modelo.load_model(model_path)
    print("[OK] Modelo cargado correctamente")
else:
    print("[WARNING] Modelo no encontrado. Ejecute naive_bayes.py primero")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para clasificar un ticket."""
    try:
        data = request.get_json()
        ticket_text = data.get('ticket_text', '')
        
        if not ticket_text:
            return jsonify({'error': 'No se proporcionó texto'}), 400
        
        # Preprocesar y predecir
        tokens = clean_text(ticket_text)
        categoria, probabilidades = modelo.predict_proba(tokens)
        
        # Preparar respuesta
        response = {
            'categoria': categoria,
            'probabilidades': probabilidades,
            'confianza': max(probabilidades.values()) if probabilidades else 0,
            'tokens_encontrados': len(tokens),
            'color': get_color_for_category(categoria)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar estado del servicio."""
    return jsonify({
        'status': 'ok',
        'modelo_cargado': modelo.classes != [],
        'clases_disponibles': modelo.classes
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)