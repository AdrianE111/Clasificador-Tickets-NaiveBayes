import math
import os
import pickle
from collections import defaultdict
import pandas as pd

class NaiveBayesClassifier:
    """
    Clasificador Naïve Bayes Multinomial implementado desde cero.
    Incluye:
    - Laplace Smoothing para evitar probabilidades cero
    - Suma de logaritmos para evitar underflow numérico
    - Persistencia del modelo (guardar/cargar)
    """
    
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab_size = 0
        self.vocab = {}
    
    def train(self, df, vocab):
        """
        Entrena el modelo Naïve Bayes.
        
        Args:
            df: DataFrame con columnas 'Category' y 'cleaned_tokens'
            vocab: Diccionario de vocabulario {palabra: indice}
        """
        print("\n" + "="*50)
        print("INICIANDO ENTRENAMIENTO DE NAIVE BAYES")
        print("="*50)
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        total_docs = len(df)
        
        # 1. Calcular probabilidades a priori P(c)
        class_counts = df['Category'].value_counts().to_dict()
        self.classes = list(class_counts.keys())
        
        print(f"\nClases encontradas: {self.classes}")
        print(f"Total de documentos: {total_docs}")
        
        for c in self.classes:
            self.priors[c] = math.log(float(class_counts[c]) / float(total_docs))
            print(f"   P({c}) = log({class_counts[c]}/{total_docs}) = {self.priors[c]:.4f}")
        
        # 2. Contar frecuencias de palabras por clase
        print("\nContando frecuencias de palabras por clase...")
        for _, row in df.iterrows():
            category = row['Category']
            for word in row['cleaned_tokens']:
                self.word_counts[category][word] += 1
                self.class_word_totals[category] += 1
        
        # Mostrar estadísticas
        print("\nEstadísticas de entrenamiento:")
        for c in self.classes:
            print(f"   {c}: {self.class_word_totals[c]} palabras totales, {len(self.word_counts[c])} palabras únicas")
        
        print("\n[OK] Entrenamiento finalizado!")
        print("="*50)
    
    def predict(self, tokens):
        """
        Predice la clase de un documento usando suma de logaritmos.
        
        Args:
            tokens: Lista de tokens (palabras) del documento
            
        Returns:
            best_class: Clase con mayor probabilidad
            probabilities: Diccionario con log-probabilidades por clase
        """
        if not tokens:
            return None, {}
        
        best_class = None
        max_log_prob = -float('inf')
        log_probs = {}
        
        for c in self.classes:
            # log(P(c)) - ya está en log
            log_prob_sum = self.priors[c]
            
            for word in tokens:
                count_w_c = self.word_counts[c].get(word, 0)
                total_words_c = self.class_word_totals[c]
                
                # Laplace Smoothing: (count + 1) / (total + vocab_size)
                prob_w_c = (count_w_c + 1) / (total_words_c + self.vocab_size)
                
                # Suma de logaritmos
                log_prob_sum += math.log(prob_w_c)
            
            log_probs[c] = log_prob_sum
            
            if log_prob_sum > max_log_prob:
                max_log_prob = log_prob_sum
                best_class = c
        
        return best_class, log_probs
    
    def predict_proba(self, tokens):
        """
        Predice y devuelve las probabilidades normalizadas (no en log).
        
        Args:
            tokens: Lista de tokens del documento
            
        Returns:
            best_class: Clase más probable
            probabilities: Diccionario con probabilidades normalizadas (0-1)
        """
        best_class, log_probs = self.predict(tokens)
        
        if not log_probs:
            return None, {}
        
        # Convertir de log a probabilidad normalizada
        # Encontrar el máximo para estabilidad numérica
        max_log = max(log_probs.values())
        exp_probs = {c: math.exp(lp - max_log) for c, lp in log_probs.items()}
        total = sum(exp_probs.values())
        
        probs = {c: exp_probs[c] / total for c in exp_probs}
        
        return best_class, probs
    
    def save_model(self, filepath):
        """Guarda el modelo entrenado en un archivo pickle."""
        data = {
            'classes': self.classes,
            'priors': self.priors,
            'word_counts': {k: dict(v) for k, v in self.word_counts.items()},
            'class_word_totals': dict(self.class_word_totals),
            'vocab_size': self.vocab_size,
            'vocab': self.vocab
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"\n[OK] Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Carga un modelo entrenado desde un archivo pickle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.classes = data['classes']
        self.priors = data['priors']
        self.word_counts = defaultdict(lambda: defaultdict(int))
        for class_name, word_dict in data['word_counts'].items():
            self.word_counts[class_name] = defaultdict(int)
            for word, count in word_dict.items():
                self.word_counts[class_name][word] = int(count)
        self.class_word_totals = defaultdict(int)
        for class_name, total in data['class_word_totals'].items():
            self.class_word_totals[class_name] = int(total)
        self.vocab_size = int(data['vocab_size'])
        self.vocab = data['vocab']
        
        print(f"\n[OK] Modelo cargado desde: {filepath}")
        print(f"   Clases: {self.classes}")
        print(f"   Vocabulario: {self.vocab_size} palabras")

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data, clean_text
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'data', 'archive', 'customer_support_tickets.csv')
    
    try:
        # 1. Cargar y preprocesar datos
        df, vocab = load_and_preprocess_data(csv_file)
        
        # 2. Entrenar modelo
        modelo = NaiveBayesClassifier()
        modelo.train(df, vocab)
        
        # 3. Guardar modelo
        modelo.save_model(os.path.join(base_dir, 'modelo_entrenado.pkl'))
        
        # 4. Prueba en vivo
        test_tickets = [
            "My computer won't turn on, I need technical support immediately",
            "I was charged twice for my subscription this month, please fix my billing",
            "How do I use the new dashboard? Just a general question about features",
            "The product arrived damaged and broken, I want to complain and get a refund",
            "Please cancel my account and delete all my personal data"
        ]
        
        print("\n" + "="*50)
        print("PRUEBAS DE CLASIFICACION")
        print("="*50)
        
        for ticket in test_tickets:
            tokens = clean_text(ticket)
            clase, probs = modelo.predict_proba(tokens)
            print(f"\nTicket: {ticket[:60]}...")
            print(f"   Prediccion: {clase}")
            print(f"   Confianza: {max(probs.values())*100:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()