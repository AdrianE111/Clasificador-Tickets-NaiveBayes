import os
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from preprocess import load_and_preprocess_data, clean_text
from naive_bayes import NaiveBayesClassifier

class ModelEvaluator:
    """
    Evaluador de modelos con K-Folds Cross Validation.
    Calcula todas las métricas requeridas:
    - Accuracy
    - Precision por clase
    - Recall por clase
    - F1-Score por clase
    - Macro F1
    - Matriz de Confusión
    """
    
    def __init__(self, k=5):
        self.k = k
        self.results = {}
    
    def confusion_matrix(self, y_true, y_pred, classes):
        """Construye la matriz de confusión manualmente."""
        # Crear diccionario de índices
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        n = len(classes)
        matrix = np.zeros((n, n), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            if true in class_to_idx and pred in class_to_idx:
                matrix[class_to_idx[true]][class_to_idx[pred]] += 1
        
        return matrix, class_to_idx
    
    def calculate_metrics(self, confusion_mat, class_to_idx):
        """
        Calcula Precision, Recall, F1-Score por clase de forma corregida.
        """
        metrics = {}
        n = len(confusion_mat)
        # Suma total de todos los elementos de la matriz
        total_sum = np.sum(confusion_mat)
        
        for i, (cls_name, idx) in enumerate(class_to_idx.items()):
            tp = confusion_mat[i][i]
            # Suma de la columna i (predichos como i) menos el TP
            fp = np.sum(confusion_mat[:, i]) - tp
            # Suma de la fila i (reales de i) menos el TP
            fn = np.sum(confusion_mat[i, :]) - tp
            # El resto son los True Negatives
            tn = total_sum - tp - fp - fn
            
            # Precision = TP / (TP + FP)
            precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall = TP / (TP + FN)
            recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 = 2 * (P * R) / (P + R)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[cls_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        
        return metrics
    
    def accuracy(self, confusion_mat):
        """Calcula el accuracy global."""
        total = np.sum(confusion_mat)
        correct = np.trace(confusion_mat)
        return correct / total if total > 0 else 0
    
    def macro_f1(self, metrics):
        """Calcula el Macro F1 (promedio de F1 por clase)."""
        f1_scores = [m['f1'] for m in metrics.values()]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    def k_fold_cross_validation(self, df, vocab):
        """
        Implementa K-Folds Cross Validation manualmente.
        
        Args:
            df: DataFrame con los datos
            vocab: Diccionario de vocabulario
        """
        print("\n" + "="*60)
        print(f"K-FOLD CROSS VALIDATION (k={self.k})")
        print("="*60)
        
        # Obtener todos los índices
        indices = list(range(len(df)))
        np.random.seed(42)  # Para reproducibilidad
        np.random.shuffle(indices)
        
        # Dividir en k folds
        fold_size = len(indices) // self.k
        folds = []
        for i in range(self.k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.k - 1 else len(indices)
            folds.append(indices[start:end])
        
        # Almacenar resultados por fold
        all_predictions = []
        all_truths = []
        fold_metrics = []
        
        for fold_idx in range(self.k):
            print(f"\n{'='*40}")
            print(f"FOLD {fold_idx + 1}/{self.k}")
            print(f"{'='*40}")
            
            # Dividir entrenamiento y prueba
            test_indices = folds[fold_idx]
            train_indices = [idx for i in range(self.k) if i != fold_idx for idx in folds[i]]
            
            train_df = df.iloc[train_indices]
            test_df = df.iloc[test_indices]
            
            print(f"   Entrenamiento: {len(train_df)} muestras")
            print(f"   Prueba: {len(test_df)} muestras")
            
            # Entrenar modelo
            model = NaiveBayesClassifier()
            model.train(train_df, vocab)
            
            # Predecir en test
            predictions = []
            truths = []
            
            for _, row in test_df.iterrows():
                tokens = row['cleaned_tokens']
                predicted_class, _ = model.predict(tokens)
                true_class = row['Category']
                
                predictions.append(predicted_class)
                truths.append(true_class)
            
            # Calcular métricas para este fold
            classes = model.classes
            conf_mat, class_to_idx = self.confusion_matrix(truths, predictions, classes)
            acc = self.accuracy(conf_mat)
            metrics = self.calculate_metrics(conf_mat, class_to_idx)
            mf1 = self.macro_f1(metrics)
            
            print(f"\n   📊 Resultados Fold {fold_idx + 1}:")
            print(f"      Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"      Macro F1: {mf1:.4f}")
            
            fold_metrics.append({
                'fold': fold_idx + 1,
                'accuracy': acc,
                'macro_f1': mf1,
                'metrics': metrics,
                'confusion_matrix': conf_mat,
                'classes': classes
            })
            
            all_predictions.extend(predictions)
            all_truths.extend(truths)
        
        # Calcular promedios
        avg_accuracy = sum(m['accuracy'] for m in fold_metrics) / self.k
        avg_macro_f1 = sum(m['macro_f1'] for m in fold_metrics) / self.k
        
        # Matriz de confusión global
        global_conf_mat, global_class_to_idx = self.confusion_matrix(
            all_truths, all_predictions, 
            list(set(all_truths + all_predictions))
        )
        global_metrics = self.calculate_metrics(global_conf_mat, global_class_to_idx)
        
        # Mostrar resultados finales
        print("\n" + "="*60)
        print("RESULTADOS FINALES DE K-FOLD CROSS VALIDATION")
        print("="*60)
        
        print(f"\n📊 PROMEDIOS (sobre {self.k} folds):")
        print(f"   Accuracy promedio: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"   Macro F1 promedio: {avg_macro_f1:.4f}")
        
        print(f"\n📊 MÉTRICAS POR CLASE (global):")
        print("-" * 50)
        print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 50)
        
        for cls_name, m in global_metrics.items():
            print(f"{cls_name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
        
        # Matriz de confusión formateada
        print("\n📊 MATRIZ DE CONFUSIÓN GLOBAL:")
        print("-" * 50)
        
        # Encabezados
        classes_list = list(global_class_to_idx.keys())
        print(f"{'Real \\ Predicho':<20}", end="")
        for cls in classes_list:
            print(f"{cls[:12]:<12}", end="")
        print()
        
        for i, true_cls in enumerate(classes_list):
            print(f"{true_cls:<20}", end="")
            for j in range(len(classes_list)):
                print(f"{global_conf_mat[i][j]:<12}", end="")
            print()
        
        # Análisis de confusión
        print("\n📊 ANÁLISIS DE CONFUSIÓN:")
        print("-" * 50)
        
        # Encontrar pares más confundidos
        confusion_pairs = []
        for i in range(len(classes_list)):
            for j in range(len(classes_list)):
                if i != j and global_conf_mat[i][j] > 0:
                    confusion_pairs.append((classes_list[i], classes_list[j], global_conf_mat[i][j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("   Pares más confundidos (verdadero -> predicho):")
        for true_cls, pred_cls, count in confusion_pairs[:5]:
            print(f"   • {true_cls} → {pred_cls}: {count} casos")
        
        # Análisis de varianza entre folds
        accuracies = [m['accuracy'] for m in fold_metrics]
        variance = np.var(accuracies)
        std_dev = np.std(accuracies)
        
        print(f"\n📊 ESTABILIDAD DEL MODELO:")
        print(f"   Desviación estándar de accuracy: {std_dev:.4f}")
        print(f"   Varianza: {variance:.6f}")
        
        self.results = {
            'avg_accuracy': avg_accuracy,
            'avg_macro_f1': avg_macro_f1,
            'fold_metrics': fold_metrics,
            'global_confusion_matrix': global_conf_mat,
            'global_metrics': global_metrics,
            'classes': classes_list,
            'std_accuracy': std_dev
        }
        
        return self.results
    
    def print_detailed_analysis(self):
        """Imprime un análisis detallado de los resultados."""
        if not self.results:
            print("No hay resultados para analizar. Ejecute k_fold_cross_validation primero.")
            return
        
        print("\n" + "="*60)
        print("ANÁLISIS DETALLADO DEL MODELO")
        print("="*60)
        
        print("\n🔍 ¿Qué clases se predicen mejor?")
        metrics = self.results['global_metrics']
        
        best_class = max(metrics.items(), key=lambda x: x[1]['f1'])
        worst_class = min(metrics.items(), key=lambda x: x[1]['f1'])
        
        print(f"   ✅ Mejor clase: {best_class[0]} (F1 = {best_class[1]['f1']:.4f})")
        print(f"   ❌ Peor clase: {worst_class[0]} (F1 = {worst_class[1]['f1']:.4f})")
        
        print("\n🔍 Interpretación:")
        print(f"   • Precisión alta = pocos falsos positivos")
        print(f"   • Recall alto = pocos falsos negativos")
        print(f"   • Si precisión es baja pero recall alto → muchas falsas alarmas")
        print(f"   • Si recall es bajo pero precisión alta → el modelo es conservador")
        
        print("\n🔍 Recomendaciones para mejorar:")
        if self.results['std_accuracy'] > 0.05:
            print("   • Alta varianza entre folds → posiblemente más datos o mejor preprocesamiento")
        else:
            print("   • Baja varianza → el modelo es estable y generaliza bien")
        
        if self.results['avg_accuracy'] < 0.8:
            print("   • Accuracy bajo (<80%) → considerar:")
            print("     - Aumentar el tamaño del dataset")
            print("     - Mejorar la limpieza de datos")
            print("     - Ajustar el parámetro de Laplace Smoothing")
            print("     - Agregar más características (n-gramas)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, 'data', 'archive', 'customer_support_tickets.csv')
    
    try:
        # Cargar datos
        df, vocab = load_and_preprocess_data(csv_file)
        
        # Evaluar con K-Folds
        evaluator = ModelEvaluator(k=5)
        results = evaluator.k_fold_cross_validation(df, vocab)
        evaluator.print_detailed_analysis()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()