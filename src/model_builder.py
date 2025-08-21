from sklearn.neural_network import MLPClassifier

def build_mlp() -> MLPClassifier:
    """
    Constrói um modelo MLP (Multilayer Perceptron) com hiperparâmetros definidos.

    Returns:
        MLPClassifier: Classificador MLP configurado com uma camada escondida e ativação ReLU.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        alpha=0.0001,
        solver='adam',
        learning_rate='constant',
        max_iter=500,
        random_state=42
    )

# Multilayer Perceptron (MLP) = Uma pequena rede neural
# Entrada: receberá os vetores de 2048 características
# Camada oculta: terá uma camada intermediária com 100 neurônios (hidden_layer_sizes=(100,)) que aprenderão a encontrar padrões nesses 2048 números
# Saída: produzirá a classificação final (0 ou 1) 