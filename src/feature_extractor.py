from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model

def get_features_extractor() -> Model:
    """
    Retorna uma instância do modelo ResNet50 pré-treinado sem a camada de classificação final. \n
    Utiliza pooling global average para gerar vetores de características.

    Returns:
        Model: Modelo Keras ResNet50 com saída de pooling 'avg'
    """
    # ResNet50(weights="imagenet", ...) = Está carregando o modelo ResNet50 que já foi pré-treinado pela Google com milhões de imagens (o dataset "ImageNet")
    # include_top=False = A ResNet50 original termina com uma camada que classifica imagens em 1000 categorias (cães, gatos, carros, etc.). O `false` remove essa camada final de classificação.
    # pooling="avg" = Sem a camada de cima, a saída do ResNet50 seria um grande mapa de características. O código simplifica isso calculando a média de cada mapa de características, resultando em um único vetor de 2048 números.
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract(img_path: str, model: Model, preprocess_func, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extrai um vetor de características de uma imagem usando o modelo fornecido.

    Args:
        img_path (str): Caminho da imagem.
        model (Model): Modelo CNN (ex: ResNet50).
        preprocess_func (function): Função de pré-processamento (ex: preprocess_input).
        target_size (tuple): Tamanho para redimensionar a imagem (default: (224, 224)).

    Returns:
        np.ndarray: Vetor de características extraído da imagem (shape: [2048,])
    """
    # Realiza o pré-processamento (carregar, redimentsionar para 224x224, etc.) e, em seguida, usa o extractor_model.predict() para obter o vetor de 2048 características.
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = preprocess_func(np.expand_dims(x, axis=0))
    features = model.predict(x, verbose=0)
    return features[0]
