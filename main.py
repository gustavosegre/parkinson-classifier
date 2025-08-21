# Entry point for the Parkinson Classifier project
import joblib
import cv2   
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Carrega o classificador treinado
rf_model = joblib.load('./models/rf_model.pkl')

# Carrega o extrator de características sem a camada final de classificação
feature_extractor = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def processar_imagem(caminho_imagem):
    # Carrega a imagem do disco
    img = cv2.imread(caminho_imagem)
    
    # Keras espera imagens no formato de cor RGB, mas OpenCV carrega em BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # A ResNet50 foi treinada com imagens de 224x224 pixels
    img = cv2.resize(img, (224, 224))
    
    # Faz os ajustes finais (como normalização de pixels) que a ResNet50 espera
    img = preprocess_input(img)
    
    # O modelo espera um "lote" (batch) de imagens. Adicionamos uma dimensão extra.
    # O shape passa de (224, 224, 3) para (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

def prever_probabilidade_parkinson(caminho_imagem):
    """
    Função completa que recebe o caminho de uma imagem e retorna a probabilidade
    de indicar Parkinson.
    """

    imagem_processada = processar_imagem(caminho_imagem)
    
    features = feature_extractor.predict(imagem_processada)
    
    # Usa o Random Forest para prever as probabilidades nas características extraídas
    # O predict_proba retorna um array com as probs para cada classe, ex: [[prob_saudavel, prob_parkinson]]
    probabilidades = rf_model.predict_proba(features)
    
    # Probabilidade da classe "1" (Parkinson)
    prob_parkinson = probabilidades[0][1]
    
    return prob_parkinson

if __name__ == '__main__':

    caminho_da_imagem_de_teste = './data/test/parkinson/V01PE01.png'
    
    try:
        probabilidade_final = prever_probabilidade_parkinson(caminho_da_imagem_de_teste)
        
        print(f"A probabilidade de indicar Parkinson para a imagem '{caminho_da_imagem_de_teste}' é de: {probabilidade_final * 100:.2f}%")
        
    except FileNotFoundError:
        print(f"\nERRO: O arquivo de imagem não foi encontrado em '{caminho_da_imagem_de_teste}'.")
        print("Por favor, verifique se o caminho está correto e o arquivo existe.")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")