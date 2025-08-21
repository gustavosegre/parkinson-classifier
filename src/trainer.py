from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib


# Esta função:
# Treina: model.fit(X_train, y_train) = Aqui o MLP ajusta seus pesos internos para aprender a associar os vetores de treino com os rótulos corretos
# Prevê: y_pred = model.predict(X_test) = Usa o modelo treinado para fazer previsões no conjunto de teste
# Avalia: Compara as previsões (y_pred) com os rótulos verdadeiros (y_test) e calcula todas as métricas que estão na saída: o relatório de classificação, a matriz de confusão e a curva ROC
def train_and_evaluate(model, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    
    model.fit(X_train, y_train)

    nome_do_arquivo = './models/rf_model.pkl'

    joblib.dump(model, nome_do_arquivo)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=["Saudável", "Parkinson"]))

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Saudável", "Parkinson"],
                yticklabels=["Saudável", "Parkinson"])
    plt.title("Matriz de Confusão Normalizada")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("Falso Positivo")
            plt.ylabel("Verdadeiro Positivo")
            plt.title("Curva ROC")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Erro ao calcular curva ROC:", e)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": auc if auc is not None else "N/A"
    }

    return pd.DataFrame([metrics])
