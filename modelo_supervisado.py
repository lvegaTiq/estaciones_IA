
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay)
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/datos_transporte.csv")

print("=== INFORMACIÓN DEL DATASET ===")
print("Registros:", len(df))
print(df.head())

X = df.drop(columns=["congestion"])
y = df["congestion"]

categoricas = ["origen", "destino"]
numericas = [
    "hora",
    "dia_semana",
    "lluvia",
    "distancia_km",
    "hora_pico"
]

preprocesador = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            categoricas
        ),
        ("numericas", "passthrough", numericas)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

modelo = Pipeline([
    ("preprocesador", preprocesador),
    (
        "clasificador",
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    )
])

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)

exactitud = accuracy_score(y_test, predicciones)

print("\n=== RESULTADOS DEL MODELO ===")
print(f"Exactitud: {exactitud:.2%}")

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        predicciones,
        zero_division=0
    )
)


reporte = classification_report(
    y_test,
    predicciones,
    zero_division=0
)

matriz = confusion_matrix(
    y_test,
    predicciones,
    labels=modelo.classes_
)

with open(
    "resultados/reporte_pruebas.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write("PRUEBAS DEL MODELO SUPERVISADO\n")
    archivo.write("=" * 40 + "\n\n")
    archivo.write(f"Total de registros: {len(df)}\n")
    archivo.write(f"Entrenamiento: {len(X_train)}\n")
    archivo.write(f"Pruebas: {len(X_test)}\n")
    archivo.write(f"Exactitud: {exactitud:.2%}\n\n")
    archivo.write("REPORTE DE CLASIFICACIÓN\n")
    archivo.write(reporte)
    archivo.write("\nMATRIZ DE CONFUSIÓN\n")
    archivo.write(str(matriz))

print("\nReporte de pruebas guardado correctamente.")



Path("resultados").mkdir(exist_ok=True)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicciones,
    cmap="Blues"
)

plt.title("Matriz de confusión - Congestión")
plt.tight_layout()
plt.savefig(
    "resultados/matriz_confusion.png",
    dpi=200
)
plt.close()

clasificador = modelo.named_steps["clasificador"]

nombres_variables = (
    modelo.named_steps["preprocesador"]
    .get_feature_names_out()
)

plt.figure(figsize=(24, 12))

plot_tree(
    clasificador,
    feature_names=nombres_variables,
    class_names=list(clasificador.classes_),
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Árbol de decisión - Transporte masivo")
plt.savefig(
    "resultados/arbol_decision.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

joblib.dump(
    modelo,
    "resultados/modelo_congestion.pkl"
)

nuevo_viaje = pd.DataFrame([{
    "origen": "Portal Norte",
    "destino": "Toberín",
    "hora": 7,
    "dia_semana": 0,
    "lluvia": 1,
    "distancia_km": 2.5,
    "hora_pico": 1
}])

resultado = modelo.predict(nuevo_viaje)[0]

print("\n=== PREDICCIÓN DE EJEMPLO ===")
print("Origen: Portal Norte")
print("Destino: Toberín")
print("Hora: 7:00 a. m.")
print("Lluvia: Sí")
print("Congestión predicha:", resultado)

print("\nArchivos guardados en resultados/")
