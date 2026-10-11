
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RUTA_DATOS = BASE_DIR / "data" / "datos_transporte.csv"
RUTA_RESULTADOS = BASE_DIR / "resultados_no_supervisado"

RUTA_RESULTADOS.mkdir(exist_ok=True)

datos = pd.read_csv(RUTA_DATOS)

print("=== APRENDIZAJE NO SUPERVISADO ===")
print("\nDatos cargados correctamente")

print("\nCantidad de registros:", len(datos))

print("\nColumnas disponibles:")
print(datos.columns.tolist())

print("\nPrimeros 5 registros:")
print(datos.head())

print("\nValores faltantes:")
print(datos.isnull().sum())


print("\n=== PREPARACIÓN DE LOS DATOS ===")

variables = [
    "hora",
    "dia_semana",
    "lluvia",
    "distancia_km",
    "hora_pico"
]

X = datos[variables].copy()

print("\nVariables seleccionadas:")
print(variables)

print("\nPrimeros 5 registros seleccionados:")
print(X.head())

escalador = StandardScaler()

X_escalado = escalador.fit_transform(X)

print("\nDatos normalizados correctamente")

print("\nDimensiones del conjunto de datos:")
print(X_escalado.shape)

print("\nPrimeros 5 registros normalizados:")
print(X_escalado[:5])

print("\n=== ENTRENAMIENTO DEL MODELO K-MEANS ===")

numero_clusters = 8

modelo_kmeans = KMeans(
    n_clusters=numero_clusters,
    random_state=42,
    n_init=10
)

modelo_kmeans.fit(X_escalado)

grupos = modelo_kmeans.labels_

datos_agrupados = datos.copy()

datos_agrupados["cluster"] = grupos

print("\nModelo K-Means entrenado correctamente")

print("\nCantidad de grupos identificados:")
print(numero_clusters)

print("\nCantidad de recorridos por grupo:")
print(datos_agrupados["cluster"].value_counts().sort_index())

print("\nPrimeros 10 recorridos agrupados:")
print(
    datos_agrupados[
        ["origen", "destino", "hora", "distancia_km", "cluster"]
    ].head(10)
)

ruta_agrupados = RUTA_RESULTADOS / "datos_agrupados.csv"

datos_agrupados.to_csv(
    ruta_agrupados,
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado correctamente:")
print(ruta_agrupados)



print("\n=== EVALUACIÓN DEL MODELO K-MEANS ===")

valores_k = range(2, 9)

inercias = []
silhouettes = []

for k in valores_k:

    modelo_temporal = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo_temporal.fit_predict(X_escalado)

    inercias.append(modelo_temporal.inertia_)

    puntuacion = silhouette_score(
        X_escalado,
        etiquetas
    )

    silhouettes.append(puntuacion)

    print(
        f"K={k} | "
        f"Inercia={modelo_temporal.inertia_:.2f} | "
        f"Silhouette={puntuacion:.4f}"
    )

mejor_indice = silhouettes.index(max(silhouettes))

mejor_k = list(valores_k)[mejor_indice]
mejor_silhouette = silhouettes[mejor_indice]

print("\n=== RESULTADO DE LA EVALUACIÓN ===")

print("Mejor cantidad de clusters:", mejor_k)
print("Mejor Silhouette Score:", round(mejor_silhouette, 4))


plt.figure(figsize=(8, 5))

plt.plot(
    list(valores_k),
    inercias,
    marker="o"
)

plt.title("Método del codo - K-Means")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.xticks(list(valores_k))
plt.grid(True)

ruta_grafica = RUTA_RESULTADOS / "metodo_codo.png"

plt.savefig(
    ruta_grafica,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGráfica del método del codo guardada:")
print(ruta_grafica)




print("\n=== VISUALIZACIÓN DE LOS CLUSTERS ===")

plt.figure(figsize=(10, 6))

grafica = plt.scatter(
    datos_agrupados["hora"],
    datos_agrupados["distancia_km"],
    c=datos_agrupados["cluster"],
    cmap="tab10",
    alpha=0.7,
    s=35
)

plt.title("Agrupamiento de recorridos - K-Means")
plt.xlabel("Hora del recorrido")
plt.ylabel("Distancia del recorrido (km)")

plt.colorbar(
    grafica,
    label="Grupo identificado"
)

plt.grid(True, alpha=0.3)

ruta_clusters = (
    RUTA_RESULTADOS / "grafica_clusters.png"
)

plt.savefig(
    ruta_clusters,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGráfica de clusters guardada correctamente:")
print(ruta_clusters)



print("\n=== ANÁLISIS DE LOS CLUSTERS ===")

perfil_clusters = datos_agrupados.groupby("cluster").agg(
    cantidad_recorridos=("cluster", "size"),
    hora_promedio=("hora", "mean"),
    dia_semana_promedio=("dia_semana", "mean"),
    distancia_promedio_km=("distancia_km", "mean"),
    proporcion_lluvia=("lluvia", "mean"),
    proporcion_hora_pico=("hora_pico", "mean")
).reset_index()

columnas_decimales = [
    "hora_promedio",
    "dia_semana_promedio",
    "distancia_promedio_km",
    "proporcion_lluvia",
    "proporcion_hora_pico"
]

perfil_clusters[columnas_decimales] = (
    perfil_clusters[columnas_decimales].round(2)
)

print("\nCaracterísticas de cada grupo:")
print(perfil_clusters.to_string(index=False))

ruta_perfil = RUTA_RESULTADOS / "perfil_clusters.csv"

perfil_clusters.to_csv(
    ruta_perfil,
    index=False,
    encoding="utf-8-sig"
)

print("\nPerfil de clusters guardado correctamente:")
print(ruta_perfil)

total_agrupados = perfil_clusters["cantidad_recorridos"].sum()

print("\nTotal de recorridos agrupados:", total_agrupados)

assert total_agrupados == len(datos), (
    "Error: la cantidad de recorridos agrupados no coincide"
)

print("Validación de agrupamiento completada correctamente")


print("\n=== GENERACIÓN DEL REPORTE FINAL ===")

ruta_reporte = RUTA_RESULTADOS / "reporte_pruebas.txt"

with open(ruta_reporte, "w", encoding="utf-8") as archivo:

    archivo.write(
        "REPORTE DE PRUEBAS - APRENDIZAJE NO SUPERVISADO\n"
    )
    archivo.write("=" * 55 + "\n\n")

    archivo.write("1. INFORMACIÓN DEL DATASET\n")
    archivo.write(f"Total de registros: {len(datos)}\n")
    archivo.write(f"Variables utilizadas: {', '.join(variables)}\n")
    archivo.write(
        f"Valores faltantes: {datos[variables].isnull().sum().sum()}\n\n"
    )

    archivo.write("2. ALGORITMO UTILIZADO\n")
    archivo.write("Modelo: K-Means\n")
    archivo.write("Normalización: StandardScaler\n")
    archivo.write(f"Cantidad de clusters: {numero_clusters}\n\n")

    archivo.write("3. EVALUACIÓN DEL MODELO\n")

    for k, inercia, silhouette in zip(
        valores_k, inercias, silhouettes
    ):
        archivo.write(
            f"K={k} | Inercia={inercia:.2f} "
            f"| Silhouette={silhouette:.4f}\n"
        )

    archivo.write(
        f"\nMejor cantidad de clusters evaluada: {mejor_k}\n"
    )
    archivo.write(
        f"Mejor Silhouette Score: {mejor_silhouette:.4f}\n\n"
    )

    archivo.write("4. PERFIL DE LOS CLUSTERS\n")
    archivo.write(perfil_clusters.to_string(index=False))
    archivo.write("\n\n")

    archivo.write("5. VALIDACIONES\n")
    archivo.write(
        f"Total de recorridos agrupados: {total_agrupados}\n"
    )
    archivo.write(
        "Coincidencia con el dataset original: "
        f"{total_agrupados == len(datos)}\n"
    )
    archivo.write(
        "Cantidad de grupos generados: "
        f"{datos_agrupados['cluster'].nunique()}\n"
    )

print("\nReporte de pruebas generado correctamente:")
print(ruta_reporte)
