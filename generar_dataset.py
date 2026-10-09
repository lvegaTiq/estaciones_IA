import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

estaciones = [
    "Portal Norte",
    "Toberín",
    "Alcalá",
    "Calle 100",
    "Calle 72",
    "Calle 45",
    "Calle 26",
    "Avenida Jiménez",
    "Ricaurte",
    "CAN",
    "CAD",
    "Marsella",
    "Banderas",
    "Portal Américas"
]

datos = []

for i in range(1000):

    origen, destino = rng.choice(
        estaciones, size=2, replace=False
    )

    hora = int(rng.integers(5, 24))
    dia_semana = int(rng.integers(0, 7))
    lluvia = int(rng.integers(0, 2))
    distancia = round(float(rng.uniform(1, 25)), 2)

    hora_pico = int(
        (6 <= hora <= 9) or (17 <= hora <= 20)
    )

    puntuacion = 0

    if hora_pico:
        puntuacion += 2

    if dia_semana < 5:
        puntuacion += 1

    if lluvia:
        puntuacion += 1

    if distancia > 15:
        puntuacion += 1

    puntuacion += int(rng.choice(
        [-1, 0, 1], p=[0.15, 0.70, 0.15]
    ))

    if puntuacion <= 1:
        congestion = "Baja"
    elif puntuacion <= 3:
        congestion = "Media"
    else:
        congestion = "Alta"

    datos.append({
        "origen": origen,
        "destino": destino,
        "hora": hora,
        "dia_semana": dia_semana,
        "lluvia": lluvia,
        "distancia_km": distancia,
        "hora_pico": hora_pico,
        "congestion": congestion
    })

df = pd.DataFrame(datos)

carpeta = Path("data")
carpeta.mkdir(exist_ok=True)

ruta_archivo = carpeta / "datos_transporte.csv"

df.to_csv(
    ruta_archivo,
    index=False,
    encoding="utf-8-sig"
)

print("Dataset generado correctamente.")
print(f"Total de registros: {len(df)}")
print("\nPrimeros registros:")
print(df.head())

print("\nDistribución de congestión:")
print(df["congestion"].value_counts())

print(f"\nArchivo guardado en: {ruta_archivo}")
