from math import hypot
from heapq import heappop, heappush


grafo = {
    "Portal Norte": {
        "Toberín": 4
    },
    "Toberín": {
        "Portal Norte": 4,
        "Alcalá": 4
    },
    "Alcalá": {
        "Toberín": 4,
        "Calle 100": 5
    },
    "Calle 100": {
        "Alcalá": 5,
        "Calle 72": 5
    },
    "Calle 72": {
        "Calle 100": 5,
        "Calle 45": 4
    },
    "Calle 45": {
        "Calle 72": 4,
        "Calle 26": 5
    },
    "Calle 26": {
        "Calle 45": 5,
        "Avenida Jiménez": 4,
        "CAN": 5
    },
    "Avenida Jiménez": {
        "Calle 26": 4,
        "Ricaurte": 5
    },
    "Ricaurte": {
        "Avenida Jiménez": 5,
        "CAD": 4
    },
    "CAN": {
        "Calle 26": 5,
        "CAD": 5
    },
    "CAD": {
        "Ricaurte": 4,
        "CAN": 5,
        "Marsella": 6
    },
    "Marsella": {
        "CAD": 6,
        "Banderas": 5
    },
    "Banderas": {
        "Marsella": 5,
        "Portal Américas": 5
    },
    "Portal Américas": {
        "Banderas": 5
    }
}



distancias = {
    frozenset(("Portal Norte", "Toberín")): 2.2,
    frozenset(("Toberín", "Alcalá")): 1.8,
    frozenset(("Alcalá", "Calle 100")): 2.4,
    frozenset(("Calle 100", "Calle 72")): 2.8,
    frozenset(("Calle 72", "Calle 45")): 3.0,
    frozenset(("Calle 45", "Calle 26")): 2.7,
    frozenset(("Calle 26", "Avenida Jiménez")): 2.0,
    frozenset(("Avenida Jiménez", "Ricaurte")): 2.3,
    frozenset(("Calle 26", "CAN")): 2.8,
    frozenset(("CAN", "CAD")): 2.0,
    frozenset(("Ricaurte", "CAD")): 1.6,
    frozenset(("CAD", "Marsella")): 3.4,
    frozenset(("Marsella", "Banderas")): 2.7,
    frozenset(("Banderas", "Portal Américas")): 2.5
}


estaciones_transferencia = {
    "Calle 26",
    "Avenida Jiménez",
    "Ricaurte",
    "CAD",
    "CAN",
    "Banderas"
}

estaciones_cerradas = {
    "Avenida Jiménez"
    }


coordenadas = {
    "Portal Norte": (0, 10),
    "Toberín": (0, 9),
    "Alcalá": (0, 8),
    "Calle 100": (0, 7),
    "Calle 72": (0, 6),
    "Calle 45": (0, 5),
    "Calle 26": (0, 4),
    "Avenida Jiménez": (0, 3),
    "Ricaurte": (-1, 2),
    "CAN": (-2, 4),
    "CAD": (-2, 2),
    "Marsella": (-4, 2),
    "Banderas": (-6, 2),
    "Portal Américas": (-8, 2)
}


def heuristica(actual, destino):
    x1, y1 = coordenadas[actual]

    x2, y2 = coordenadas[destino]

    distancia = hypot(
        x2 - x1,
        y2 - y1
    )

    return distancia


def estacion_disponible(estacion):
    return estacion not in estaciones_cerradas

def buscar_ruta(origen, destino):
    if origen not in grafo:

        print(
            f"\nLa estación de origen '{origen}' no existe."
        )

        return None

    if destino not in grafo:

        print(
            f"\nLa estación de destino '{destino}' no existe."
        )

        return None

    if not estacion_disponible(origen):

        print(
            f"\nLa estación de origen '{origen}' está cerrada."
        )

        return None

    if not estacion_disponible(destino):

        print(
            f"\nLa estación de destino '{destino}' está cerrada."
        )

        return None

    if origen == destino:

        return [origen], 0


    frontera = []


    heappush(
        frontera,
        (
            0,
            0,
            origen,
            [origen]
        )
    )


    mejores_costos = {

        origen: 0

    }


    while frontera:
        prioridad, costo_actual, estacion_actual, ruta = heappop(
            frontera
        )

        if estacion_actual == destino:

            return ruta, costo_actual


        for vecino, costo_conexion in grafo[
            estacion_actual
        ].items():


            if not estacion_disponible(vecino):

                continue

            nuevo_costo = (
                costo_actual
                + costo_conexion
            )


            if (
                vecino not in mejores_costos
                or nuevo_costo < mejores_costos[vecino]
            ):

                mejores_costos[vecino] = nuevo_costo


                costo_estimado = heuristica(
                    vecino,
                    destino
                )


                prioridad = (
                    nuevo_costo
                    + costo_estimado
                )


                heappush(
                    frontera,
                    (
                        prioridad,
                        nuevo_costo,
                        vecino,
                        ruta + [vecino]
                    )
                )

    return None


def calcular_distancia(ruta):
    distancia_total = 0


    for i in range(len(ruta) - 1):
        origen = ruta[i]
        destino = ruta[i + 1]

        conexion = frozenset(
            (
                origen,
                destino
            )
        )

        distancia_total += distancias.get(
            conexion,
            0
        )

    return distancia_total


def obtener_transferencias(ruta):
    transferencias = []

    for estacion in ruta:

        if estacion in estaciones_transferencia:

            transferencias.append(
                estacion
            )

    return transferencias

def mostrar_estaciones():
    estaciones = list(
        grafo.keys()
    )

    print(
        "\n========================================"
    )
    print(
        "       ESTACIONES DISPONIBLES"
    )
    print(
        "========================================"
    )

    for numero, estacion in enumerate(
        estaciones,
        start=1
    ):
        print(
            f"{numero}. {estacion}"
        )


    return estaciones


def seleccionar_estacion(
    estaciones,
    mensaje
):
    while True:
        try:
            opcion = int(
                input(
                    f"\n{mensaje}: "
                )
            )

            if 1 <= opcion <= len(estaciones):

                return estaciones[
                    opcion - 1
                ]


            print(
                "\nOpción inválida."
            )

        except ValueError:

            print(
                "\nDebe ingresar un número."
            )


def mostrar_detalle_ruta(ruta):
    print(
        "\n========================================"
    )
    print(
        "       DETALLE DEL RECORRIDO"
    )
    print(
        "========================================"
    )

    for i in range(len(ruta) - 1):
        origen = ruta[i]
        destino = ruta[i + 1]
        tiempo = grafo[
            origen
        ][
            destino
        ]

        conexion = frozenset(
            (
                origen,
                destino
            )
        )

        distancia = distancias.get(
            conexion,
            0
        )

        print(
            f"\n{i + 1}. {origen}"
        )
        print(
            f"   ↓"
        )
        print(
            f"   {destino}"
        )
        print(
            f"   Tiempo: {tiempo} minutos"
        )
        print(
            f"   Distancia: {distancia:.1f} km"
        )


def mostrar_resultado(resultado):
    if resultado is None:

        print(
            "\n========================================"
        )
        print(
            "No existe una ruta disponible."
        )
        print(
            "========================================"
        )
        return

    ruta, tiempo_total = resultado
    distancia_total = calcular_distancia(
        ruta
    )

    transferencias = obtener_transferencias(
        ruta
    )
    numero_estaciones = len(
        ruta
    )
    numero_tramos = (
        numero_estaciones - 1
    )
    print(
        "\n\n========================================"
    )
    print(
        "          RUTA ENCONTRADA"
    )
    print(
        "========================================"
    )
    print(
        "\n"
        + " -> ".join(ruta)
    )

    print(
        "\n----------------------------------------"
    )
    print(
        "RESUMEN DEL VIAJE"
    )
    print(
        "----------------------------------------"
    )
    print(
        f"\nOrigen: {ruta[0]}"
    )

    print(
        f"Destino: {ruta[-1]}"
    )
    print(
        f"Tiempo estimado: {tiempo_total} minutos"
    )
    print(
        f"Distancia aproximada: "
        f"{distancia_total:.1f} km"
    )
    print(
        f"Estaciones recorridas: "
        f"{numero_estaciones}"
    )
    print(
        f"Tramos recorridos: "
        f"{numero_tramos}"
    )

    if transferencias:

        print(
            "\nEstaciones de transferencia:"
        )

        for estacion in transferencias:

            print(
                f" - {estacion}"
            )
    else:

        print(
            "\nNo se requieren estaciones "
            "de transferencia."
        )
    mostrar_detalle_ruta(
        ruta
    )


def main():
    print(
        "\n========================================"
    )
    print(
        " SISTEMA INTELIGENTE DE RUTAS"
    )
    print(
        " Transporte Masivo - Prototipo"
    )
    print(
        "========================================"
    )
    print(
        "\nEl sistema utiliza el algoritmo A*"
    )
    print(
        "para encontrar la mejor ruta."
    )

    estaciones = mostrar_estaciones()
    origen = seleccionar_estacion(
        estaciones,
        "Seleccione la estación de ORIGEN"
    )
    destino = seleccionar_estacion(
        estaciones,
        "Seleccione la estación de DESTINO"
    )

    print(
        "\nCalculando la mejor ruta..."
    )

    resultado = buscar_ruta(
        origen,
        destino
    )
    mostrar_resultado(
        resultado
    )

if __name__ == "__main__":

    main()