import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from typing import Any, Tuple, Optional


def parse_time_to_minutes(time_str: str) -> float | None:
    """
    Convierte tiempos tipo '3:05 h', '2h', '45min', '1 h 20 min' → minutos (float)
    """
    if not isinstance(time_str, str):
        return None

    s = time_str.lower().strip()

    # Formato H:MM
    match = re.match(r"(\d+):(\d+)", s)
    if match:
        h, m = map(int, match.groups())
        return h * 60 + m

    # Formato '3h', '2 h'
    match = re.match(r"(\d+)\s*h", s)
    if match:
        return int(match.group(1)) * 60

    # Formato '45min'
    match = re.match(r"(\d+)\s*min", s)
    if match:
        return int(match.group(1))

    return None


def parse_altitude(text: str) -> float | None:
    """
    Convierte altitudes donde el punto es un separador de miles (Ej: '1.234') → 1234.0
    """
    if not isinstance(text, str) or pd.isna(text):
        return None
        
    s = text.strip()
    
    # 1. Eliminar unidades y signos
    s = s.lower().replace("m", "").replace("+", "").replace("-", "").strip()
    
    # 2. Reemplazar SEPARADORES DE MILES (puntos) por cadena vacía.
    # Esto transforma '1.824' en '1824' o '714.0' en '7140'
    s_cleaned = s.replace(".", "").replace(",", "") 
    
    # 3. La lógica de tu CSV parece a veces usar '714.0' sin ser miles. 
    # Para manejar esto: solo eliminamos el punto si el valor es mayor a 1000
    
    if s_cleaned.isdigit():
        # Aquí se asume que si el valor original tenía un punto, pero NO era una unidad decimal 
        # (como 1.5, sino 1.500), entonces es un separador de miles.
        if '.' in text and float(s_cleaned) > 1000:
             # Si el valor original tenía punto y el número es grande, es un separador de miles.
             # Regresa la versión que limpia el punto para obtener el valor correcto (Ej. 1824)
             return float(s.replace('.', '').replace(',', ''))
        
        # Versión simplificada y segura: eliminamos todos los puntos/comas si no hay un patrón claro.
        return float(s_cleaned)
        
    return None


def parse_distance(text: str) -> float | None:
    """
    Convierte '7 km' → 7.0
    """
    if not isinstance(text, str):
        return None

    match = re.match(r"(\d+[.,]?\d*)\s*km", text.lower())
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def clean_numeric_column(value: Any) -> float | None:
    
    if pd.isna(value):
        return None
        
    valor_str = str(value).strip()
    
    # Expresión regular robusta para encontrar el primer patrón numérico
    RE_NUMERICO_LIMPIO = re.compile(r"([+\-]?\s*\d+[\.,]?\d*)") 
    match = RE_NUMERICO_LIMPIO.search(valor_str)
    
    if match:
        num_str = match.group(1).strip()
        num_str = num_str.replace(",", ".")
        
        # Eliminar el punto como separador de miles si parece ser un número grande
        if num_str.count('.') > 0 and len(num_str.split('.')[-1]) >= 3:
            # Ejemplo: 1.824 -> 1824
            num_str = num_str.replace('.', '')
        
        # Limpiar signos
        num_str = num_str.replace('+', '').replace('-', '').strip()

        try:
            return float(num_str)
        except ValueError:
            return None 
    return None



def parse_desnivel_dual(value) -> Tuple[float | None, float | None]:
    """
    Extrae los componentes positivo y negativo del desnivel acumulado 
    utilizando la función clean_numeric_column.
    
    Ej: '+1.100m / -110m' -> (1100.0, -110.0)
    """
    if pd.isna(value):
        return (None, None)
    
    valor_str = str(value).strip()
    
    # 1. Buscar los dos componentes: un número con '+' y un número con '-'
    # Usamos una regex que busca el patrón de un número con su signo, seguido de una unidad (m).
    # Captura dos números, el primero que viene después del signo '+' (o sin signo) 
    # y el segundo que viene después del signo '-'.
    match = re.search(r"(\+?\s*[\d\.]+m).*(\-\s*[\d\.]+m)", valor_str, re.IGNORECASE)
    
    if match:
        # Grupo 1: Contiene el valor positivo/neto (Ej: "+1.100m")
        positive_component = match.group(1) 
        # Grupo 2: Contiene el valor negativo (Ej: "-110m")
        negative_component = match.group(2)
        
        # Desnivel Positivo (Eliminamos el signo '+' para obtener el valor absoluto)
        pos_val = clean_numeric_column(positive_component.replace('+', '').strip())
        
        # Desnivel Negativo (Eliminamos el signo '-' y lo convertimos a float negativo)
        neg_val_abs = clean_numeric_column(negative_component.replace('-', '').strip())
        
        # 3. Retornar los valores (el negativo debe ser negativo)
        return (pos_val, -neg_val_abs if neg_val_abs is not None else None)
        
    # Manejo de casos donde solo hay un valor (Ej: '565 m' o '1.600m')
    # Si la regex no encuentra el patrón dual, asumimos que el valor es el desnivel total positivo
    pos_val = clean_numeric_column(valor_str)
    return (pos_val, None) # Solo devuelve el positivo

def limpiar_rutas(
    input_csv_path: str,
    output_csv_path: str
):
    """
    Limpieza principal del CSV de rutas.

    - Elimina columnas no necesarias
    - Añade route_id
    - Normaliza pdf_path SOLO si existe físicamente
    - Convierte campos numéricos a formatos útiles para consultas (float)
    """
    df = pd.read_csv(input_csv_path, encoding="utf-8")


    # -----------------------------
    # 1. Normalización numérica
    # -----------------------------
    # Distancia (km)
    if "distancia_total" in df.columns:
        df["distancia_total_km"] = df["distancia_total"].apply(parse_distance)

    if "altitud_minima" in df.columns:
        # Convertir a string para poder eliminar el separador de miles (el punto)
        # y luego aplicar la limpieza numérica, forzando la magnitud.
        df["altitud_minima_m"] = (
        df["altitud_minima"]
            .astype(str)
            .apply(clean_numeric_column)
    )

    if "altitud_maxima" in df.columns:
        df["altitud_maxima_m"] = (
        df["altitud_maxima"]
            .astype(str) # <-- CRÍTICO: Convertir a string antes de limpiar
            .apply(clean_numeric_column)
        )
    

    if "desnivel_acumulado" in df.columns:
        df[["desnivel_acumulado_pos", "desnivel_acumulado_neg"]] = (
        df["desnivel_acumulado"]
         .apply(lambda x: pd.Series(parse_desnivel_dual(x)))
    )

    # Tiempo total → minutos
    if "Tiempo_total" in df.columns:
        df["tiempo_total_min"] = df["Tiempo_total"].apply(parse_time_to_minutes)

    # -----------------------------
    # 5. Guardar resultado
    # -----------------------------
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n✔ CSV limpio generado en: {output_csv_path}")
    print(f"✔ Total rutas: {len(df)}")


# ----------------------------------------------------------
# Ejecutable
# ----------------------------------------------------------

if __name__ == "__main__":
    limpiar_rutas(
        input_csv_path="./staged/rutas.csv",
        output_csv_path="./staged/rutas_v2.csv"
    )
