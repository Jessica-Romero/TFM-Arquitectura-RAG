import pandas as pd
from collections import Counter

df = pd.read_csv('rutas_pirineos_secciones2.csv')

text = " ".join(df['text'].astype(str))

weird_chars = [c for c in text if ord(c) > 126]
Counter(weird_chars).most_common(100)
print(Counter(weird_chars).most_common(100))