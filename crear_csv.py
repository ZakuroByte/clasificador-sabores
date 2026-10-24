import os, random
import pandas as pd

RUTA = "food-101"          # carpeta descomprimida
POR_CLASE = 100            # fotos por categoría
SABORES = {"A": "acido", "M": "amargo", "D": "dulce", "P": "picante", "S": "salado"}

MAPA = {
 "apple_pie":"D","baby_back_ribs":"SD","baklava":"D","beef_carpaccio":"SA","beef_tartare":"SA",
 "beet_salad":"DA","beignets":"D","bibimbap":"SP","bread_pudding":"D","breakfast_burrito":"SP",
 "bruschetta":"SA","caesar_salad":"SA","cannoli":"D","caprese_salad":"SA","carrot_cake":"D",
 "ceviche":"AS","cheesecake":"D","cheese_plate":"S","chicken_curry":"SP","chicken_quesadilla":"S",
 "chicken_wings":"SP","chocolate_cake":"DM","chocolate_mousse":"DM","churros":"D","clam_chowder":"S",
 "club_sandwich":"S","crab_cakes":"S","creme_brulee":"D","croque_madame":"S","cup_cakes":"D",
 "deviled_eggs":"S","donuts":"D","dumplings":"S","edamame":"S","eggs_benedict":"S","escargots":"S",
 "falafel":"S","filet_mignon":"S","fish_and_chips":"S","foie_gras":"S","french_fries":"S",
 "french_onion_soup":"S","french_toast":"D","fried_calamari":"S","fried_rice":"S","frozen_yogurt":"DA",
 "garlic_bread":"S","gnocchi":"S","greek_salad":"SA","grilled_cheese_sandwich":"S","grilled_salmon":"S",
 "guacamole":"SA","gyoza":"S","hamburger":"S","hot_and_sour_soup":"SAP","hot_dog":"S",
 "huevos_rancheros":"SP","hummus":"S","ice_cream":"D","lasagna":"S","lobster_bisque":"S",
 "lobster_roll_sandwich":"S","macaroni_and_cheese":"S","macarons":"D","miso_soup":"S","mussels":"S",
 "nachos":"SP","omelette":"S","onion_rings":"S","oysters":"S","pad_thai":"SDA","paella":"S",
 "pancakes":"D","panna_cotta":"D","peking_duck":"SD","pho":"S","pizza":"S","pork_chop":"S",
 "poutine":"S","prime_rib":"S","pulled_pork_sandwich":"SD","ramen":"S","ravioli":"S",
 "red_velvet_cake":"D","risotto":"S","samosa":"SP","sashimi":"S","scallops":"S","seaweed_salad":"SA",
 "shrimp_and_grits":"S","spaghetti_bolognese":"S","spaghetti_carbonara":"S","spring_rolls":"S",
 "steak":"S","strawberry_shortcake":"D","sushi":"S","tacos":"SP","takoyaki":"S","tiramisu":"DM",
 "tuna_tartare":"SA","waffles":"D",
}

random.seed(42)
filas = []
for clase, letras in MAPA.items():
    archivos = os.listdir(f"{RUTA}/images/{clase}")
    for f in random.sample(archivos, POR_CLASE):
        fila = {"imagen": f"{clase}/{f}"}
        for letra, nombre in SABORES.items():
            fila[nombre] = int(letra in letras)
        filas.append(fila)

df = pd.DataFrame(filas)
df.to_csv("etiquetas.csv", index=False)
print(df[list(SABORES.values())].sum())   # revisa el balance
