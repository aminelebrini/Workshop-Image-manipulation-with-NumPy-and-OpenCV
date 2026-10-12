# Importation des bibliothèques nécessaires
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger une image avec OpenCV
# Remplace 'sample.jpg' par le chemin de ton image
image_path = 'sample.jpg'
image_bgr = cv2.imread(image_path)

# Si l'image n'existe pas, créons une image de test synthétique
if image_bgr is None:
    print("Image non trouvée. Création d'une image synthétique de test...")
    image_bgr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

# Examiner la représentation avec NumPy
print('--- ÉTAPE 1 : EXPLORATION ---')
print(f'Shape (Hauteur, Largeur, Canaux) : {image_bgr.shape}')
print(f'Type de données (dtype) : {image_bgr.dtype}')
print(f'Valeur Min pixel : {image_bgr.min()}')
print(f'Valeur Max pixel : {image_bgr.max()}')

# Affichage avec Matplotlib (conversion BGR -> RGB indispensable)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.title(f'Image Originale\nDimensions: {image_bgr.shape[1]}x{image_bgr.shape[0]} (HxLxC: {image_bgr.shape})')
plt.axis('off')
plt.show()

print('--- ÉTAPE 2 : MANIPULATION ---')

# 1. Redimensionnement (Resize)
img_resized = cv2.resize(image_bgr, (224, 224))

# 2. Recadrage (Crop) : découper une région d'intérêt [ymin:ymax, xmin:xmax]
h, w, _ = image_bgr.shape
img_cropped = image_bgr[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

# 3. Conversion en niveaux de gris
img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 4. Rotation / Retournement (Flip horizontal)
img_flipped = cv2.flip(image_bgr, 1)

# Visualisation des transformations
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Redimensionnée (224x224)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Recadrée (Crop)')
axes[0, 1].axis('off')

axes[1, 0].imshow(img_gray, cmap='gray')
axes[1, 0].set_title('Niveaux de gris')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Retournée (Flip horizontal)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()


print('--- ÉTAPE 3 : PRÉTRAITEMENT DEEP LEARNING ---')

# 1. Redimensionner vers une taille commune
target_size = (128, 128)
prep_img = cv2.resize(image_bgr, target_size)
prep_img = cv2.cvtColor(prep_img, cv2.COLOR_BGR2RGB)

# 2. Normalisation des pixels de [0, 255] vers [0, 1] (Conversion en float32)
prep_img = prep_img.astype(np.float32) / 255.0

# 3. Vérification de la forme et des valeurs
print(f'Forme après redimensionnement : {prep_img.shape}')
print(f'Nouveaux min/max : {prep_img.min():.2f} / {prep_img.max():.2f}')
print(f'Nouveau dtype : {prep_img.dtype}')

# 4. Ajout d\'une dimension batch (Format attendu : [Batch_Size, Height, Width, Channels])
tensor_img = np.expand_dims(prep_img, axis=0)
print(f'Forme finale du tenseur (Batch) : {tensor_img.shape}')

print('--- ÉTAPE 4 : CAS PRATIQUE ---')

def process_pipeline(img_path):
    # 1. Charger l\'image
    img = cv2.imread(img_path)
    if img is None:
        img = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)
        
    # 2. Afficher ses dimensions
    print(f'Dimensions initiales : {img.shape}')
    
    # 3. La redimensionner
    img_res = cv2.resize(img, (64, 64))
    
    # 4. La normaliser
    img_norm = img_res.astype(np.float32) / 255.0
    
    # 5. Appliquer une transformation (ici en gris normalisée)
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    img_gray_norm = img_gray.astype(np.float32) / 255.0
    
    # 6. Afficher le résultat final (Avant / Après)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray_norm, cmap='gray')
    plt.title('Transformée (Gris + Normalisée)')
    plt.axis('off')
    
    plt.show()
    
    return img_gray_norm

# Exécution du pipeline du cas pratique
final_result = process_pipeline(image_path)