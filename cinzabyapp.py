from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"

response = requests.get(url)
img = Image.open(BytesIO(response.content))

imagem_cinza = img.convert('L')
limiar = 128

def binarizar(pixel):
    return 255 if pixel >= limiar else 0

imagem_binaria = imagem_cinza.point(binarizar)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Imagem Colorida")
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Imagem em Cinza")
plt.imshow(imagem_cinza, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Imagem Binarizada")
plt.imshow(imagem_binaria, cmap='gray')
plt.axis('off')

plt.show()
