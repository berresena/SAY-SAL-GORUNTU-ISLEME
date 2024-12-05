# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import os
import cv2
import numpy as np
from scipy.signal import wiener
from skimage.util import random_noise
from skimage.filters import threshold_otsu
from skimage.morphology import opening, square

# Kayıt klasörünü kontrol et ve yoksa oluştur
save_dir = "/home/username/Desktop/odev"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 1. adim: Renkli bir resmi okuyun
image_path = "/home/username/Downloads/renkli_resim.jpeg"  # Resminizi buraya ekleyin, username kısmına kendi kullanıcı adınızı yazabilirsiniz.
image = cv2.imread(image_path)
cv2.imwrite(os.path.join(save_dir, "adim_1.jpeg"), image)

# 2. adim: Gri tonlu resme çevirin
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(save_dir, "adim_2.jpeg"), gray_image)

# 3. adim: Resmi 256x256 olacak şekilde yeniden boyutlandırın
resized_image = cv2.resize(gray_image, (256, 256))
cv2.imwrite(os.path.join(save_dir, "adim_3.jpeg"), resized_image)

# 4. adim: Gauss gürültüsü ekleyin
noisy_image = random_noise(resized_image, mode='gaussian', var=0.01)
noisy_image = (255 * noisy_image).astype(np.uint8)
cv2.imwrite(os.path.join(save_dir, "adim_4.jpeg"), noisy_image)

# 5. adim: Gürültüyü Mean ve Wiener filtreleriyle temizleyin
mean_filtered = cv2.blur(noisy_image, (5, 5))
cv2.imwrite(os.path.join(save_dir, "adim_5_mean.jpeg"), mean_filtered)

wiener_filtered = wiener(noisy_image, (5, 5))
wiener_filtered = np.uint8(np.clip(wiener_filtered, 0, 255))
cv2.imwrite(os.path.join(save_dir, "adim_5_wiener.jpeg"), wiener_filtered)

# 6. adim: Sobel kenar bulma operatörü kullanarak kenarları bulun
sobel_x = cv2.Sobel(resized_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(resized_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)
sobel_edge = np.uint8(sobel_edge)
cv2.imwrite(os.path.join(save_dir, "adim_6.jpeg"), sobel_edge)

# 7. adim: OTSU'nun metoduyla siyah beyaz resme çevirin
thresh_value = threshold_otsu(resized_image)
binary_image = (resized_image > thresh_value).astype(np.uint8) * 255
cv2.imwrite(os.path.join(save_dir, "adim_7.jpeg"), binary_image)

# 8. adim: Morfolojik açma işlemi uygulayın
morphed_image = opening(binary_image, square(3))
cv2.imwrite(os.path.join(save_dir, "adim_8.jpeg"), morphed_image)

# 9. Adım: Hazır komutlar kullanmadan konvolüsyon işlemi
def apply_convolution(image, kernel):
    # Çekirdeği çevir (konvolüsyon için gerekli)
    kernel = np.flipud(np.fliplr(kernel))
    
    # Görüntü boyutları ve kernel boyutlarını al
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Padleme boyutlarını hesapla
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Görüntüyü sıfırlarla doldur (padding)
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Çıktı görüntüsü
    output = np.zeros_like(image)
    
    # Konvolüsyon işlemi
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    # Değerleri normalize et
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

# Filtre tanımları
mean_kernel = np.ones((3, 3)) / 9  # Mean (ortalama) filtre çekirdeği
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Keskinleştirme çekirdeği
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Kenar bulma çekirdeği

# 9.1: Mean Filtresi (Gürültü Giderme)
mean_filtered_image = apply_convolution(resized_image, mean_kernel)
cv2.imwrite(os.path.join(save_dir, "adim_9_mean.jpeg"), mean_filtered_image)

# 9.2: Keskinleştirme
sharpened_image = apply_convolution(resized_image, sharpen_kernel)
cv2.imwrite(os.path.join(save_dir, "adim_9_sharpen.jpeg"), sharpened_image)

# 9.3: Kenar Bulma
edge_detected_image = apply_convolution(resized_image, edge_kernel)
cv2.imwrite(os.path.join(save_dir, "adim_9_edge.jpeg"), edge_detected_image) 
print("9. Adım başarıyla tamamlandı ve sonuçlar kaydedildi.")

print("Tüm adimlar başarıyla tamamlandı ve sonuçlar kaydedildi.")

