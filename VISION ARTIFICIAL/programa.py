import cv2
import pytesseract
import numpy as np
import os
import time

# Configuración de Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Crear la carpeta para guardar las imágenes si no existe
output_folder = "capturas_placas"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def capturar_imagen():
    cam = cv2.VideoCapture(0)
    placa_detectada = False
    tiempo_inicio = None
    numero_placa = ""

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error al capturar la imagen")
            break

        # Convertir la imagen al espacio de color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filtramos solo el color amarillo
        amarillo_bajo = np.array([20, 100, 100], dtype="uint8")
        amarillo_alto = np.array([30, 255, 255], dtype="uint8")
        mascara_amarilla = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)

        # Aplicamos morfología para eliminar ruido
        kernel = np.ones((5, 5), np.uint8)
        mascara_amarilla = cv2.morphologyEx(mascara_amarilla, cv2.MORPH_OPEN, kernel)

        # Encontramos los contornos de las áreas amarillas
        contornos, _ = cv2.findContours(mascara_amarilla, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Recorremos los contornos buscando algo con forma de placa (rectangular)
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

            if len(approx) == 4:  # Forma rectangular
                x, y, w, h = cv2.boundingRect(approx)

                # Aseguramos que el rectángulo sea de tamaño razonable
                if w > 100 and h > 40:  # Ajusta estos valores según el tamaño de la placa
                    placa = frame[y:y+h, x:x+w]

                    # Dibujar un rectángulo alrededor de la placa detectada
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Procesar y detectar el número de la placa
                    numero = detectar_numero(placa)

                    # Si se detecta una placa válida
                    if len(numero) == 6:
                        numero_placa = numero
                        placa_detectada = True
                        tiempo_inicio = time.time()

                        # Mostrar número en la terminal
                        print(f"Placa detectada: {numero_placa}")

                        # Guardar la imagen en la carpeta designada
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        cv2.imwrite(f"{output_folder}/placa_{numero_placa}_{timestamp}.png", placa)

        # Mostrar el número de la placa y el color en la parte superior de la imagen
        if placa_detectada:
            cv2.putText(frame, f"Placa: {numero_placa} Color: Amarillo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Si han pasado 30 segundos, reiniciamos la detección
            if time.time() - tiempo_inicio > 30:
                placa_detectada = False
                numero_placa = ""

        cv2.imshow("Camara con Reconocimiento de Placa", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def detectar_numero(placa):
    # Convertir la imagen de la placa a escala de grises
    gris = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para resaltar los números negros sobre el fondo amarillo
    umbral = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Aplicar morfología para limpiar la imagen
    kernel = np.ones((3, 3), np.uint8)
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)

    # Usar Tesseract para extraer texto
    texto = pytesseract.image_to_string(umbral, config='--psm 8')

    # Filtrar solo letras y números
    texto_filtrado = ''.join([c for c in texto if c.isalnum() or c == '-'])
    return texto_filtrado.strip() if len(texto_filtrado) == 6 else "No se detectó bien"

if __name__ == "__main__":
    capturar_imagen()
