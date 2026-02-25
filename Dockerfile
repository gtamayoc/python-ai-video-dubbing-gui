# Usa una imagen oficial, estable y ligera de Python 3.10
FROM python:3.10-slim

# Evita que Python escriba archivos .pyc en el disco (ahorrando espacio y evitando problemas caché)
ENV PYTHONDONTWRITEBYTECODE=1

# Evita que Python haga buffering de stdout y stderr (útil para ver logs en tiempo real)
ENV PYTHONUNBUFFERED=1

# Instala las dependencias del sistema operativo:
# - build-essential: Para compilar ciertas dependencias de Python (como llama-cpp)
# - ffmpeg: Requerido por moviepy, edge-tts y pydub para manejo de audio/video
# - tk / tcl / libx11-6 / libxext6: Requerido para la interfaz gráfica (CustomTkinter / tkinter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    tk \
    tcl \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Define el directorio de trabajo donde residirá el código dentro del contenedor
WORKDIR /app

# Copia solo los requerimientos primero para aprovechar la memoria caché de las capas de Docker.
# Si los requerimientos no cambian, Docker reutilizará la instalación previa de pip install.
COPY requirements.txt .

# Actualiza pip y ejecuta la instalación de las dependencias de Python de forma eficiente
# Nota: Esta instalación prioriza CPU/Portabilidad por defecto, sin hacks extraños.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente al contenedor (se ejecutará cada vez que se cambie el código)
COPY . .

# Define el comando principal o ENTRYPOINT para ejecutar la aplicación
# (Dado que es una app GUI, al ejecutar 'docker run' deberás mandar la variable de entorno DISPLAY para ver la app)
CMD ["python", "main.py"]
