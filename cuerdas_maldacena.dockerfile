# Dockerfile
FROM ubuntu:22.04

# Evita preguntas interactivas durante build
ENV DEBIAN_FRONTEND=noninteractive

# Instala Python y herramientas básicas
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configura Python 3.11 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Instala Julia (REQUERIDO por PySR)
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz && \
    tar -xvzf julia-1.10.4-linux-x86_64.tar.gz && \
    mv julia-1.10.4 /opt/julia && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia-1.10.4-linux-x86_64.tar.gz

# Copia el código
WORKDIR /app
COPY . .

# Instala dependencias Python
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    h5py \
    torch==2.9.1 \
    pysr==1.5.9 \
    juliacall==0.9.26

# Precompila paquetes Julia (evita delays en primera ejecución)
RUN python3 -c "import juliacall; jl = juliacall.newmodule('Precompile'); jl.seval('using SymbolicRegression')"

# Comando por defecto
CMD ["python3", "-c", "print('Entorno CUERDAS-Maldacena listo')"]