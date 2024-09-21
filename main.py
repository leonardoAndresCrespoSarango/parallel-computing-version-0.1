import numpy as np
import pycuda.autoinit
from pycuda import driver, compiler

# Definir el tamaño de la matriz y el tamaño del bloque
N = 2  # Tamaño de la matriz
BLOCK_SIZE = 16  # Tamaño del bloque

# Definir el código del kernel
kernel_code = """
__global__ void Bloques(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row < %(N)s && col < %(N)s)
    {
        c[row * %(N)s + col] = a[row * %(N)s + col] + b[row * %(N)s + col];
    }
}
__global__ void Hilos(float *a, float *b, float *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < %(N)s * %(N)s)
    {
        c[index] = a[index] + b[index];
    }
}

__global__ void BlockAndThread(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    if (row < %(N)s && col < %(N)s)
    {
        int index = row * %(N)s + col;
        c[index] = a[index] + b[index];
    }
}

""" % {'N': N}

# Compila el kernel
mod = compiler.SourceModule(kernel_code)
bloques = mod.get_function("Bloques")
hilos = mod.get_function("Hilos")
total=mod.get_function("BlockAndThread")

# Crea las matrices en el host
a = np.random.randn(N, N).astype(np.float32)
b = np.random.randn(N, N).astype(np.float32)

# Crea  las matrices en el dispositivo
a_gpu = driver.mem_alloc(a.nbytes)
b_gpu = driver.mem_alloc(b.nbytes)
c_gpu = driver.mem_alloc(a.nbytes)
cThread_gpu = driver.mem_alloc(a.nbytes)
cTotal_gpu = driver.mem_alloc(a.nbytes)
# Copia  las matrices al dispositivo
driver.memcpy_htod(a_gpu, a)
driver.memcpy_htod(b_gpu, b)

# Configura la cuadrícula y los bloques
grid = (int(np.ceil(N / float(BLOCK_SIZE))), int(np.ceil(N / float(BLOCK_SIZE))), 1)
block = (BLOCK_SIZE, BLOCK_SIZE, 1)

# Configurar la cuadrícula y los hilos
gridThread = (int(np.ceil(N * N / 1024.0)), 1)
blockThread = (1024, 1, 1)

# Crea eventos para medir el tiempo
start = driver.Event()
end = driver.Event()

# Llama al kernel para bloques y mide el tiempo
start.record()
bloques(a_gpu, b_gpu, c_gpu, block=block, grid=grid)
end.record()
end.synchronize()
time_bloques = start.time_till(end) * 1e-3  # Convertir a segundos

# Llama al kernel para hilos y mide el tiempo
start.record()
hilos(a_gpu, b_gpu, cThread_gpu, block=blockThread, grid=gridThread)
end.record()
end.synchronize()
time_hilos = start.time_till(end) * 1e-3  # Convertir a segundos

# Llama al kernel para bloques e hilos y mide el tiempo
start.record()
total(a_gpu, b_gpu, cTotal_gpu, block=block, grid=grid)
end.record()
end.synchronize()
time_total = start.time_till(end) * 1e-3  # Convertir a segundos

# Crea una matriz vacía para el resultado
c = np.empty_like(a)
cThread=np.empty_like(a)
cTotal = np.empty_like(a)

# Copia el resultado al host
driver.memcpy_dtoh(c, c_gpu)
driver.memcpy_dtoh(cThread,cThread_gpu)
driver.memcpy_dtoh(cTotal,cTotal_gpu)

# Imprime el resultado con bloques
print("Resultado de la suma usando bloques")
print(c)
print("Tiempo de ejecución: ", time_bloques, "segundos")

# Imprime el resultado con hilos
print("Resultado de la suma usando hilos")
print(cThread)
print("Tiempo de ejecución: ", time_hilos, "segundos")

# Imprime el resultado con bloques e hilos
print("Resultado de la suma usando bloques e hilos")
print(cTotal)
print("Tiempo de ejecución: ", time_total, "segundos")
