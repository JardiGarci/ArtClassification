import cv2
import numpy as np
from numba import njit
 
def normalize_image(img, max_size=1380, gray = True):
    """
    Normaliza una imagen a max_size x max_size píxeles.
    
    Procedimiento:
    1. Redimensiona preservando relación de aspecto (solo reducción).
    2. Rellena el espacio restante mediante reflexión de bordes.
    
    Parameters
    ----------
    img : ndarray
        Imagen en escala de grises (H, W) o color (H, W, C).
    max_size : int
        Tamaño objetivo. Por defecto 1380, elegido por ser cercano a la
        dimensión mínima del dataset y divisible entre 1, 2, 3 y 4.
    
    Returns
    -------
    img_norm : ndarray
        Imagen de tamaño (max_size, max_size) o (max_size, max_size, C).
    """
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")
 
    # Escalar preservando aspecto (solo reducción)
    if h >= w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
 
    img_resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )
 
    # Padding simétrico por reflexión
    pad_top = (max_size - new_h) // 2
    pad_bottom = max_size - new_h - pad_top
    pad_left = (max_size - new_w) // 2
    pad_right = max_size - new_w - pad_left
 
    img_norm = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )

    if gray == True: img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
 
    return img_norm
 
 
def segment_image(img, grid_size=1):
    """
    Divide una imagen en una cuadrícula de grid_size x grid_size secciones.
    
    Parameters
    ----------
    img : ndarray
        Imagen 2D (H, W) o 3D (H, W, C). Se espera que las dimensiones
        sean divisibles entre grid_size (garantizado si max_size=1380).
    grid_size : int
        Número de divisiones por lado. Valores válidos: 1, 2, 3 o 4,
        que producen 1, 4, 9 o 16 secciones respectivamente.
    
    Returns
    -------
    segments : list of ndarray
        Lista de secciones en orden fila-mayor (izquierda a derecha,
        arriba a abajo).
    """
    if grid_size < 1:
        raise ValueError("grid_size debe ser >= 1.")
 
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")
 
    if h % grid_size != 0 or w % grid_size != 0:
        raise ValueError(
            f"Las dimensiones ({h}, {w}) no son divisibles entre grid_size={grid_size}."
        )
 
    step_h = h // grid_size
    step_w = w // grid_size
    segments = []
 
    for i in range(grid_size):
        for j in range(grid_size):
            segment = img[
                i * step_h : (i + 1) * step_h,
                j * step_w : (j + 1) * step_w
            ]
            segments.append(segment)
 
    return segments
 
 
def preprocess_image(img_path, max_size=1380, grid_sizes=(1, 2, 3, 4)):
    """
    Pipeline completo de preprocesamiento para una imagen.
    
    Parameters
    ----------
    img_path : str
        Ruta a la imagen.
    max_size : int
        Tamaño de normalización.
    grid_sizes : tuple of int
        Tamaños de cuadrícula para segmentación.
    
    Returns
    -------
    results : dict
        Diccionario con clave 'grid_{n}' para cada grid_size,
        donde el valor es una lista de secciones (ndarrays en escala de grises).
    """
    # Leer en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
 
    # Normalizar
    img_norm = normalize_image(img, max_size=max_size)
 
    # Segmentar en cada nivel de partición
    results = {}
    for gs in grid_sizes:
        segments = segment_image(img_norm, grid_size=gs)
        results[f'grid_{gs}'] = segments
 
    return results

# =============================================================================
# Funciones auxiliares
# =============================================================================

def sub_mean(x):
    """Sustrae la media de un arreglo."""
    return x - np.mean(x)


@njit
def bineo(s_min, s_max, degree=1):
    """
    Genera una secuencia de escalas espaciadas logarítmicamente.

    El espaciado se controla mediante raíces progresivas de 2:
    degree=1 usa √2 ≈ 1.414 (espaciado amplio),
    degree=2 usa 2^(1/4) ≈ 1.189 (intermedio),
    degree=3 usa 2^(1/8) ≈ 1.091 (denso, más puntos para regresión).

    Parameters
    ----------
    s_min : int
        Escala mínima en píxeles.
    s_max : int
        Escala máxima en píxeles.
    degree : int
        Controla la densidad del muestreo. Valores mayores generan
        más escalas intermedias.

    Returns
    -------
    N_s : ndarray
        Arreglo de escalas enteras.
    """
    s = s_min
    val = 2
    N_s = []
    for i in range(1, degree + 1):
        val = np.sqrt(val)  # Raíz progresiva: degree=3 → 2^(1/8)
    while s < s_max:
        N_s.append(s)
        s = int(s * val) + 1
    return np.array(N_s)


def valor_cercano(lista, valor):
    """Encuentra el valor más cercano a 'valor' dentro de 'lista'."""
    lista = np.array(lista)
    resta = np.abs(lista - valor)
    return lista[resta == min(resta)]


def vals_Qs(q_n, q_p):
    """
    Genera valores de q con muestreo no uniforme.

    Los valores son más densos cerca de q=0, donde h(q) presenta
    mayor variación, y más espaciados hacia los extremos. Esto mejora
    la resolución del espectro multifractal en la zona de transición
    entre fluctuaciones débiles (q<0) y fuertes (q>0).

    Parameters
    ----------
    q_n : float
        Extremo negativo del rango de momentos (ej. -5.0).
    q_p : float
        Extremo positivo del rango de momentos (ej. 5.0).

    Returns
    -------
    Qs : list of float
        Valores de q ordenados de menor a mayor.
    """
    cuts = ([float(0)]
            + [float(i) for i in bineo(s_min=0.25, s_max=q_p, degree=3)]
            + [float(q_p)])

    l = []
    for i, cut in enumerate(cuts[1:]):
        l += [float(x) for x in np.arange(cuts[i], cut, step=cut / 2)]

    i_min = l.index(valor_cercano(l, np.abs(q_n)))
    i_max = l.index(valor_cercano(l, q_p))

    # Rama negativa (espejo de la positiva)
    l_min = [ll * -1 for ll in l[1:i_min]]
    l_min.reverse()

    l_max = l[:i_max]
    Qs = l_min + l_max
    return Qs


def binomial_cascade_2d(N,p1 = 0.1,p2 =0.2,p3 = 0.3,p4 = 0.4, q_min = -10, q_max = 10, int_img=True):
    mult = np.array([[p1, p2], [p3, p4]])  # Multiplicative probabilities in 2D
    cascade = np.array([[1]])  # Initial state of the cascade
    for _ in range(N):
        X, Y = cascade.shape  # Get the current dimensions of the cascade
        new_cascade = np.zeros([X * 2, Y * 2])  # Create a new matrix for the next iteration
        for x in range(X):
            for y in range(Y):
                # Multiply each cell by the probabilities and expand into a 2x2 matrix
                opera = cascade[x, y] * mult
                new_cascade[x * 2:x * 2 + 2, y * 2:y * 2 + 2] = opera
        cascade = new_cascade  # Update the cascade for the next iteration

        # qs = ut.vals_Qs(q_min,q_max)
        qs = np.arange(q_min - 0.25,q_max + 0.25,0.25)
        Dqs = []
        tau_q = []


        for q in qs:
            
            if q != 1:
                Dq =  np.log(p1**q + p2**q + p3**q + p4**q) / ((1 - q) * np.log(2))
                Tq = Dq * (q-1)
                
            else:
                # Dq =  np.log(p1**q + p2**q + p3**q + p4**q) / np.log(2)
                Dq = -(p1*np.log(p1) + p2*np.log(p2) + p3*np.log(p3) + p4*np.log(p4)) / np.log(2)
                Tq = 0
                

            Dqs.append( Dq )
            tau_q.append( Tq )

        # ---- Espectro multifractal f(α) via transformada de Legendre ----
        alpha = np.gradient(tau_q, qs)       # α(q) = dτ/dq
        f_alpha = qs * alpha - tau_q         # f(α) = q·α - τ(q)

        alpha = alpha[1:-1]
        f_alpha = f_alpha[1:-1]

    data = { 'alpha':alpha, 'f_alpha':f_alpha, 'tau':tau_q, 'qs':qs}
    
    if int_img == True: cascade = np.array(np.round((cascade / np.max(cascade)) * 255), dtype=np.int16)
    return cascade,data



