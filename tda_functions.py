"""
TDA: Análisis Topológico de Datos para imágenes 2D.

Este módulo implementa el cálculo de homología persistente sobre una
filtración cubical por niveles de gris, incluyendo:
- Filtración de H0 (componentes conexas) con 8-conectividad
- Filtración de H1 (ciclos) via dualidad de Alexander con 4-conectividad
- Extracción de descriptores topológicos: entropía de persistencia,
  estadísticas de tiempos de vida y conteo de características

La implementación usa Union-Find con compresión de caminos, compilado
con numba para rendimiento en imágenes de alta resolución.

Referencia principal:
    - Avilés-Rodríguez et al. (2021). Topological Data Analysis for
      Eye Fundus Image Quality Assessment.
    - Edelsbrunner & Harer (2010). Computational Topology.
"""

import numpy as np
from numba import njit


# =============================================================================
# Union-Find con compresión de caminos
# =============================================================================

@njit
def find(parent, x):
    """
    Encuentra la raíz del componente al que pertenece x,
    con compresión de caminos para eficiencia amortizada O(α(n)).

    Parameters
    ----------
    parent : ndarray
        Arreglo de padres de la estructura Union-Find.
    x : int
        Índice del elemento a buscar.

    Returns
    -------
    int
        Raíz del componente.
    """
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Compresión de caminos
        x = parent[x]
    return x


@njit
def union(parent, birth, death, a, b, level):
    """
    Fusiona dos componentes aplicando la regla del más viejo.

    El componente con nacimiento más temprano (más viejo) sobrevive,
    y el más joven muere en el nivel de filtración actual. Esto
    garantiza que el tiempo de vida refleje la persistencia real.

    Parameters
    ----------
    parent : ndarray
        Arreglo de padres.
    birth : ndarray
        Nivel de nacimiento de cada componente.
    death : ndarray
        Nivel de muerte de cada componente (-1 si aún vivo).
    a, b : int
        Índices de los elementos a fusionar.
    level : int
        Nivel de filtración actual (momento de la muerte).
    """
    ra = find(parent, a)
    rb = find(parent, b)

    if ra == rb:
        return

    if birth[ra] <= birth[rb]:
        parent[rb] = ra
        death[rb] = level
    else:
        parent[ra] = rb
        death[ra] = level


# =============================================================================
# Descriptores topológicos
# =============================================================================

@njit
def entropy_persistence(lifetimes):
    """
    Calcula la entropía de persistencia normalizada.

    E = -Σ p_i · log(p_i) / log(n)

    donde p_i = l_i / Σl_j es la proporción del tiempo de vida
    de la i-ésima característica. La normalización por log(n) acota
    el valor entre 0 y 1.

    Un valor bajo indica que pocas estructuras dominan el diagrama
    (imagen con organización simple). Un valor alto indica una
    distribución uniforme de tiempos de vida (complejidad en
    múltiples escalas).

    Parameters
    ----------
    lifetimes : ndarray
        Tiempos de vida (d_i - b_i) de cada característica.

    Returns
    -------
    float
        Entropía de persistencia normalizada en [0, 1].
    """
    total = 0.0
    n = 0
    for i in range(lifetimes.shape[0]):
        if lifetimes[i] > 0:
            total += lifetimes[i]
            n += 1

    if total == 0.0 or n <= 1:
        return 0.0

    H = 0.0
    for i in range(lifetimes.shape[0]):
        li = lifetimes[i]
        if li > 0:
            p = li / total
            H -= p * np.log(p)

    return H / np.log(n)


@njit
def compute_lifetime_stats(lifetimes):
    """
    Calcula estadísticas de la distribución de tiempos de vida.

    Parameters
    ----------
    lifetimes : ndarray
        Tiempos de vida (d_i - b_i) de cada característica.

    Returns
    -------
    mean_life : float
        Media de los tiempos de vida.
    std_life : float
        Desviación estándar de los tiempos de vida.
    max_life : float
        Tiempo de vida máximo (estructura más persistente).
    n_features : int
        Número de características con vida > 0.
    """
    # Filtrar tiempos de vida positivos
    n = 0
    for i in range(lifetimes.shape[0]):
        if lifetimes[i] > 0:
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0

    # Media
    total = 0.0
    max_life = 0.0
    for i in range(lifetimes.shape[0]):
        li = lifetimes[i]
        if li > 0:
            total += li
            if li > max_life:
                max_life = li
    mean_life = total / n

    # Desviación estándar
    var_acc = 0.0
    for i in range(lifetimes.shape[0]):
        li = lifetimes[i]
        if li > 0:
            diff = li - mean_life
            var_acc += diff * diff
    std_life = np.sqrt(var_acc / n)

    return mean_life, std_life, max_life, n


# =============================================================================
# Imagen invertida para dualidad de Alexander
# =============================================================================

@njit
def inverse(img):
    """
    Calcula la imagen complementaria: I_inv = max(I) - I.

    Los ciclos (H1) en la imagen original corresponden a componentes
    conexas (H0) en la imagen invertida (dualidad de Alexander).

    Parameters
    ----------
    img : ndarray (H, W)
        Imagen en escala de grises.

    Returns
    -------
    new_img : ndarray (H, W)
        Imagen invertida.
    """
    img_shape = img.shape
    max_gray = np.max(img)
    new_img = np.zeros(shape=img_shape, dtype=np.int64)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            val = max_gray - img[i, j]
            if val < 0:
                val = -val
            new_img[i, j] = val
    return new_img


# =============================================================================
# Filtración cubical y homología persistente
# =============================================================================

@njit
def tda_fast(img, step=5):
    """
    Calcula la homología persistente H0 y H1 de una imagen mediante
    filtración cubical por niveles de gris.

    Procedimiento:
    1. Recorre los niveles de gris de menor a mayor con paso 'step'.
    2. Para H0 (componentes conexas, 8-conectividad):
       - Cada píxel sin vecinos activos NACE como nueva componente.
       - Cada píxel con vecinos se FUSIONA (el más viejo sobrevive).
    3. Para H1 (ciclos, via dualidad de Alexander):
       - Simultáneamente, aplica el mismo procedimiento sobre la
         imagen invertida con 4-conectividad.
       - Al final, invierte los tiempos (b,d) a la escala original.
    4. Extrae descriptores de ambos diagramas de persistencia.

    Parameters
    ----------
    img : ndarray (H, W)
        Imagen en escala de grises (uint8 o similar).
    step : int
        Paso de discretización de los niveles de gris.
        step=1: filtración completa (256 niveles, más lento).
        step=5: filtración gruesa (~51 niveles, más rápido).

    Returns
    -------
    features : dict-like tuple
        (H0_entropy, H0_mean, H0_std, H0_max, H0_n,
         H1_entropy, H1_mean, H1_std, H1_max, H1_n,
         H0_lifetimes, H1_lifetimes)

        Los primeros 10 valores son los descriptores escalares.
        Los últimos 2 son los arreglos de tiempos de vida completos,
        útiles para visualización de diagramas de persistencia.
    """
    H, W = img.shape
    max_gray = np.max(img)
    levels = np.arange(-step, max_gray + step, step, dtype=np.int64)
    K = len(levels) - 1
    neg_img = inverse(img)
    max_labels = H * W

    # --- Estructuras Union-Find para H0 ---
    labels = np.zeros((H + 2, W + 2), dtype=np.int64)
    parent = np.zeros(max_labels + 1, dtype=np.int64)
    birth = np.zeros(max_labels + 1, dtype=np.int64)
    death = -np.ones(max_labels + 1, dtype=np.int64)
    n_label = 0

    # --- Estructuras Union-Find para H1 (imagen invertida) ---
    # Se inicializa con un marco de borde activo (label=1) que
    # representa la componente conexa del "exterior" de la imagen.
    # Esto es necesario para la dualidad: los ciclos de la imagen
    # original se detectan como fusiones con este marco exterior.
    labels_h1 = np.zeros((H + 2, W + 2), dtype=np.int64)
    labels_h1[0, :] = 1
    labels_h1[-1, :] = 1
    labels_h1[:, 0] = 1
    labels_h1[:, -1] = 1
    parent_h1 = np.zeros(max_labels + 1, dtype=np.int64)
    parent_h1[1] = 1
    birth_h1 = np.zeros(max_labels + 1, dtype=np.int64)
    birth_h1[1] = -1  # El marco exterior nace antes de la filtración
    death_h1 = -np.ones(max_labels + 1, dtype=np.int64)
    n_label_h1 = 1

    # --- Filtración principal ---
    for k in range(K):
        level = levels[k]
        level_up = levels[k + 1]

        for i in range(H):
            for j in range(W):

                # === H0: componentes conexas (8-conectividad) ===
                if level < img[i, j] <= level_up:
                    neighbors = np.zeros(8, dtype=np.int64)
                    c = 0

                    lbl = labels[i, j]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i + 1, j]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i + 2, j]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i, j + 1]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i + 2, j + 1]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i, j + 2]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i + 1, j + 2]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1
                    lbl = labels[i + 2, j + 2]
                    if lbl > 0: neighbors[c] = find(parent, lbl); c += 1

                    if c == 0:
                        n_label += 1
                        labels[i + 1, j + 1] = n_label
                        parent[n_label] = n_label
                        birth[n_label] = level_up
                    else:
                        root = neighbors[0]
                        for t in range(1, c):
                            union(parent, birth, death, root, neighbors[t], level_up)
                        labels[i + 1, j + 1] = find(parent, root)

                # === H1: ciclos via dualidad (4-conectividad) ===
                if level - 1 <= neg_img[i, j] < level_up - 1:
                    neighbors_h1 = np.zeros(4, dtype=np.int64)
                    c_h1 = 0

                    lbl = labels_h1[i + 1, j]
                    if lbl > 0: neighbors_h1[c_h1] = find(parent_h1, lbl); c_h1 += 1
                    lbl = labels_h1[i, j + 1]
                    if lbl > 0: neighbors_h1[c_h1] = find(parent_h1, lbl); c_h1 += 1
                    lbl = labels_h1[i + 2, j + 1]
                    if lbl > 0: neighbors_h1[c_h1] = find(parent_h1, lbl); c_h1 += 1
                    lbl = labels_h1[i + 1, j + 2]
                    if lbl > 0: neighbors_h1[c_h1] = find(parent_h1, lbl); c_h1 += 1

                    if c_h1 == 0:
                        n_label_h1 += 1
                        labels_h1[i + 1, j + 1] = n_label_h1
                        parent_h1[n_label_h1] = n_label_h1
                        birth_h1[n_label_h1] = level_up - 1
                    else:
                        root_h1 = neighbors_h1[0]
                        for t in range(1, c_h1):
                            union(parent_h1, birth_h1, death_h1,
                                  root_h1, neighbors_h1[t], level_up - 1)
                        labels_h1[i + 1, j + 1] = find(parent_h1, root_h1)

    # =================================================================
    # Extracción de descriptores del diagrama de persistencia
    # =================================================================

    # --- H0: componentes conexas ---
    h0_birth = birth[1:n_label + 1]
    h0_death = death[1:n_label + 1]
    h0_death[h0_death == -1] = max_gray  # Componentes que nunca mueren
    h0_lifetimes = h0_death - h0_birth

    h0_entropy = entropy_persistence(h0_lifetimes)
    h0_mean, h0_std, h0_max, h0_n = compute_lifetime_stats(h0_lifetimes)

    # --- H1: ciclos (transformar de vuelta a escala original) ---
    # En la imagen invertida, birth y death están invertidos.
    # Transformación: b_original = max_gray - d_invertido + step
    #                 d_original = max_gray - b_invertido + step
    # Se excluye label=1 (marco exterior) y características con vida=0
    raw_birth_h1 = death_h1[2:n_label_h1 + 1]
    raw_death_h1 = birth_h1[2:n_label_h1 + 1]

    final_birth_list = []
    final_death_list = []
    for b_inv, d_inv in zip(raw_birth_h1, raw_death_h1):
        b_orig = max_gray - b_inv + step
        d_orig = max_gray - d_inv + step
        if b_orig != d_orig:  # Excluir características con vida nula
            final_birth_list.append(b_orig)
            final_death_list.append(d_orig)

    if len(final_birth_list) > 0:
        final_birth_h1 = np.array(final_birth_list)
        final_death_h1 = np.array(final_death_list)
        h1_lifetimes = final_death_h1 - final_birth_h1
    else:
        h1_lifetimes = np.zeros(1, dtype=np.int64)

    h1_entropy = entropy_persistence(h1_lifetimes)
    h1_mean, h1_std, h1_max, h1_n = compute_lifetime_stats(h1_lifetimes)

    return (h0_entropy, h0_mean, h0_std, h0_max, h0_n,
            h1_entropy, h1_mean, h1_std, h1_max, h1_n,
            h0_lifetimes, h1_lifetimes)


# =============================================================================
# Wrapper para extracción de características con nombres
# =============================================================================

def tda_features(img, step=5):
    """
    Extrae los 12 descriptores topológicos de una imagen, organizados
    en un diccionario con nombres descriptivos.

    Wrapper de tda_fast que devuelve los resultados en formato
    compatible con el pipeline de extracción de características.

    Parameters
    ----------
    img : ndarray (H, W)
        Imagen en escala de grises.
    step : int
        Paso de discretización de la filtración.

    Returns
    -------
    features : dict
        12 descriptores topológicos:
        - 'H0_entropy': entropía de persistencia normalizada de H0
        - 'H0_mean': media de tiempos de vida de H0
        - 'H0_std': desviación estándar de tiempos de vida de H0
        - 'H0_max': tiempo de vida máximo de H0
        - 'H0_n': número de componentes conexas
        - 'H0_norm_entropy': entropía normalizada por rango de gris
        - 'H1_entropy': entropía de persistencia normalizada de H1
        - 'H1_mean': media de tiempos de vida de H1
        - 'H1_std': desviación estándar de tiempos de vida de H1
        - 'H1_max': tiempo de vida máximo de H1
        - 'H1_n': número de ciclos
        - 'H1_norm_entropy': entropía normalizada por rango de gris
    data : dict
        Datos completos para visualización:
        - 'H0_lifetimes': tiempos de vida de H0
        - 'H1_lifetimes': tiempos de vida de H1
    """
    result = tda_fast(img.astype(np.int64), step=step)

    (h0_entropy, h0_mean, h0_std, h0_max, h0_n,
     h1_entropy, h1_mean, h1_std, h1_max, h1_n,
     h0_lifetimes, h1_lifetimes) = result

    # Normalizar media y max por el rango de gris para comparabilidad
    # entre imágenes con diferentes rangos dinámicos
    gray_range = float(np.max(img) - np.min(img))
    if gray_range == 0:
        gray_range = 1.0

    features = {
        'H0_entropy': float(h0_entropy),
        'H0_mean': float(h0_mean) / gray_range,
        'H0_std': float(h0_std) / gray_range,
        'H0_max': float(h0_max) / gray_range,
        'H0_n': int(h0_n),
        'H1_entropy': float(h1_entropy),
        'H1_mean': float(h1_mean) / gray_range,
        'H1_std': float(h1_std) / gray_range,
        'H1_max': float(h1_max) / gray_range,
        'H1_n': int(h1_n),
    }

    data = {
        'H0_lifetimes': np.array(h0_lifetimes),
        'H1_lifetimes': np.array(h1_lifetimes),
    }

    return  data,features