import numpy as np
import utils as ut
from numba import njit, prange



@njit(parallel=True, fastmath=True)
def measures(idxy, img, s):
    """
    Calcula tres métricas de masa para cada caja de tamaño s×s:
    suma de intensidades, varianza y entropía de Shannon normalizada.

    Optimizado: un solo recorrido de píxeles por caja acumula
    suma, suma de cuadrados e histograma simultáneamente.
    La varianza se obtiene por E[X²] - E[X]².

    Parameters
    ----------
    idxy : ndarray (n, 2)
        Coordenadas (i0, j0) de cada caja.
    img : ndarray (H, W)
        Imagen en escala de grises.
    s : int
        Tamaño de la caja en píxeles.

    Returns
    -------
    intensity : ndarray (n,)
        Suma de intensidades en cada caja.
    variance : ndarray (n,)
        Varianza de intensidades en cada caja.
    entropy : ndarray (n,)
        Entropía de Shannon normalizada en [0, 1].
    """
    n = len(idxy)
    n_pixels = s * s

    intensity = np.zeros(n, dtype=np.float64)
    variance = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)

    for k in prange(n):
        x = idxy[k, 0]
        y = idxy[k, 1]

        # Un solo recorrido: suma, suma de cuadrados e histograma
        sum_val = 0.0
        sum_sq = 0.0
        counts = np.zeros(256, dtype=np.int64)

        for i in range(s):
            for j in range(s):
                val = img[x + i, y + j]
                sum_val += val
                sum_sq += val * val
                idx = int(val)
                if idx < 0:
                    idx = 0
                if idx > 255:
                    idx = 255
                counts[idx] += 1

        # Intensidad
        intensity[k] = sum_val

        # Varianza: E[X²] - E[X]²
        mean_val = sum_val / n_pixels
        variance[k] = sum_sq / n_pixels - mean_val * mean_val

        # Entropía de Shannon normalizada
        n_unique = 0
        for u in range(256):
            if counts[u] > 0:
                n_unique += 1

        if n_unique <= 1:
            entropy[k] = 0.0
        else:
            H = 0.0
            log_n = np.log(n_unique)
            for u in range(256):
                if counts[u] > 0:
                    p = counts[u] / n_pixels
                    H -= p * np.log(p)
            entropy[k] = H #/ log_n

    return intensity, variance, entropy

@njit
def idxy(img_shape, s):
    """
    Genera índices de ventanas s×s desde la esquina superior izquierda.

    A diferencia del MF-DFA que cubre desde las 4 esquinas, en Rényi
    se usa una sola dirección para evitar contar masa de píxeles
    solapados múltiples veces en la función de partición.

    Parameters
    ----------
    img_shape : tuple (nx, ny)
        Dimensiones de la imagen.
    s : int
        Tamaño de la ventana en píxeles.

    Returns
    -------
    coords : ndarray (n, 2)
        Coordenadas (i0, j0) de cada ventana.
    """
    nx, ny = img_shape
    n = (nx // s) * (ny // s)
    coords = np.zeros((n, 2), dtype=np.int32)
    idx = 0

    for i in range(0, (nx // s) * s, s):
        for j in range(0, (ny // s) * s, s):
            coords[idx, 0] = i
            coords[idx, 1] = j
            idx += 1

    return coords


def mf_renyi_features(
    img,
    q_min=-5.0,
    q_max=5.0,
    s_min=6,
    s_max=0.1
):
    """
    Calcula las dimensiones generalizadas de Rényi y el espectro
    multifractal para tres métricas de masa: intensidad, varianza
    y entropía de Shannon normalizada.

    Procedimiento por métrica:
    1. Para cada escala ε, particiona la imagen en cajas y calcula
       la masa (intensidad/varianza/entropía) de cada caja.
    2. Normaliza las masas a probabilidad: p_i = μ_i / Σμ_j.
    3. Calcula la función de partición: χ(q,ε) = Σ p_i^q.
    4. Estima τ(q) como pendiente de log χ(q,ε) vs log ε.
    5. Obtiene D_q = τ(q) / (q-1).
    6. Obtiene f(α) mediante transformada de Legendre.
    7. Extrae características del espectro.

    Parameters
    ----------
    img : ndarray (H, W)
        Imagen en escala de grises.
    q_min : float
        Extremo negativo del rango de momentos.
    q_max : float
        Extremo positivo del rango de momentos.
    s_min : int
        Escala mínima en píxeles.
    s_max : float
        Escala máxima como fracción de la dimensión menor.

    Returns
    -------
    data : dict
        Datos completos del análisis para cada métrica:
        - 'intensity', 'variance', 'entropy': cada uno contiene
          {'Dq', 'tq', 'alpha', 'f_alpha', 'qs', 'scales'}
    features : dict
        Características extraídas con prefijo por métrica:
        'int_*', 'var_*', 'ent_*'
    """
    

    # ---- Valores de q ----
    # qs = np.array(ut.vals_Qs(q_n=q_min, q_p=q_max))
    qs = np.arange(q_min - 0.25, q_max + 0.25, 0.25)
    nq = len(qs)

    # ---- Escalas ----
    img_shape = img.shape
    scales = ut.bineo(s_min, int(min(img.shape) * s_max), degree=2)
    ns = len(scales)

    # Matrices de función de partición para cada métrica
    chi_sum = np.zeros((nq, ns), dtype=np.float64)
    chi_var = np.zeros((nq, ns), dtype=np.float64)
    chi_ent = np.zeros((nq, ns), dtype=np.float64)

    # ---- Calcular función de partición por escala ----
    for is_, s in enumerate(scales):
        xy = idxy(img_shape=img_shape, s=int(s))
        xy = np.ascontiguousarray(xy, dtype=np.int32)
        Sum, Var, Ent = measures(idxy=xy, img=img, s=int(s))

        # Normalizar a probabilidad (excluir valores <= 0)
        p_sum = _normalize(Sum)
        p_var = _normalize(Var)
        p_ent = _normalize(Ent)

        # Función de partición para cada q
        chi_sum[:, is_] = _partition_function(p_sum, qs)
        chi_var[:, is_] = _partition_function(p_var, qs)
        chi_ent[:, is_] = _partition_function(p_ent, qs)


    # ---- Obtener espectros y características por métrica ----
    log_s = np.log(scales)

    data = {}
    features = {}

    for name, chi in [('sum', chi_sum), ('var', chi_var), ('ent', chi_ent)]:
        tq, Dq, alpha, f_alpha, chi = _compute_spectrum(chi, qs, log_s)

        data[name] = {
            'Dq': np.array(Dq),
            'tq': np.array(tq),
            'alpha': np.array(alpha),
            'f_alpha': np.array(f_alpha),
            'qs': np.array(qs),
            'scales': np.array(scales),
            'functions': np.array(chi)
        }

        feat = _extract_features(alpha, f_alpha, Dq, qs, tq)
        for key, val in feat.items():
            features[f'{name}_{key}'] = val

        # break

    return data, features


def _normalize(masses):
    """
    Normaliza masas a probabilidad, excluyendo valores <= 0.

    Parameters
    ----------
    masses : ndarray (n,)

    Returns
    -------
    p : ndarray (n,)
        Probabilidades normalizadas. Valores <= 0 se mantienen en 0.
    """
    p = np.copy(masses)
    p[p <= 0] = 0.0
    total = np.sum(p)
    if total > 0:
        p = p / total
    return p


def _partition_function(p, qs):
    """
    Calcula la función de partición χ(q) = Σ p_i^q para cada q.

    Para q=1 se usa la convención χ = exp(Σ p_i * log(p_i))
    para evitar la indeterminación en D_q = τ(q)/(q-1).

    Parameters
    ----------
    p : ndarray (n,)
        Probabilidades normalizadas (> 0).
    qs : ndarray (nq,)
        Valores de q.

    Returns
    -------
    chi : ndarray (nq,)
        Función de partición para cada q.
    """
    nq = len(qs)
    chi = np.zeros(nq, dtype=np.float64)

    # Filtrar probabilidades positivas
    p_pos = p[p > 0]
    if len(p_pos) == 0:
        return chi

    for iq in range(nq):
        q = qs[iq]
        if abs(q - 1.0) < 1e-10:
            # q ≈ 1: usar Σ p_i * log(p_i) directamente
            # chi[iq] = np.exp(np.sum(p_pos * np.log(p_pos)))
            chi[iq] = - np.sum(p_pos * np.log(p_pos))
        else:
            chi[iq] = np.sum( p_pos ** q )

    return chi


def _compute_spectrum(chi, qs, log_s):
    """
    Calcula τ(q), D_q, α y f(α) a partir de la función de partición.

    Parameters
    ----------
    chi : ndarray (nq, ns)
        Función de partición para cada q y escala.
    qs : ndarray (nq,)
        Valores de q.
    log_s : ndarray (ns,)
        Logaritmo de las escalas.

    Returns
    -------
    tq : ndarray (nq,)
        Función de masa τ(q).
    Dq : ndarray (nq,)
        Dimensiones generalizadas D_q.
    alpha : ndarray (nq,)
        Exponentes de singularidad.
    f_alpha : ndarray (nq,)
        Espectro multifractal.
    """
    nq = len(qs)
    tq = np.zeros(nq, dtype=np.float64)
    Dq = np.zeros(nq, dtype=np.float64)

    for iq in range(nq):
        pi = chi[iq, :]
        log_chi = np.log( pi + 1e-300)  # Evitar log(0)
        
        q = qs[iq]
        if abs(q - 1.0) < 1e-10: # q = 1
            # D_1: dimensión de información (límite)
            coeffs = np.polyfit(log_s, pi, 1)
            tq[iq] = 0.0  # τ(1) = 0 siempre
            Dq[iq] = - coeffs[0]  
            
        elif abs(q) < 1e-10: # q = 0
            # D_0: dimensión box-counting
            coeffs = np.polyfit(log_s, log_chi, 1)
            tq[iq] = coeffs[0]
            Dq[iq] = - tq[iq]  
            
        else:
            coeffs = np.polyfit(log_s, log_chi, 1)
            tq[iq] = coeffs[0]
            Dq[iq] = tq[iq] / (q - 1.0)

    # Espectro multifractal via Legendre
    alpha = np.gradient(tq, qs)
    f_alpha = qs * alpha - tq

    alpha = alpha[1:-1]
    f_alpha = f_alpha[1:-1]

    return tq, Dq, alpha, f_alpha, chi


def _extract_features(alpha, f_alpha, Dq, qs, tq):
    """
    Extrae características del espectro multifractal y las dimensiones
    generalizadas.

    Parameters
    ----------
    alpha : ndarray
        Exponentes de singularidad.
    f_alpha : ndarray
        Espectro multifractal.
    Dq : ndarray
        Dimensiones generalizadas.
    qs : ndarray
        Valores de q.
    tq : ndarray
        Función de masa.

    Returns
    -------
    features : dict
        Características extraídas:
        - 'a_max', 'a_min': extremos del espectro
        - 'dif_a': ancho del espectro
        - 'a_star': posición del máximo de f(α)
        - 'dif_L', 'dif_R': brazos izquierdo y derecho
        - 'asy_i': índice de asimetría
        - 'f_max', 'f_min': alturas extremas del espectro
        - 'dif_f': diferencia de alturas
        - 'D0': dimensión box-counting
        - 'D1': dimensión de información
        - 'D2': dimensión de correlación
    """
    a_max = alpha[0]
    a_min = alpha[-1]
    dif_a = np.abs(a_max - a_min)

    a_star = alpha[np.argmax(f_alpha)]

    dif_L = np.abs(a_star - a_min)
    dif_R = np.abs(a_max - a_star)

    denom = dif_L + dif_R
    asy_i = (dif_L - dif_R) / denom if denom > 0 else 0.0

    f_max = f_alpha[0]
    f_min = f_alpha[-1]
    dif_f = np.abs(np.max(f_alpha) - np.min(f_alpha))

    # Dimensiones especiales: D0, D1, D2
    # Buscar los q más cercanos a 0, 1 y 2
    idx_q0 = np.argmin(np.abs(qs))
    idx_q1 = np.argmin(np.abs(qs - 1.0))
    idx_q2 = np.argmin(np.abs(qs - 2.0))

    D0 = float(Dq[idx_q0])
    D1 = float(Dq[idx_q1])
    D2 = float(Dq[idx_q2])

    features = {
        'a_max': float(a_max),
        'a_min': float(a_min),
        'dif_a': float(dif_a),
        'a_star': float(a_star),
        'dif_L': float(dif_L),
        'dif_R': float(dif_R),
        'asy_i': float(asy_i),
        'f_max': float(f_max),
        'f_min': float(f_min),
        'dif_f': float(dif_f),
        'D0': D0,
        'D1': D1,
        'D2': D2,
    }

    return features