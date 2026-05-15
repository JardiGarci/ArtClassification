import numpy as np
from numba import njit, prange
import utils as ut

# =============================================================================
# Partición en ventanas desde las 4 esquinas
# =============================================================================

@njit
def idxy_4(img_shape, s):

    nx, ny = img_shape
    rx = int((nx % s) != 0)  # 1 si hay residuo en filas
    ry = int((ny % s) != 0)  # 1 si hay residuo en columnas

    # Número total de ventanas considerando las 4 orientaciones
    n = (nx // s) * (ny // s) * (1 + rx + ry + (rx * ry))
 
    idxy = np.zeros((n, 2), dtype=np.int32)
    idx = 0

    # Esquina superior izquierda → (arriba-abajo, izquierda-derecha)
    for i in range(0, nx - (s * rx), s):
        for j in range(0, ny - (s * ry), s):
            idxy[idx, 0] = i
            idxy[idx, 1] = j
            idx += 1

    # Esquina superior derecha → (arriba-abajo, derecha-izquierda)
    if ry != 0:
        for i in range(0, nx - (s * rx), s):
            for j in range(ny - (s * ry), 0, -s):
                idxy[idx, 0] = i
                idxy[idx, 1] = j
                idx += 1

    # Esquina inferior izquierda → (abajo-arriba, izquierda-derecha)
    if rx != 0:
        for i in range(nx - (s * rx), 0, -s):
            for j in range(0, ny - (s * ry), s):
                idxy[idx, 0] = i
                idxy[idx, 1] = j
                idx += 1

        # Esquina inferior derecha → (abajo-arriba, derecha-izquierda)
        if ry != 0:
            for i in range(nx - (s * rx), 0, -s):
                for j in range(ny - (s * ry), 0, -s):
                    idxy[idx, 0] = i
                    idxy[idx, 1] = j
                    idx += 1

    return idxy


@njit(fastmath=True)
def poly2d_fluctuation_order1(img, i0, j0, s, integration=True):
    """MF-DFA1: detrending lineal (3 coeficientes: i, j, 1)."""
    n = s * s
    Y = np.zeros((s, s), dtype=np.float64)

    if integration:
        mean_val = 0.0
        for di in range(s):
            for dj in range(s):
                mean_val += img[i0 + di, j0 + dj]
        mean_val /= n

        for di in range(s):
            for dj in range(s):
                val = img[i0 + di, j0 + dj] - mean_val
                Y[di, dj] = val
                if di > 0:
                    Y[di, dj] += Y[di - 1, dj]
                if dj > 0:
                    Y[di, dj] += Y[di, dj - 1]
                if di > 0 and dj > 0:
                    Y[di, dj] -= Y[di - 1, dj - 1]
    else:
        for di in range(s):
            for dj in range(s):
                Y[di, dj] = img[i0 + di, j0 + dj]

    # 3 coeficientes: a*i + b*j + c
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    for di in range(s):
        for dj in range(s):
            z = Y[di, dj]
            x0 = float(di)
            x1 = float(dj)

            A[0, 0] += x0 * x0
            A[0, 1] += x0 * x1
            A[0, 2] += x0
            A[1, 1] += x1 * x1
            A[1, 2] += x1
            A[2, 2] += 1.0

            b[0] += x0 * z
            b[1] += x1 * z
            b[2] += z

    for i in range(3):
        for j in range(i):
            A[i, j] = A[j, i]

    for k in range(3):
        piv = A[k, k]
        if piv == 0.0:
            return 0.0
        inv = 1.0 / piv
        for j in range(k, 3):
            A[k, j] *= inv
        b[k] *= inv
        for i in range(3):
            if i != k:
                f = A[i, k]
                for j in range(k, 3):
                    A[i, j] -= f * A[k, j]
                b[i] -= f * b[k]

    res = 0.0
    for di in range(s):
        for dj in range(s):
            z_hat = b[0] * di + b[1] * dj + b[2]
            d = Y[di, dj] - z_hat
            res += d * d

    return np.sqrt(res / n)


@njit(fastmath=True)
def poly2d_fluctuation_order2(img, i0, j0, s, integration=True):

    n = s * s

    # --- Paso 1: Integración local (suma acumulada dentro de la ventana) ---
    Y = np.zeros((s, s), dtype=np.float64)

    if integration:
        # Sustraer media local
        mean_val = 0.0
        for di in range(s):
            for dj in range(s):
                mean_val += img[i0 + di, j0 + dj]
        mean_val /= n

        # Suma acumulada 2D local
        for di in range(s):
            for dj in range(s):
                val = img[i0 + di, j0 + dj] - mean_val
                Y[di, dj] = val
                if di > 0:
                    Y[di, dj] += Y[di - 1, dj]
                if dj > 0:
                    Y[di, dj] += Y[di, dj - 1]
                if di > 0 and dj > 0:
                    Y[di, dj] -= Y[di - 1, dj - 1]
    else:
        # Sin integración: usar valores directos
        for di in range(s):
            for dj in range(s):
                Y[di, dj] = img[i0 + di, j0 + dj]

    # --- Paso 2: Detrending polinomial de segundo orden ---
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros(6, dtype=np.float64)

    for di in range(s):
        for dj in range(s):
            z = Y[di, dj]

            # xc = di - (s - 1) / 2.0
            # yc = dj - (s - 1) / 2.0
            # x0 = xc
            # x1 = yc

            x0 = di * di
            x1 = dj * dj
            x2 = di * dj
            x3 = di
            x4 = dj

            A[0, 0] += x0 * x0
            A[0, 1] += x0 * x1
            A[0, 2] += x0 * x2
            A[0, 3] += x0 * x3
            A[0, 4] += x0 * x4
            A[0, 5] += x0

            A[1, 1] += x1 * x1
            A[1, 2] += x1 * x2
            A[1, 3] += x1 * x3
            A[1, 4] += x1 * x4
            A[1, 5] += x1

            A[2, 2] += x2 * x2
            A[2, 3] += x2 * x3
            A[2, 4] += x2 * x4
            A[2, 5] += x2

            A[3, 3] += x3 * x3
            A[3, 4] += x3 * x4
            A[3, 5] += x3

            A[4, 4] += x4 * x4
            A[4, 5] += x4

            A[5, 5] += 1.0

            b[0] += x0 * z
            b[1] += x1 * z
            b[2] += x2 * z
            b[3] += x3 * z
            b[4] += x4 * z
            b[5] += z

    for i in range(6):
        for j in range(i):
            A[i, j] = A[j, i]

    for k in range(6):
        piv = A[k, k]
        if piv == 0.0:
            return 0.0
        inv = 1.0 / piv
        for j in range(k, 6):
            A[k, j] *= inv
        b[k] *= inv

        for i in range(6):
            if i != k:
                f = A[i, k]
                for j in range(k, 6):
                    A[i, j] -= f * A[k, j]
                b[i] -= f * b[k]

    # --- Paso 3: Residuos y RMSE ---
    res = 0.0
    for di in range(s):
        for dj in range(s):
            z_hat = (b[0] * di * di + b[1] * dj * dj + b[2] * di * dj
                     + b[3] * di + b[4] * dj + b[5])
            d = Y[di, dj] - z_hat
            res += d * d

    return np.sqrt(res / n)


@njit(fastmath=True)
def poly2d_fluctuation_order3(img, i0, j0, s, integration=True):
    """MF-DFA3: detrending cúbico (10 coeficientes)."""
    n = s * s
    Y = np.zeros((s, s), dtype=np.float64)

    if integration:
        mean_val = 0.0
        for di in range(s):
            for dj in range(s):
                mean_val += img[i0 + di, j0 + dj]
        mean_val /= n

        for di in range(s):
            for dj in range(s):
                val = img[i0 + di, j0 + dj] - mean_val
                Y[di, dj] = val
                if di > 0:
                    Y[di, dj] += Y[di - 1, dj]
                if dj > 0:
                    Y[di, dj] += Y[di, dj - 1]
                if di > 0 and dj > 0:
                    Y[di, dj] -= Y[di - 1, dj - 1]
    else:
        for di in range(s):
            for dj in range(s):
                Y[di, dj] = img[i0 + di, j0 + dj]

    # 10 coeficientes: i³, j³, i²j, ij², i², j², ij, i, j, 1
    NC = 10
    A = np.zeros((NC, NC), dtype=np.float64)
    bv = np.zeros(NC, dtype=np.float64)

    for di in range(s):
        for dj in range(s):
            z  = Y[di, dj]
            ii = float(di)
            jj = float(dj)

            x0 = ii * ii * ii;  x1 = jj * jj * jj;  x2 = ii * ii * jj
            x3 = ii * jj * jj;  x4 = ii * ii;        x5 = jj * jj
            x6 = ii * jj;       x7 = ii;              x8 = jj

            A[0,0]+=x0*x0; A[0,1]+=x0*x1; A[0,2]+=x0*x2; A[0,3]+=x0*x3; A[0,4]+=x0*x4
            A[0,5]+=x0*x5; A[0,6]+=x0*x6; A[0,7]+=x0*x7; A[0,8]+=x0*x8; A[0,9]+=x0
            A[1,1]+=x1*x1; A[1,2]+=x1*x2; A[1,3]+=x1*x3; A[1,4]+=x1*x4
            A[1,5]+=x1*x5; A[1,6]+=x1*x6; A[1,7]+=x1*x7; A[1,8]+=x1*x8; A[1,9]+=x1
            A[2,2]+=x2*x2; A[2,3]+=x2*x3; A[2,4]+=x2*x4
            A[2,5]+=x2*x5; A[2,6]+=x2*x6; A[2,7]+=x2*x7; A[2,8]+=x2*x8; A[2,9]+=x2
            A[3,3]+=x3*x3; A[3,4]+=x3*x4
            A[3,5]+=x3*x5; A[3,6]+=x3*x6; A[3,7]+=x3*x7; A[3,8]+=x3*x8; A[3,9]+=x3
            A[4,4]+=x4*x4; A[4,5]+=x4*x5; A[4,6]+=x4*x6; A[4,7]+=x4*x7; A[4,8]+=x4*x8; A[4,9]+=x4
            A[5,5]+=x5*x5; A[5,6]+=x5*x6; A[5,7]+=x5*x7; A[5,8]+=x5*x8; A[5,9]+=x5
            A[6,6]+=x6*x6; A[6,7]+=x6*x7; A[6,8]+=x6*x8; A[6,9]+=x6
            A[7,7]+=x7*x7; A[7,8]+=x7*x8; A[7,9]+=x7
            A[8,8]+=x8*x8; A[8,9]+=x8
            A[9,9]+=1.0

            bv[0]+=x0*z; bv[1]+=x1*z; bv[2]+=x2*z; bv[3]+=x3*z; bv[4]+=x4*z
            bv[5]+=x5*z; bv[6]+=x6*z; bv[7]+=x7*z; bv[8]+=x8*z; bv[9]+=z

    for i in range(NC):
        for j in range(i):
            A[i, j] = A[j, i]

    for k in range(NC):
        piv = A[k, k]
        if piv == 0.0:
            return 0.0
        inv = 1.0 / piv
        for j in range(k, NC):
            A[k, j] *= inv
        bv[k] *= inv
        for i in range(NC):
            if i != k:
                f = A[i, k]
                for j in range(k, NC):
                    A[i, j] -= f * A[k, j]
                bv[i] -= f * bv[k]

    res = 0.0
    for di in range(s):
        for dj in range(s):
            ii = float(di)
            jj = float(dj)
            z_hat = (bv[0] * ii * ii * ii + bv[1] * jj * jj * jj
                     + bv[2] * ii * ii * jj + bv[3] * ii * jj * jj
                     + bv[4] * ii * ii + bv[5] * jj * jj
                     + bv[6] * ii * jj + bv[7] * ii + bv[8] * jj + bv[9])
            d = Y[di, dj] - z_hat   # uv −  ̃uv
            res += d * d

    return np.sqrt(res / n) # F(u,w,s)


@njit(parallel=True, fastmath=True)
def local_fluctuation(Y, idxy, s, integration = True, degree_trend = 2):
    n = idxy.shape[0]
    F_uw = np.zeros(n)
    for k in prange(n):
        i0 = idxy[k, 0]
        j0 = idxy[k, 1]
        if degree_trend == 1:
            F_uw[k] = poly2d_fluctuation_order1(Y, i0, j0, s, integration=integration)
        elif degree_trend == 2:
            F_uw[k] = poly2d_fluctuation_order2(Y, i0, j0, s, integration=integration)
        elif degree_trend == 3:
            F_uw[k] = poly2d_fluctuation_order3(Y, i0, j0, s, integration=integration)
    return F_uw


# # =============================================================================
# # Función de fluctuación generalizada Fq(s)
# # =============================================================================

@njit(parallel=True, fastmath=True)
def mf_fluctuation(f_loc, qs):
    nq = qs.shape[0]
    n = f_loc.shape[0]
    # Omitir fluctuaciones nulas (ventanas degeneradas)
    n_0 = 0
    for i in range(n):
        if f_loc[i] > 0:
            n_0 += 1
    f_loc_0 = np.empty(n_0, dtype=np.float64)
    idx = 0
    for i in range(n):
        if f_loc[i] > 0:
            f_loc_0[idx] = f_loc[i]
            idx += 1

    Fq = np.zeros(nq)
    for iq in prange(nq):
        q = qs[iq]

        if q == 0.0:
            # Media geométrica (límite de Fq cuando q→0)
            acc = 0.0
            for i in range(n_0):
                acc += np.log(f_loc_0[i])
            Fq[iq] = np.exp(acc / n)
        else:
            acc = 0.0
            for i in range(n_0):
                acc += f_loc_0[i] ** q
            Fq[iq] = (acc / n) ** (1.0 / q) #Fqs

    return Fq


# =============================================================================
# MF-DFA completo con extracción de características
# =============================================================================

def mf_dfa_features(
    img,
    q_min=-5.0,
    q_max=5.0,
    s_min=6,
    s_max=0.1,
    integration=True,
    degree_trend = 2,
    degree_scales = 2,
):

    # ---- Valores de q ----
    # qs = np.array(ut.vals_Qs(q_n=q_min, q_p=q_max))
    qs = np.arange(q_min - 0.5,q_max + 0.75,0.25)

    # ---- Escalas ----
    img_shape = img.shape
    if type(s_min) == int:
        if type(s_max) == int:
            scales = ut.bineo(s_min, s_max, degree = degree_scales)
        else:
            scales = ut.bineo(s_min, int(min(img.shape) * s_max), degree = degree_scales)
    elif type(s_max) == int:
        scales = ut.bineo( int(min(img.shape) * s_min), s_max, degree = degree_scales)
    else:
        scales = ut.bineo( int(min(img.shape) * s_min), int(min(img.shape) * s_max), degree = degree_scales)


    Y = img.copy()
    

    nq = qs.shape[0]
    ns = scales.shape[0]

    Fqs = np.zeros((nq, ns), dtype=np.float64)
    F_s = []  # Para el Hurst clásico (q=2)

    # ---- MF-DFA: fluctuaciones por escala ----
    for is_, s in enumerate(scales):
        idxy = idxy_4(img_shape, s)
        idxy = np.ascontiguousarray(idxy, dtype=np.int32)

        f_loc = local_fluctuation(Y, idxy, int(s), integration = integration, degree_trend=degree_trend)

        # Fluctuación generalizada para todos los q
        Fqs[:, is_] = mf_fluctuation(f_loc, qs)

        # Fluctuación clásica (q=2) para el exponente de Hurst
        f_s = float(np.sqrt(np.mean(np.power(f_loc, 2))))
        F_s.append(f_s)

    # ---- Exponente de Hurst clásico ----
    vals = np.polyfit(np.log(scales), np.log(F_s), deg=1)

    # ---- h(q): exponente de Hurst generalizado ----
    log_s = np.log(scales)
    hq = np.zeros(nq, dtype=np.float64)
    for iq in range(nq):
        coeffs = np.polyfit(log_s, np.log(Fqs[iq, :]), 1)
        hq[iq] = coeffs[0]

    # ---- τ(q): función de masa ----
    D = 2.0  # Dimensión del espacio (imagen 2D)
    tau_q = qs * hq - D

    # ---- Espectro multifractal f(α) via transformada de Legendre ----
    alpha = np.gradient(tau_q, qs)       # α(q) = dτ/dq
    f_alpha = qs * alpha - tau_q         # f(α) = q·α - τ(q)

    # ---- Empaquetar datos completos ----
    data = {
        'alpha': np.array(alpha)[2:-2],
        'f_alpha': np.array(f_alpha)[2:-2],
        'hq': np.array(hq)[2:-2],
        'tq': np.array(tau_q)[2:-2],
        'qs': np.array(qs)[2:-2],
        's_sizes': np.array(scales),
        'fluctuations': Fqs[2:-2],
    }


    # ---- Extracción de 14 características ----

    # Extremos y ancho del espectro
    a_max = data['alpha'][0]          # Extremo derecho (q más negativo)
    a_min = data['alpha'][-1]         # Extremo izquierdo (q más positivo)
    dif_a = np.abs(a_max - a_min)     # Ancho total Δα

    # Posición del máximo
    a_star = data['alpha'][data['f_alpha'] == np.max(data['f_alpha'])][0]

    # Asimetría del espectro
    dif_L = np.abs(a_star - a_min)    # Brazo izquierdo
    dif_R = np.abs(a_max - a_star)    # Brazo derecho
    asy_i = (dif_L - dif_R) / (dif_L + dif_R)  # Índice de asimetría

    # Alturas del espectro
    f_max = data['f_alpha'][0]        # Altura en α_max
    f_min = data['f_alpha'][-1]       # Altura en α_min
    dif_f = np.abs(np.max(data['f_alpha']) - np.min(data['f_alpha']))

    # Ajuste cuadrático de τ(q): captura la curvatura global
    a, b, c = np.polyfit(data['qs'], data['tq'], 2)

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
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'Hurst': float(vals[0])
    }

    return data, features


@njit(parallel=True, fastmath=True)
def measures_int(idxy, img, s):
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


@njit(parallel=True, fastmath=True)
def measures(idxy, img, s):

    n = len(idxy)
    n_pixels = s * s
 
    intensity = np.zeros(n, dtype=np.float64)
    variance = np.zeros(n, dtype=np.float64)
    entropy = np.zeros(n, dtype=np.float64)
 
    for k in prange(n):
        x = idxy[k, 0]
        y = idxy[k, 1]
 
        # --- Un solo recorrido: suma, suma de cuadrados y valores ---
        sum_val = 0.0
        sum_sq = 0.0
        values = np.zeros(n_pixels, dtype=np.float64)
        idx = 0
 
        for i in range(s):
            for j in range(s):
                val = img[x + i, y + j]
                sum_val += val
                sum_sq += val * val
                values[idx] = val
                idx += 1
 
        # Intensidad
        intensity[k] = sum_val
 
        # Varianza: E[X²] - E[X]²
        mean_val = sum_val / n_pixels
        variance[k] = sum_sq / n_pixels - mean_val * mean_val
 
        # --- Entropía de Shannon ---
        # Contar valores únicos con búsqueda lineal
        unique_vals = np.zeros(n_pixels, dtype=np.float64)
        unique_counts = np.zeros(n_pixels, dtype=np.int64)
        n_unique = 0
 
        for i in range(n_pixels):
            v = values[i]
            found = False
            for u in range(n_unique):
                if unique_vals[u] == v:
                    unique_counts[u] += 1
                    found = True
                    break
            if not found:
                unique_vals[n_unique] = v
                unique_counts[n_unique] = 1
                n_unique += 1
 
        if n_unique <= 1:
            entropy[k] = 0.0
        else:
            H = 0.0
            for u in range(n_unique):
                p = unique_counts[u] / n_pixels
                H -= p * np.log(p)
            entropy[k] = H
 
    return intensity, variance, entropy


@njit(parallel=True, fastmath=True)
def measures_sum_only(idxy, img, s):
    n = len(idxy)
    intensity = np.zeros(n, dtype=np.float64)
    for k in prange(n):
        x = idxy[k, 0]
        y = idxy[k, 1]
        s_val = 0.0
        for i in range(s):
            for j in range(s):
                s_val += img[x + i, y + j]
        intensity[k] = s_val
    return intensity


@njit
def idxy(img_shape, s):

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
    s_max=0.1,
    img_256 = True,
    degree_scales = 2
):

    # ---- Valores de q ----
    # qs = np.array(ut.vals_Qs(q_n=q_min, q_p=q_max))
    qs = np.arange(q_min - 0.25, q_max + 0.25, 0.25)
    nq = len(qs)

    # ---- Escalas ----
    img_shape = img.shape
    if type(s_min) == int:
        if type(s_max) == int:
            scales = ut.bineo(s_min, s_max, degree = degree_scales)
        else:
            scales = ut.bineo(s_min, int(min(img.shape) * s_max), degree = degree_scales)
    elif type(s_max) == int:
        scales = ut.bineo( int(min(img.shape) * s_min), s_max, degree = degree_scales)
    else:
        scales = ut.bineo( int(min(img.shape) * s_min), int(min(img.shape) * s_max), degree = degree_scales)
    # scales = ut.bineo(s_min, int(min(img.shape) * s_max), degree=2)
    ns = len(scales)

    # Matrices de función de partición para cada métrica
    chi_sum = np.zeros((nq, ns), dtype=np.float64)
    chi_var = np.zeros((nq, ns), dtype=np.float64)
    chi_ent = np.zeros((nq, ns), dtype=np.float64)

    # ---- Calcular función de partición por escala ----
    for is_, s in enumerate(scales):
        xy = idxy(img_shape=img_shape, s=int(s))
        xy = np.ascontiguousarray(xy, dtype=np.int32)
        if img_256 == True:
            Sum, Var, Ent = measures_int(idxy=xy, img=img, s=int(s))
        else:
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

    p = np.copy(masses)
    p[p <= 0] = 0.0
    total = np.sum(p)
    if total > 0:
        p = p / total
    return p


def _partition_function(p, qs):

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


# =============================================================================
# Helpers compartidos para la función combinada
# =============================================================================

def _compute_scales(img, s_min, s_max, degree_scales):
    if isinstance(s_min, int):
        if isinstance(s_max, int):
            return ut.bineo(s_min, s_max, degree=degree_scales)
        else:
            return ut.bineo(s_min, int(min(img.shape) * s_max), degree=degree_scales)
    elif isinstance(s_max, int):
        return ut.bineo(int(min(img.shape) * s_min), s_max, degree=degree_scales)
    else:
        return ut.bineo(int(min(img.shape) * s_min), int(min(img.shape) * s_max), degree=degree_scales)


def _postprocess_dfa(qs, scales, Fqs, F_s):
    log_s = np.log(scales)
    vals = np.polyfit(log_s, np.log(F_s), deg=1)
    nq = len(qs)
    hq = np.zeros(nq)
    for iq in range(nq):
        coeffs = np.polyfit(log_s, np.log(Fqs[iq, :]), 1)
        hq[iq] = coeffs[0]
    D = 2.0
    tau_q = qs * hq - D
    alpha = np.gradient(tau_q, qs)
    f_alpha = qs * alpha - tau_q

    data = {
        'alpha':        np.array(alpha)[2:-2],
        'f_alpha':      np.array(f_alpha)[2:-2],
        'hq':           np.array(hq)[2:-2],
        'tq':           np.array(tau_q)[2:-2],
        'qs':           np.array(qs)[2:-2],
        's_sizes':      np.array(scales),
        'fluctuations': Fqs[2:-2],
    }

    alpha_ = data['alpha']
    fa_    = data['f_alpha']
    a_max  = alpha_[0]
    a_min  = alpha_[-1]
    dif_a  = np.abs(a_max - a_min)
    a_star = alpha_[fa_ == np.max(fa_)][0]
    dif_L  = np.abs(a_star - a_min)
    dif_R  = np.abs(a_max - a_star)
    asy_i  = (dif_L - dif_R) / (dif_L + dif_R)
    f_max  = fa_[0]
    f_min  = fa_[-1]
    dif_f  = np.abs(np.max(fa_) - np.min(fa_))
    a, b, c = np.polyfit(data['qs'], data['tq'], 2)

    features = {
        'a_max':  float(a_max),  'a_min':  float(a_min),  'dif_a':  float(dif_a),
        'a_star': float(a_star), 'dif_L':  float(dif_L),  'dif_R':  float(dif_R),
        'asy_i':  float(asy_i),  'f_max':  float(f_max),  'f_min':  float(f_min),
        'dif_f':  float(dif_f),  'a':      float(a),      'b':      float(b),
        'c':      float(c),      'Hurst':  float(vals[0]),
    }
    return data, features


def _postprocess_renyi_sum(chi_sum, qs, scales):
    log_s = np.log(scales)
    tq, Dq, alpha, f_alpha, chi_sum = _compute_spectrum(chi_sum, qs, log_s)
    data = {
        'Dq':       np.array(Dq),
        'tq':       np.array(tq),
        'alpha':    np.array(alpha),
        'f_alpha':  np.array(f_alpha),
        'qs':       np.array(qs),
        'scales':   np.array(scales),
        'functions': np.array(chi_sum),
    }
    features = _extract_features(alpha, f_alpha, Dq, qs, tq)
    return data, features


# =============================================================================
# Análisis multifractal combinado: MF-DFA + MF-Rényi (solo suma)
# =============================================================================

def mf_combined_features(
    img,
    q_min=-5.0,
    q_max=5.0,
    s_min=6,
    s_max=0.1,
    integration=True,
    degree_trend=2,
    degree_scales=2,
    img_256=True,
):
    qs_dfa   = np.arange(q_min - 0.5,  q_max + 0.75, 0.25)
    qs_renyi = np.arange(q_min - 0.25, q_max + 0.25, 0.25)

    scales    = _compute_scales(img, s_min, s_max, degree_scales)
    img_shape = img.shape
    Y         = img.copy()
    ns        = len(scales)

    Fqs     = np.zeros((len(qs_dfa),   ns), dtype=np.float64)
    chi_sum = np.zeros((len(qs_renyi), ns), dtype=np.float64)
    F_s     = []

    for is_, s in enumerate(scales):
        s_int = int(s)

        # MF-DFA: ventanas desde las 4 esquinas con detrending polinomial
        idxy4 = idxy_4(img_shape, s_int)
        idxy4 = np.ascontiguousarray(idxy4, dtype=np.int32)
        f_loc = local_fluctuation(Y, idxy4, s_int, integration=integration, degree_trend=degree_trend)
        Fqs[:, is_] = mf_fluctuation(f_loc, qs_dfa)
        F_s.append(float(np.sqrt(np.mean(np.power(f_loc, 2)))))

        # MF-Rényi: grilla simple, solo métrica suma
        xy = idxy(img_shape, s_int)
        xy = np.ascontiguousarray(xy, dtype=np.int32)
        Sum = measures_sum_only(xy, img, s_int)
        chi_sum[:, is_] = _partition_function(_normalize(Sum), qs_renyi)

    dfa_data,   dfa_feat   = _postprocess_dfa(qs_dfa, scales, Fqs, F_s)
    renyi_data, renyi_feat = _postprocess_renyi_sum(chi_sum, qs_renyi, scales)

    return dfa_data, dfa_feat, renyi_data, renyi_feat