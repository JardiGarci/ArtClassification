"""
Pruebas corregidas para H1.

El problema anterior: cuando el interior de un marco tiene la misma
intensidad que el fondo (ambos 0), la filtración no puede distinguir
"dentro" de "fuera" y no forma ciclos.

Corrección: el interior debe tener una intensidad diferente tanto
del borde como del fondo exterior.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_hollow_squares_v2(size=256, n_squares=4, square_size=40, border=5):
    """
    Marcos huecos con fondo=30, borde=180, interior=80.
    Tres niveles de intensidad garantizan que la filtración
    forme ciclos al rodear el interior con el borde.
    """
    img = np.full((size, size), 30, dtype=np.int64)  # Fondo gris oscuro
    np.random.seed(42)

    positions = []
    for k in range(n_squares):
        for _ in range(200):
            x = np.random.randint(10, size - square_size - 10)
            y = np.random.randint(10, size - square_size - 10)
            overlap = False
            for px, py in positions:
                if (abs(x - px) < square_size + 15 and
                        abs(y - py) < square_size + 15):
                    overlap = True
                    break
            if not overlap:
                positions.append((x, y))
                break

        if len(positions) == k + 1:
            x, y = positions[-1]
            img[x:x + square_size, y:y + square_size] = 180       # Borde
            img[x + border:x + square_size - border,
                y + border:y + square_size - border] = 80          # Interior

    return img


def generate_nested_circles_v2(size=256):
    """
    Fondo=20, anillo=180, disco interior=80.
    El anillo rodea al disco con intensidad diferente en ambos lados.
    """
    img = np.full((size, size), 20, dtype=np.int64)
    cx, cy = size // 2, size // 2

    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if r < 30:
                img[i, j] = 80
            elif r < 80:
                img[i, j] = 180

    return img


def generate_bullseye(size=256):
    """
    Diana con anillos alternados: 40, 160, 40, 160, 40.
    Cada transición de alto a bajo genera un ciclo.
    """
    img = np.full((size, size), 20, dtype=np.int64)
    cx, cy = size // 2, size // 2
    radii = [100, 80, 60, 40, 20]
    intensities = [40, 160, 40, 160, 40]

    for i in range(size):
        for j in range(size):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            for rad, val in zip(radii, intensities):
                if r < rad:
                    img[i, j] = val

    return img


if __name__ == '__main__':
    import tda as tda_module

    step = 5
    tests = [
        ('Marcos huecos v2', generate_hollow_squares_v2()),
        ('Círculos anidados v2', generate_nested_circles_v2()),
        ('Diana (bullseye)', generate_bullseye()),
    ]

    fig, axes = plt.subplots(2, len(tests), figsize=(5 * len(tests), 8))

    for i, (name, img) in enumerate(tests):
        features, data = tda_module.tda_features(img.astype(np.int64), step=step)

        print(f"\n{name}:")
        print(f"  H0: entropy={features['H0_entropy']:.4f}, n={features['H0_n']}")
        print(f"  H1: entropy={features['H1_entropy']:.4f}, n={features['H1_n']}")

        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(name)
        axes[0, i].axis('off')

        h1_life = data['H1_lifetimes']
        h1_pos = h1_life[h1_life > 0]
        if len(h1_pos) > 0:
            births = np.random.uniform(0, 255 - h1_pos.clip(max=254), len(h1_pos))
            deaths = births + h1_pos
            axes[1, i].scatter(births, deaths, s=15, alpha=0.5, c='#D85A30')
        axes[1, i].plot([0, 255], [0, 255], 'k--', linewidth=0.5, alpha=0.5)
        axes[1, i].set_title(f'H1: E={features["H1_entropy"]:.3f}, n={features["H1_n"]}')
        axes[1, i].set_xlabel('Birth')
        axes[1, i].set_ylabel('Death')
        axes[1, i].set_xlim(-5, 260)
        axes[1, i].set_ylim(-5, 260)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('validation_tda_h1_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nGráfica guardada: validation_tda_h1_v2.png")