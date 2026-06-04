from abc import ABC
import numpy as np


class Simulation(ABC):
    def __init__(self, n, r=1e-4, rho=1e4, eta=1e-3):
        # Particle
        self.n = n
        self.r = r  # radius
        self.x = np.zeros((n, 3))  # positions
        self.v = np.zeros((n, 3))  # velocities

        # Fluid
        self.rho = rho  # density
        self.eta = eta  # viscosity

        # Time
        self.t = 0
        self.dt = 0.1

    def update(self):
        # Update positions
        self.x += self.v * self.dt

        # Gravitational force
        F_grav = np.array([0.0, 0.0, -self.rho * 4/3 * np.pi * self.r**3])

        # Update velocities
        for i in range(self.n):
            self.v[i] = F_grav / (6 * np.pi * self.eta * self.r)
            for j in range(self.n):
                if i != j:
                    self.v[i] += self.S(j)(i)

        # Update time
        self.t += self.dt

    def run(self, steps):
        states = []
        for _ in range(steps):
            self.update()
            states.append(self.x.copy())
        return states

    def S(self, m):
        F_m = np.array([0.0, 0.0, -self.rho * 4/3 * np.pi * self.r**3])
        return lambda i: self.stokeslet(self.x[i], self.x[m], F_m)

    def stokeslet(self, x, x_0, F) -> np.ndarray:
        r = np.linalg.norm(x - x_0)
        c = 1/(8*np.pi*self.eta*r)
        return c * (np.eye(3) + np.outer(x - x_0, x - x_0)/r**2) @ F


class SingleParticleSim(Simulation):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class NGonSim(Simulation):
    def __init__(self, n, R=1, **kwargs):
        super().__init__(n, **kwargs)
        # Initialise particles in a regular n-gon
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = (R * np.cos(theta), R * np.sin(theta), np.zeros(n))
        self.x = np.stack(points, axis=-1)


class CloudSim(Simulation):
    def __init__(self, n, R=1, **kwargs):
        super().__init__(n, **kwargs)
        # Initialise cloud randomly
        dir = np.random.normal(size=(3, n))
        dir /= np.linalg.norm(dir, axis=0)
        mag = R * np.random.random(n) ** 1/3
        self.x = (dir * mag).T
