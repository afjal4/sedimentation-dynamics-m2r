from abc import ABC
import numpy as np


class Simulation(ABC):
    def __init__(self, n, r=0.01, rho=1, eta=1):
        # Particle
        self.n = n
        self.r = r  # size
        self.x = np.zeros((self.n, 3))  # positions
        self.v = np.zeros((self.n, 3))  # velocities

        # Fluid
        self.rho = rho  # pressure
        self.eta = eta  # viscosity

        # Time
        self.t = 0
        self.dt = 0.1

    def update(self):
        # Update positions
        self.x += self.v * self.dt

        # Update velocities
        for i in range(self.n):
            F = np.array([0, 0, -self.rho * 4/3 * np.pi * self.r**3])  # gravity
            for j in range(self.n):
                if i != j:
                    F += self.S(j)(i)  # stokeslet interaction
            self.v[i] = F / (6 * np.pi * self.eta * self.r)  # Stokes' law

        # Update time
        self.t += self.dt

    def run(self, steps):
        states = []
        for _ in range(steps):
            self.update()
            states.append(self.x.copy())
        return states

    def S(self, m):
        return lambda i: self.stokeslet(self.x[i], self.x[m], self.v[m])

    def stokeslet(self, x, x_0, F) -> np.ndarray:
        r = np.linalg.norm(x - x_0)
        c = 1/(8*np.pi*self.eta*r)
        return c * (np.eye(3) + np.outer(x - x_0, x - x_0)/r**2) * F
    
    def render(self):
        pass


class SingleParticleSim(Simulation):
    def __init__(self):
        super().__init__(1)


class CloudSim(Simulation):
    def __init__(self, n, cloud_r=1):
        super().__init__(n)
        # Initialise cloud randomly
        dir = np.random.normal(size=(3, n))
        dir /= np.linalg.norm(dir, axis=0)
        mag = cloud_r * np.random.random(n) ** 1/3
        self.x = dir * mag
