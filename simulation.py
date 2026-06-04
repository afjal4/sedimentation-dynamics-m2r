from abc import ABC
import numpy as np


class Simulation(ABC):
    def __init__(self, n, r=1, rho=1, eta=1):
        # Particle
        self.n = n
        self.r = r  # size
        self.x = np.zeros((self.n, 3))  # positions
        self.v = np.zeros((self.n, 3))  # velocities

        # Fluid
        self.rho = rho  # pressure
        self.eta = eta  # viscosity


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
        return dir * mag
