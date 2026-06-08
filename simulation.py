from abc import ABC
import numpy as np


class Simulation(ABC):
    def __init__(self, n, r=1e-4, rho=1e4, eta=1e-3, fall=True):
        # Particle
        self.n = n
        self.r = r  # radius
        self.x = np.zeros((n, 3))  # positions
        self.v = np.zeros((n, 3))  # velocities

        # Fluid
        self.rho = rho  # density
        self.eta = eta  # viscosity
        self.g = 9.81

        # Fall
        self.fall = fall  # whether particles fall under gravity

        # Time
        self.t = 0
        self.dt = 0.1

    def update(self):
        # Update positions
        self.x += self.v * self.dt

        # Terminal velocity from gravity (Stokes drag)
        F_grav = np.array([0.0, 0.0, -self.rho * 4/3 * np.pi * self.r**3 * self.g])
        self.v = np.tile(F_grav / (6 * np.pi * self.eta * self.r), (self.n, 1))

        # Update velocities
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.v[i] += self.S(j)(i)

        # Update time
        self.t += self.dt

        # Re-center so center of mass stays at origin
        if not self.fall:
            self.x -= self.x.mean(axis=0)

    def run(self, steps):
        states = []
        for _ in range(steps):
            self.update()
            states.append(self.x.copy())
        return states

    def S(self, m):
        F_m = np.array([0.0, 0.0, -self.rho * 4/3 * np.pi * self.r**3 * self.g])
        return lambda i: self.stokeslet(self.x[i], self.x[m], F_m)

    def stokeslet(self, x, x_0, F) -> np.ndarray:
        r = np.linalg.norm(x - x_0)
        c = 1/(8*np.pi*self.eta*r)
        return c * (np.eye(3) + np.outer(x - x_0, x - x_0)/r**2) @ F


class SingleParticleSim(Simulation):
    def __init__(self, **kwargs):
        super().__init__(1, **kwargs)


class NGonSim(Simulation):
    def __init__(self, n, R=1.0, **kwargs):
        super().__init__(n, **kwargs)
        # Initialise particles in a regular n-gon
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = (R * np.cos(theta), R * np.sin(theta), np.zeros(n))
        self.x = np.stack(points, axis=-1)


class CloudSim(Simulation):
    def __init__(self, n, R=1, **kwargs):
        super().__init__(n, **kwargs)
        self.x = self.random_sphere(n, R)

    def random_sphere(self, n, R):
        dir = np.random.normal(size=(3, n))
        dir /= np.linalg.norm(dir, axis=0)
        mag = R * np.random.random(n) ** (1/3)
        return (dir * mag).T


class SheetSim(Simulation):
    def __init__(self, n, R=1, **kwargs):
        super().__init__(n, **kwargs)

        # Random direction in 2d (so only y-z plane)
        dir = np.random.normal(size=(2, n))
        dir /= np.linalg.norm(dir, axis=0)

        mag = R * np.random.random(n) ** (1/2)

        y = dir[0] * mag
        z = dir[1] * mag
        x = np.zeros(n)

        self.x = np.column_stack([x, y, z])


# two particles one above the other
class VerticalTwoParticleSim(Simulation):
    def __init__(self, separation=1.0, **kwargs):
        super().__init__(2, **kwargs)
        self.x = np.array([[0, 0, separation/2], [0,0, -separation/2]])


# two particles side by side
class HorizontalTwoParticleSim(Simulation):
    def __init__(self, separation=1.0, **kwargs):
        super().__init__(2, **kwargs)
        self.x = np.array([[separation/2, 0, 0], [-separation/2,0, 0]])


# two separate clouds released one above the other
class TwoCloudSim(CloudSim):
    def __init__(self, n, R=1, separation=4.0, **kwargs):
        super().__init__(2*n, **kwargs)
        cloud1 = self.random_sphere(n, R)
        cloud2 = self.random_sphere(n, R)

        for particle in cloud2:
            particle[2] += separation

        self.x = np.concatenate([cloud1, cloud2])
