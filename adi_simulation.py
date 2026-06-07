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

    def update(self):  # Adi change
        # Update positions
        self.x += self.v * self.dt
        self.t += self.dt

        # Gravitational force
        F_grav = np.array([0.0, 0.0, -self.rho * 4/3 * np.pi * self.r**3 * 9.81])

        v_stokes = F_grav / (6 * np.pi * self.eta * self.r)

        # Update velocities in a more efficient way
        stokeslet_sum = np.zeros((self.n, 3))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    stokeslet_sum[i] += self.S(j)(i)
        
        self.v = v_stokes + stokeslet_sum


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
        # creates 3 arrays of length n for each direction
        dir = np.random.normal(size=(3, n))
        dir /= np.linalg.norm(dir, axis=0)
        # cube root so they don;t cluster around 0
        mag = R * np.random.random(n) ** (1/3)
        self.x = (dir * mag).T


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
    
        self.x = np.column_stack([x,y,z])

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
class TwoCloudSim(Simulation):
    def __init__(self, n, R=1, separation = 4.0, **kwargs):
        super().__init__(2*n, **kwargs)
        def random_sphere(n, R):
            dir = np.random.normal(size=(3, n))
            dir /= np.linalg.norm(dir, axis=0)
            mag = R * np.random.random(n) ** (1/3)
            return (dir * mag).T
        cloud1 = random_sphere(n, R)
        cloud2 = random_sphere(n, R)

        for particle in cloud2:
            particle[2] += separation

        self.x  = np.concatenate([cloud1, cloud2])
