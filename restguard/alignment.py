import numpy as np


def symmetric_function(matrix, function, floor=1e-8):
    value, vector = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vector * function(np.maximum(value, floor))) @ vector.T


def covariance(segment, shrinkage=0.1, epsilon=1e-6):
    centered = segment - segment.mean(1, keepdims=True)
    empirical = centered @ centered.T / max(1, segment.shape[1] - 1)
    dimension = len(segment)
    return ((1 - shrinkage) * empirical
            + shrinkage * np.trace(empirical) / dimension * np.eye(dimension)
            + epsilon * np.eye(dimension))


class RestGuard:
    def __init__(self, reference, shrinkage=0.1, epsilon=1e-6, rate=0.1, radius=0.5,
                 bounded=True, euclidean=False):
        self.whitener = symmetric_function(reference, lambda value: value ** -0.5)
        self.state = np.zeros_like(reference)
        self.average = np.eye(len(reference))
        self.shrinkage = shrinkage
        self.epsilon = epsilon
        self.rate = rate
        self.radius = radius
        self.bounded = bounded
        self.euclidean = euclidean
        self.updates = 0

    def update(self, rest):
        reference = self.whitener @ covariance(rest, self.shrinkage, self.epsilon) @ self.whitener
        if self.euclidean:
            self.average = (1 - self.rate) * self.average + self.rate * reference
        else:
            logarithm = symmetric_function(reference, np.log)
            candidate = (1 - self.rate) * self.state + self.rate * logarithm
            norm = np.linalg.norm(candidate, "fro")
            self.state = candidate * min(1, self.radius / norm) if self.bounded and norm else candidate
        self.updates += 1

    def transform(self, trial):
        if self.euclidean:
            correction = symmetric_function(self.average, lambda value: value ** -0.5)
        else:
            value, vector = np.linalg.eigh((self.state + self.state.T) / 2)
            correction = (vector * np.exp(-value / 2)) @ vector.T
        return (correction @ self.whitener @ trial).astype(np.float32)

    def displacement_bound(self):
        if self.euclidean:
            eigenvalues = np.linalg.eigvalsh(self.average).clip(1e-8)
            return float(np.max(np.abs(eigenvalues ** -0.5 - 1)))
        return float(np.expm1(np.linalg.norm(self.state, "fro") / 2))
