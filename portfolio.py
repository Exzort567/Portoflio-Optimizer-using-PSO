import numpy as np
import pandas as pd

class PSO:
    def __init__(self):
        self.particles = None
        self.assets = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = None
    
    def initialize(self, NoOfParticles, assets):
        self.particles = NoOfParticles
        self.assets = assets
    
    def y(self, particle):
        return self.objective_function(particle)
    
    def search(self, maxIteration, desiredY, w, c1, c2):
        self.optimize(maxIteration, desiredY, w, c1, c2)
        
    def objective_function(self, x):
        risk_free_rate = 0.01
        expected_return = np.dot(x, self.data.mean().values)
        standard_deviation = np.sqrt(np.dot(x.T, np.dot(self.data.cov(), x)))
        sharpe_ratio = (expected_return - risk_free_rate) / standard_deviation
        return -sharpe_ratio

    def optimize(self, maxIteration, desiredY, w, c1, c2):
        self.maxIteration = maxIteration
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        data = self.data
        
        swarm = np.random.rand(self.particles, self.assets)
        swarm = swarm / swarm.sum(axis=1, keepdims=True)

        velocities = np.zeros((self.particles, self.assets))
        pbest = swarm.copy()
        pbest_scores = np.array([self.objective_function(p) for p in pbest])
        gbest = pbest[pbest_scores.argmin()]
        gbest_score = pbest_scores.min()

        for i in range(self.maxIteration):
            velocities = self.w * velocities + self.c1 * np.random.rand(self.particles, self.assets) * (pbest - swarm) + self.c2 * np.random.rand(self.particles, self.assets) * (gbest - swarm)
            swarm = swarm + velocities
            swarm = np.clip(swarm, 0, 1)
            swarm = swarm / swarm.sum(axis=1, keepdims=True)
            swarm_scores = np.array([self.objective_function(p) for p in swarm])
            mask = swarm_scores < pbest_scores
            pbest[mask] = swarm[mask]
            pbest_scores[mask] = swarm_scores[mask]

            if swarm_scores.min() < gbest_score:
                gbest = swarm[swarm_scores.argmin()]
                gbest_score = swarm_scores.min() 
            print(f"Iteration {i+1}: Global best score: {-gbest_score}")

        self.pbest = pbest
        self.pbest_scores = pbest_scores
        self.gbest = gbest
        self.gbest_score = gbest_score
        
        optimal_allocation = {f'Asset {idx+1}': val for idx, val in enumerate(gbest)}
        print("Optimal Portfolio Allocation:")
        for asset, allocation in optimal_allocation.items():
            print(f"- {asset}: {allocation:.2%}")

# Example usage
data = pd.DataFrame(np.array([[20, 23, 62, 22],
                               [20, 11, 73, 21],
                               [16, 10, 47, 30],
                               [21, 11, 12, 13],
                               [21, 11, 2, 13],
                               [21, 17, 9, 12],
                               [21, 13, 9, 3],
                               [25, 11, 1, 13],
                               [11, 11, 4, 15],
                               [9, 11, 2, 13],
                               [2, 15, 57, 3],
                               [21, 11, 9, 8]]))

particles = 12
assets = 4
maxIteration = 100
w = 0.9
c1 = 2
c2 = 2

pso = PSO()
pso.data = data
pso.initialize(particles, assets)
pso.search(maxIteration, desiredY=None, w=w, c1=c1, c2=c2)
