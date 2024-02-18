import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PSO:
    def __init__(self, data, particles, assets, iterations, w, c1, c2):
        self.data = data
        self.particles = particles
        self.assets = assets
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def objective_function(self, x):
        risk_free_rate = 0.01
        #calculate the expected return
        expected_return = np.dot(x, self.data.mean().values)
        #calculate the standar deviation of the portfolio
        standard_deviation = np.sqrt(np.dot(x.T, np.dot(self.data.cov(), x)))
        """
        To calculate the Sharpe ratio investors can subtract the risk-free rate of return from the expected rate of return, and then divide that result by the standard deviation (the asset's volatility.)

        """
        sharpe_ratio = (expected_return - risk_free_rate) / standard_deviation
        #negative since we want to maximize it
        return -sharpe_ratio

    def optimize(self):
        #initialize the swarm
        swarm = np.random.rand(self.particles, self.assets)
        swarm = swarm / swarm.sum(axis=1, keepdims=True)

        velocities = np.zeros((self.particles, self.assets))
        pbest = swarm.copy()
        pbest_scores = np.array([self.objective_function(p) for p in pbest])
        gbest = pbest[pbest_scores.argmin()]
        gbest_score = pbest_scores.min()

        #PSO loop
        for i in range(self.iterations):
            #update the velocities
            velocities = w * velocities + c1 * np.random.rand(self.particles, self.assets) * (pbest - swarm) + c2 * np.random.rand(self.particles, self.assets) * (gbest - swarm)
            swarm = swarm + velocities
            swarm = np.clip(swarm, 0, 1)
            swarm = swarm / swarm.sum(axis=1, keepdims=True)
            #evaluate the new positions
            swarm_scores = np.array([self.objective_function(p) for p in swarm])
            #update the pbest and pbest scores
            mask = swarm_scores < pbest_scores
            pbest[mask] = swarm[mask]
            pbest_scores[mask] = swarm_scores[mask]

            #update the gbest and gbest score
            if swarm_scores.min() < gbest_score:
                gbest = swarm[swarm_scores.argmin()]
                gbest_score = swarm_scores.min() 
            print(f"Iteration {i+1}: Global best score: {-gbest_score}")

        optimal_allocation = {f'Asset {idx+1}': val for idx, val in enumerate(gbest)}
        print("Optimal Portfolio Allocation:")
        for asset, allocation in optimal_allocation.items():
            print(f"- {asset}: {allocation:.2%}")

        plt.plot(-pbest_scores)
        plt.xlabel("Iteration")
        plt.ylabel("Sharpe Ratio")
        plt.title("Portfolio Optimization")
        plt.show()






data = pd.DataFrame(np.array([[20, 23, 62, 52],
                               [20, 11, 73, 21],
                               [16, 10, 47, 30],
                               [21, 11, 9, 13],
                               [21, 11, 6, 13],
                               [21, 17, 9, 12],
                               [21, 13, 9, 3],
                               [25, 11, 9, 13],
                               [11, 11, 9, 15],
                               [9, 11, 9, 13],
                               [2, 15, 9, 3],
                               [21, 11, 9, 8]]))

# Define the PSO parameters
particles = 50  # Number of particles in the swarm
assets = 4      # Number of assets in the portfolio
iterations = 100  # Number of iterations to run the PSO
w = 0.9           # Inertia weight
c1 = 2            # Cognitive parameter
c2 = 2            # Social parameter

pso = PSO(data, particles, assets, iterations, w, c1, c2)
pso.optimize()