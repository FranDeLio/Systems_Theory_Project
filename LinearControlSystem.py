import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable


class LinearControlSystem:

    def __init__(
        self,
        input_parameters: dict,
        u: Callable,
        z: Callable,
        time: float

    ):
        self.u = u
        self.z = z
        self.time = time
        self.input_parameters = input_parameters
        m1, d1, k1, m2, d2, k2 = input_parameters.values()
        self.A = np.array([
            [0, -(k2 + k1) / m1, 0, k2 / m1],
            [1, -(d2 + d1) / m1, 0, d2 / m1],
            [0, k2 / m2, 0, -k2 / m2],
            [0, d2 / m2, 1, -d2 / m2]
        ])
        self.B = np.array([k1 / m1, d1 / m1, 0, 0]).reshape(-1, 1)
        self.d1 = d1
        self.y1_height_baseline = 0.3
        self.y2_height_baseline = 0.6

    def compute_eigenvalues(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.A)
        return print("Eigenvalues:", eigenvalues)

    def compute_max_elevation(self):
        # Compute maximum elevation difference y2(t) - y1(t)
        elevation_diff = self.x_values[:, 3] - self.x_values[:, 1]

        max_elevation = np.max(elevation_diff)
        max_time = self.time_steps[np.argmax(elevation_diff)]

        print(f"Maximum Elevation Difference (y2 - y1): {max_elevation:.4f} m")
        print(f"Time of Maximum Elevation: t = {max_time:.4f} s")

        return max_elevation, max_time

    def compute_speeds(self):
        C = np.array([
            [0, 1, 0, 0],  # y1'(t)
            [0, 0, 0, 1]   # y2'(t)
        ])
        speeds = []
        for x, u_val in zip(self.x_values, self.u_values):
            x = x.reshape(-1, 1)
            y_dot = C @ (self.A @ x + self.B * u_val)
            speeds.append(y_dot.flatten())
        speeds = np.array(speeds)
        return speeds

    def solve(self, initial_conditions: dict, step_size: float = 0.1):
        self.x_values = []
        self.u_values = []
        x = np.array(list(initial_conditions.values())).reshape(-1, 1)
        self.time_steps =  np.linspace(0, self.time, int(self.time / step_size))

        for t in self.time_steps:
            self.x_values.append(x.flatten())
            self.u_values.append(self.u(t))
            dx = self.A @ x + self.B * self.u(self.z(t))
            x = x + dx * step_size

    def get_y1(self):
        df = pd.DataFrame({
        'y1': np.array(self.x_values)[:, 1],
        'time_steps': self.time_steps,
        'd1': self.d1
        })
        return df


    def plot(self, plot_u=False, deviation_only=False):
        self.x_values = np.array(self.x_values)
        plt.figure(figsize=(10, 6))

        if deviation_only==True:
            y1_height_baseline = 0
            y2_height_baseline = 0
        else:
            y1_height_baseline = self.y1_height_baseline
            y2_height_baseline = self.y2_height_baseline
        
        plt.plot(self.time_steps, self.x_values[:, 1] + y1_height_baseline, label=r"$y_{1*} + y_1$")
        plt.plot(self.time_steps, self.x_values[:, 3] + y2_height_baseline, label=r"$y_{2*} + y_2$")

        if plot_u==True: 
            plt.plot(self.time_steps, self.u_values, label="u(t)")

        plt.xlabel("Time (s)")
        plt.ylabel("State Variables")
        plt.title("State Variables of the Linear Control System (Euler Method)")
        plt.suptitle(f"System Parameters: {self.input_parameters}", y=0.02, fontsize=10)
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0.5)
        plt.grid(True)

        plt.show()
