This project is mostly about coming up with, and solving the following system of ODEs:

```math
\begin{align*}
& m_2 \ddot{y}_2 + k_2 (y_2 - y_1) + d_2 (\dot{y}_2 - \dot{y}_1) = 0 \\
& m_1 \ddot{y}_1 + k_2 (y_1 - y_2) + d_2 (\dot{y}_1 - \dot{y}_2) + k_1 (y_1 - u) + d_1 (\dot{y}_1 - \dot{u}) = 0
\end{align*}
```
<br />

We numerically solved the system based on its state-space representation, which is presented below:


```math
\begin{bmatrix}
\dot{X}_1 \\ \dot{X}_2 \\ \dot{X}_3 \\ \dot{X}_4
\end{bmatrix}
=
\begin{bmatrix}
0 & -\frac{k_2 + k_1}{m_1} & 0 & \frac{k_2}{m_1} \\
1 & -\frac{d_2 + d_1}{m_1} & 0 & \frac{d_2}{m_1} \\
0 & \frac{k_2}{m_2} & 0 & -\frac{k_2}{m_2} \\
0 & \frac{d_2}{m_2} & 1 & -\frac{d_2}{m_2}
\end{bmatrix}
\begin{bmatrix}
X_1 \\ X_2 \\ X_3 \\ X_4
\end{bmatrix}
+
\begin{bmatrix}
\frac{k_1}{m_1} \\ \frac{d_1}{m_1} \\ 0 \\ 0
\end{bmatrix} u
```
```math
\begin{bmatrix}
y_1 \\ y_2
\end{bmatrix}
=
\begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_1 \\ X_2 \\ X_3 \\ X_4
\end{bmatrix}
```

## Getting Started

The recommended python version is `3.10.12`, and your working directory should be set to the root of this repository.

```bash
export PYTHONPATH=/your/path/Systems_Theory_Project
```

This project uses poetry version `1.8.2` to manage its dependencies. You can install poetry, download the project requirements and initialize your poetry virtual environment by executing the following lines on your terminal.

```bash
pip install poetry==1.8.2
poetry install
poetry shell
```
