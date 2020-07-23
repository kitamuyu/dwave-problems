from pyqubo import Array, Placeholder, solve_qubo
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import numpy as np

N = 8

Q = Array.create('Q', N*N, 'BINARY')
Q_shape = np.reshape(Q, (N, N))


H_column = sum((sum(i for i in Q_shape[:, column])-1)**2
               for column in range(N))

H_row = sum((sum(i for i in Q_shape[row, :])-1)**2
            for row in range(N))

# 左上から右下の斜め
H_diagonal = sum(
    sum(i for i in np.diag(Q_shape, k=k)) *
    (sum(i for i in np.diag(Q_shape, k=k))-1)
    for k in range(-N+1, N-1))

# 左下から右上の斜め
H_diagonal_f = sum(
    sum(i for i in np.diag(np.fliplr(Q_shape), k=k)) *
    (sum(i for i in np.diag(np.fliplr(Q_shape), k=k))-1)
    for k in range(-N+1, N-1))

A = Placeholder('A')
B = Placeholder('B')
C = Placeholder('C')

feed_dict = {'A': 1.0, 'B': 1.0, 'C': 1.0}

H = A * H_column + B * H_row + C * (H_diagonal + H_diagonal_f)
model = H.compile()
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# SA
sol = solve_qubo(qubo)
decoded_sol, broken, energy = model.decode_solution(
    sol, vartype="BINARY", feed_dict=feed_dict)
sol_list = []
for i in range(N*N):
    sol_list.append(sol[f'Q[{i}]'])
ans = np.reshape(sol_list, (N, N))
print(f'energy : {energy}')
print(ans)

# D-Wave
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(qubo, num_reads=100)
print(response)
