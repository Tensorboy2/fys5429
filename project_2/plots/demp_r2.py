import numpy as np
import matplotlib.pyplot as plt

r2_raw = np.linspace(-1, 1, 200)
epsilon = 1e-6

# Hybrid transformation: project negatives using 1/|R2|
r2_safe = np.where(r2_raw > 0, r2_raw, epsilon / (np.abs(r2_raw) + epsilon))
neg_log_r2 = -np.log(r2_safe)

plt.figure(figsize=(6, 4))
plt.plot(r2_raw, neg_log_r2, label=r'$-\ln(\text{hybrid}(R^2))$')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(1, color='gray', linestyle=':', alpha=0.5)
plt.xlabel(r'$R^2$')
plt.ylabel('Transformed Metric')
plt.title('Hybrid Log Penalty of $R^2$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
