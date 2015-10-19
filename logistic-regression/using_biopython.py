from Bio import LogisticRegression
import numpy as np


all_data = np.loadtxt("../datasets/iris/iris.data", delimiter=",",
                      dtype="float, float, float, float, S11")

xs = []
ys = []

for i in all_data:
    if 'virgi' not in str(i[-1]):
        xs.append([i[0], i[1], i[2], i[3]])
        if 'setosa' in str(i[-1]):
            ys.append(0)
        else:
            ys.append(1)

test_xs = xs.pop()
test_ys = ys.pop()

def show_progress(iteration, loglikelihood):
    print("Iteration:", iteration, "Log-likelihood function:", loglikelihood)

model = LogisticRegression.train(xs, ys, update_fn=show_progress)
print("This should be Iris-versic (1): {}".format(LogisticRegression.classify(model, test_xs)))
