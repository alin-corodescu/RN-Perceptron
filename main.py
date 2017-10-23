import pickle, gzip, numpy, random

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()


# train_set[0][k] is the representation of the k-th number, train_set[1][k] is the tag\


def activation(input):
    return input > 0

allClasified = False
iterations = 1
w = numpy.array([random.random() for i in range(0,784)])
b = - random.random() * 784
learn_rate = 0.01
while iterations > 0:
    # allClasified = True
    for i in range(0,len(train_set[0])):
        x = train_set[0][i]
        t = train_set[1][i]
        target = t == 0
        z = w.dot(x)
        output = activation(z)

        w = w + (target - output) * x * learn_rate
        b = b + (target - output) * learn_rate
            # if output != t:
                # allClasified = False

    iterations-=1

count = 0
totalZeros = 0
misclassified = 0
for i in range(0, len(test_set[0])):
    x = test_set[0][i]
    t = test_set[1][i]
    if t == 0:
        totalZeros+=1
    target = t == 0
    z = w.dot(x) + b
    output = activation(z)
    if output and t != 0:
        misclassified+=1
    if output and t == 0:
        count+=1


print(count)
print(misclassified)
print(count / totalZeros)
