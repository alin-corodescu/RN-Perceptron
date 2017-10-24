import pickle, gzip, numpy, threading


# train_set[0][k] is the representation of the k-th number, train_set[1][k] is the tag\


class Perceptron:
    w = numpy.array([0 for i in range(784)])
    b = 0
    def train_perceptron(self, train_set, iterations, learn_rate, target_digit):
        allClasified = False

        while iterations > 0 and not allClasified:
            allClasified = True
            for i in range(0, len(train_set[0])):
                x = train_set[0][i]
                t = train_set[1][i]
                target = t == target_digit
                z = self.classify_instance(x)
                output = z > 0
                a = numpy.array([(int(target) - int(output)) * learn_rate])
                aux = numpy.multiply(x, a)
                self.w = self.w + aux
                self.b = self.b + (int(target) - int(output)) * learn_rate
                if output != target:
                    allClasified = False

            iterations -= 1

    def classify_instance(self, instance):
        return self.w.dot(instance) + self.b

# allClasified = False
# iterations = 20
# w = numpy.array([0 for i in range(0,784)])
# b = 0
# learn_rate = 0.005
# while iterations > 0 and not allClasified:
#     allClasified = True
#     for i in range(0,len(train_set[0])):
#         x = train_set[0][i]
#         t = train_set[1][i]
#         target = t == 0
#         z = w.dot(x) + b
#         output = activation(z)
#         w = w + (int(target) - int(output)) * x * learn_rate
#         b = b + (int(target) - int(output)) * learn_rate
#         if output != target:
#             allClasified = False
#
#     iterations-=1
#
# count = 0
# totalZeros = 0
# misclassified = 0
# for i in range(0, len(test_set[0])):
#     x = test_set[0][i]
#     t = test_set[1][i]
#     if t == 0:
#         totalZeros+=1
#     target = t == 0
#     z = w.dot(x) + b
#     output = activation(z)
#     if output and t != 0:
#         misclassified+=1
#     if output and t == 0:
#         count+=1
#
#
# print("Correctly classified : ",count)
# print("Misclassified : ", misclassified)
# print("Total zeros :",totalZeros)
# print("Recall : ",count / totalZeros)

if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    perceptrons = []
    threads = []
    for digit in range(10):
        per = Perceptron()
        perceptrons.append(per)
        thread  = threading.Thread(target=per.train_perceptron(train_set, 10, 0.005, digit))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

#     training finished
    correct = 0
    incorrect = 0
    for i in range(0, len(test_set[0])):
        x = test_set[0][i]
        t = test_set[1][i]
        results = [perceptron.classify_instance(x) for perceptron in perceptrons]
        recognised_digit = results.index(max(results))
        if recognised_digit == t:
            correct+=1
        else:
            incorrect+=1

    print("Correct instances : ", correct)
    print("Incorrect instances : ", incorrect)
    print("Precision :", correct/(correct + incorrect))
