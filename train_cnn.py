import random

from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


def fix_seeds():
    seed = 10
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    fix_seeds()

    num_classes = 10
    img_dim = 28
    img_depth = 1
    save_path = "C:/Users/LARISA/git/conv-net/output/output.txt"

    model = build_model(num_classes=num_classes, img_dim=img_dim, img_depth=img_depth, save_path=save_path)
    _ = train(model, img_dim)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, _, _, w3, w4, b1, b2, _, _, b3, b4] = params

    # Plot cost
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()

    # Get test data
    # m = 60000
    m = 1000
    X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
    y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m, 1)

    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f" % (float(corr / len(test_data) * 100)))
    x = np.arange(10)
    digit_recall = [x / y for x, y in zip(digit_correct, digit_count)]
    plt.xlabel('Clothes')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x, digit_recall)
    plt.show()
