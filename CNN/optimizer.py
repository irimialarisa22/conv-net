from tqdm import tqdm

import numpy as np


class AdamOptimizer:
    def __init__(self, num_classes=10, img_dim=28, img_depth=1, lr=0.01, beta1=0.95, beta2=0.99, batch_size=32,
                 num_epochs=2):
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.img_depth = img_depth
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss = None
        self.callbacks = {}
        self.frequency = 1

    def set_loss(self, loss_fct):
        self.loss = loss_fct

    def addCallbacks(self, callback_dict):
        self.callbacks.update(callback_dict)

    def setFrequency(self, freq):
        self.frequency = freq

    def train(self, model, train_data):
        cost = []

        print("LR:" + str(self.lr) + ", Batch Size:" + str(self.batch_size))

        for indx, epoch in enumerate(range(self.num_epochs)):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + self.batch_size] for k in range(0, train_data.shape[0], self.batch_size)]

            t = tqdm(batches)
            for x, batch in enumerate(t):
                params, cost = self.adamGD(batch, self.num_classes, self.img_dim, self.img_depth, model, cost)

                t.set_description("Cost: %.2f" % (cost[-1]))

        if indx % self.frequency == 0:
            to_save = [params, cost]
            self.callbacks["SaveModel"](to_save)

        return cost

    def adamGD(self, batch, num_classes, dim, n_c, model, cost):
        lr = self.lr
        beta1 = self.beta1
        beta2 = self.beta2
        """
        Update the parameters through Adam gradient descent.
        """
        global grads
        X = batch[:, 0:-1]  # get batch inputs
        X = X.reshape(len(batch), n_c, dim, dim)
        Y = batch[:, -1]  # get batch labels

        cost_ = 0
        batch_size = len(batch)

        diff_grad = None
        params = model.params()
        for _ in range(3):
            weights = []
            for w_b in params:  # FULL PYTHON LIST
                if w_b is not None:
                    weights.append(np.zeros(w_b.shape))
                else:
                    weights.append(None)
            if diff_grad is None:
                diff_grad = [weights]
            else:
                diff_grad.append(weights)

        # full forward run
        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

            # Collect Gradients for training example
            probs, feed_results = model.full_forward(x)
            loss_value = self.loss(probs, y)  # categorical cross-entropy loss value
            grads_w, grads_b = model.full_backprop(probs, y, feed_results)

            grads = []
            grads.extend(grads_w)
            grads.extend(grads_b)
            for xx in range(model.no_layers() * 2):
                if grads[xx] is not None:
                    diff_grad[0][xx] += grads[xx]
                else:
                    diff_grad[0][xx] = None

            cost_ += loss_value

        # backprop
        for my_i in range(8):
            if diff_grad[0][my_i] is None or diff_grad[1][my_i] is None or diff_grad[2][my_i] is None:
                continue
            diff_grad[1][my_i] = beta1 * diff_grad[1][my_i] + (1 - beta1) * diff_grad[0][my_i] / batch_size  # momentum update
            diff_grad[2][my_i] = beta2 * diff_grad[2][my_i] + (1 - beta2) * (diff_grad[0][my_i] / batch_size) ** 2  # RMSProp update
            # combine momentum and RMSProp to perform update with Adam
            params[my_i] -= lr * diff_grad[1][my_i] / np.sqrt(diff_grad[2][my_i] + 1e-7)

        cost_ = cost_ / batch_size
        cost.append(cost_)

        model.set_params(params)
        return model.params(), cost