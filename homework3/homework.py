import numpy as np
import pandas as pd

class MLP:
    def __init__(self, input_size, hidden_layers, output_classes):
        self.layers = [input_size] + hidden_layers + [output_classes]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) / np.sqrt(self.layers[i]) for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, layer)) for layer in self.layers[1:]]
        self.v_dw, self.s_dw = [np.zeros_like(w) for w in self.weights], [np.zeros_like(w) for w in self.weights]
        self.v_db, self.s_db = [np.zeros_like(b) for b in self.biases], [np.zeros_like(b) for b in self.biases]
        self.epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
    
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        self.activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.activations[-1], w) + b
            a = self.relu(z) if i < len(self.weights) - 1 else self.softmax(z)
            self.activations.append(a)
        return self.activations[-1]

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_true_encoded = np.eye(self.layers[-1])[y_true]
        log_likelihood = -np.log(y_pred[range(m)] * y_true_encoded + self.epsilon)
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y):
        y_true = np.eye(self.layers[-1])[y]
        delta = self.activations[-1] - y_true

        deltas = [delta]
        for i in reversed(range(len(self.activations) - 2)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self.relu_derivative(self.activations[i + 1])
            deltas.insert(0, delta)

        m = y.shape[0]
        grads_w = [np.dot(self.activations[i].T, deltas[i]) / m for i in range(len(self.weights))]
        grads_b = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.biases))]

        return grads_w, grads_b

    def update_weights(self, grads_w, grads_b, eta=0.01):
        # Update weights with Adam optimization
        for i in range(len(self.weights)):
            self.v_dw[i] = self.beta1 * self.v_dw[i] + (1 - self.beta1) * grads_w[i]
            self.s_dw[i] = self.beta2 * self.s_dw[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.v_db[i] = self.beta1 * self.v_db[i] + (1 - self.beta1) * grads_b[i]
            self.s_db[i] = self.beta2 * self.s_db[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            # Correct bias
            v_dw_corr = self.v_dw[i] / (1 - self.beta1 ** (i + 1))
            s_dw_corr = self.s_dw[i] / (1 - self.beta2 ** (i + 1))
            v_db_corr = self.v_db[i] / (1 - self.beta1 ** (i + 1))
            s_db_corr = self.s_db[i] / (1 - self.beta2 ** (i + 1))

            # Update weights and biases
            self.weights[i] -= eta * v_dw_corr / (np.sqrt(s_dw_corr) + self.epsilon)
            self.biases[i] -= eta * v_db_corr / (np.sqrt(s_db_corr) + self.epsilon)

    def fit(self, X, y, epochs=1000, eta=0.01, eta_decay=0.0001):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            grads_w, grads_b = self.backward(y)
            eta *= (1 / (1 + eta_decay * epoch))  # Decay learning rate
            self.update_weights(grads_w, grads_b, eta)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        y_hat = self.forward(X)
        return np.argmax(y_hat, axis=1)  # Return the class with the highest probability


def label_encoder(series):
    unique_classes = series.unique()
    class_map = {label: idx for idx, label in enumerate(unique_classes)}
    return series.map(class_map), class_map

def standard_scaler(data, epsilon=1e-8):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + epsilon), mean, std  # Adding epsilon to avoid division by zero

def accuracy_score(true_labels, predicted_labels):
    correct_count = np.sum(true_labels == predicted_labels)
    return (correct_count / len(true_labels)) * 100

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def add_nearest_neighbors_features(df, n_neighbors=3):
    lat = df['LATITUDE'].to_numpy()
    lon = df['LONGITUDE'].to_numpy()
    sqft = df['PROPERTYSQFT'].to_numpy()
    price_sqft = df['PRICE'] / df['PROPERTYSQFT']
    for i in range(1, n_neighbors+1):
        df[f'neighbor_{i}_distance'] = 0
        df[f'neighbor_{i}_beds'] = 0
        df[f'neighbor_{i}_sqft'] = 0
        df[f'neighbor_{i}_price_sqft'] = 0
    for i in range(len(df)):
        distances = euclidean_distance(lat[i], lon[i], lat, lon)
        nearest_indices = np.argsort(distances)[1:n_neighbors+1]
        for j, idx in enumerate(nearest_indices, start=1):
            df.at[i, f'neighbor_{j}_distance'] = distances[idx]
            df.at[i, f'neighbor_{j}_sqft'] = sqft[idx]
            df.at[i, f'neighbor_{j}_price_sqft'] = price_sqft[idx]
    return df

def process_test_dataset(test_data_file,  model, numerical_features, mean, std):
    test_data = pd.read_csv(test_data_file)

    numerical_features = [feature for feature in numerical_features if feature != 'BEDS']
    
    test_data.loc[:, numerical_features] = (test_data[numerical_features] - mean) / (std + 1e-8)

    X_test = test_data[numerical_features]
    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame({'BEDS': y_pred})
    output_filename = 'output.csv'
    predictions_df.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')


def main():
    # Load datasets
    train_data = pd.read_csv('train_data.csv')
    train_labels = pd.read_csv('train_label.csv')

    # Combine datasets for feature engineering
    train = pd.concat([train_data, train_labels], axis=1)

    # Feature extraction and other preprocessing
    numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('BEDS')
    X_train = train[numerical_features]
    y_train = train['BEDS'] 

    # Standard scaling
    scaled_X_train, mean, std = standard_scaler(X_train)
    y_train = y_train.apply(lambda x: min(x, 13))

    # Initialize and train MLP model
    mlp_model = MLP(input_size=len(numerical_features), hidden_layers=[150, 150], output_classes=14)
    mlp_model.fit(scaled_X_train, y_train, epochs=500, eta=0.01, eta_decay = 0.0005)

    # Test data set
    test_dataset = 'test_data.csv'

    # Process the test data
    process_test_dataset(test_dataset, mlp_model, numerical_features, mean, std)


if __name__ == "__main__":
    main()
