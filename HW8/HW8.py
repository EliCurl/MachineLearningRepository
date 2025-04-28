import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



# 1. Prepare Dataset
class Dataset(Dataset):
    def __init__(self):
        df = pd.read_csv("house_prices.csv")

        feature_cols = ['sqft_lot', 'bedrooms', 'bathrooms','sqft_living','sqft_lot15','sqft_living15','floors','grade','sqft_basement','yr_built']

        df = df.dropna(subset=feature_cols + ['price'])

        X = df[feature_cols].values
        y = df['price'].values

        # Standardize features (important for ANN training!)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.len = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len

    def get_numpy(self):
        return self.X.numpy(), self.y.numpy()


# 2. Build an ANN
class CustomANN(nn.Module):
    def __init__(self):
        # 3 hidden layers
        super(CustomANN, self).__init__()
        # First layer going from 10 to 32 variables
        self.fc1 = nn.Linear(10, 32)
        # Second layer going from 32 to 16
        self.fc2 = nn.Linear(32, 16)
        # Final layer going from 16 to 1
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.2)  # Dropout regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


# 3. Training loop
def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=25):
    cd = Dataset()

    # Create a data loader that shuffles each time and allows for the last batch to be smaller
    # than the rest of the epoch if the batch size doesn't divide the training set size
    curve_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create my neural network
    curve_network = CustomANN()

    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')

    # Select the optimizer
    optimizer = torch.optim.Adam(curve_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(curve_loader, 0):
            x, y = data

            optimizer.zero_grad()  # resets gradients to zero

            output = curve_network(x)  # evaluate the neural network on x

            loss = mse_loss(output.view(-1), y)  # compare to the actual label value

            loss.backward()  # perform back propagation

            optimizer.step()  # perform gradient descent with an Adam optimizer

            running_loss += loss.item()  # update the total loss

        # every epoch_display epochs give the mean square error since the last update
        # this is averaged over multiple epochs
        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(cd) * epoch_display):.6f}")
            running_loss = 0.0
    return curve_network, cd


cn, cd = trainNN(epochs=300)


with torch.no_grad():
    y_pred = cn(cd.X).view(-1)

X_numpy, y_numpy = cd.get_numpy()

print(f"MSE (fully trained): {np.average((y_numpy-np.array(y_pred))**2)}")

svr = SVR()
svr.fit(X_numpy, y_numpy)
print(f"SVR Number of Support Vectors: {svr.support_vectors_.shape}")
svr_pred = svr.predict(X_numpy)
print(f"SVR MSE: {np.average((svr_pred - y_numpy) ** 2)}")


plt.scatter(y_numpy, y_pred.numpy(), s=1.0, c='b')
plt.plot([y_numpy.min(), y_numpy.max()], [y_numpy.min(), y_numpy.max()], 'r--')  # 45-degree line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
