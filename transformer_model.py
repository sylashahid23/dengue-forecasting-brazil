import random
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import matplotlib.pyplot as plt

from torch import nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.preprocessing import MinMaxScaler
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ======================================================
# Configuration
# ======================================================

DATA_FILE = "Brazil_Final_Dataset.csv"

FEATURES = [
    "dengue_cases",
    "temperature",
    "humidity",
    "precipitation"
]

TARGET = "dengue_cases"

SEQUENCE_LENGTH = 12

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv(DATA_FILE)

df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values(
    ["state", "year_month"]
).reset_index(drop=True)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

print(df.shape)

# ======================================================
# Chronological Split
# ======================================================

train_df = df[df["year_month"] < "2022-01-01"].copy()

val_df = df[
    (df["year_month"] >= "2022-01-01")
    &
    (df["year_month"] < "2023-01-01")
].copy()

test_df = df[df["year_month"] >= "2023-01-01"].copy()

print("\nTrain:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# ======================================================
# Scaling (Per State)
# ======================================================

scalers = {}

for state in train_df["state"].unique():

    scaler = MinMaxScaler()

    train_mask = train_df["state"] == state
    val_mask = val_df["state"] == state
    test_mask = test_df["state"] == state

    train_df.loc[train_mask, FEATURES] = scaler.fit_transform(
        train_df.loc[train_mask, FEATURES]
    )

    val_df.loc[val_mask, FEATURES] = scaler.transform(
        val_df.loc[val_mask, FEATURES]
    )

    test_df.loc[test_mask, FEATURES] = scaler.transform(
        test_df.loc[test_mask, FEATURES]
    )

    scalers[state] = scaler

print("\nPer-state scaling complete.")

# ======================================================
# Sequence Generator
# ======================================================

def create_sequences(train_df, eval_df, sequence_length):
    """
    Creates sequences for validation/test while preserving metadata.
    """

    X = []
    y = []
    metadata = []

    for state in train_df["state"].unique():

        history = train_df[
            train_df["state"] == state
        ].tail(sequence_length)

        current = eval_df[
            eval_df["state"] == state
        ]

        combined = pd.concat(
            [history, current],
            ignore_index=True
        )

        values = combined[FEATURES].values
        target = combined[TARGET].values

        for i in range(len(current)):

            X.append(
                values[i:i + sequence_length]
            )

            y.append(
                target[i + sequence_length]
            )

            metadata.append({
                "state": current.iloc[i]["state"],
                "year_month": current.iloc[i]["year_month"]
            })

    return (
        np.array(X),
        np.array(y),
        pd.DataFrame(metadata)
    )


def create_training_sequences(dataframe, sequence_length):

    X = []
    y = []

    for state in dataframe["state"].unique():

        state_df = dataframe[
            dataframe["state"] == state
        ]

        values = state_df[FEATURES].values
        target = state_df[TARGET].values

        for i in range(len(state_df) - sequence_length):

            X.append(
                values[i:i + sequence_length]
            )

            y.append(
                target[i + sequence_length]
            )

    return np.array(X), np.array(y)


# ======================================================
# Create Sequences
# ======================================================

X_train, y_train = create_training_sequences(
    train_df,
    SEQUENCE_LENGTH
)

X_val, y_val, val_meta = create_sequences(
    train_df,
    val_df,
    SEQUENCE_LENGTH
)

X_test, y_test, test_meta = create_sequences(
    val_df,
    test_df,
    SEQUENCE_LENGTH
)

print("\nSequence generation complete.\n")

print("Training:")
print(X_train.shape, y_train.shape)

print("\nValidation:")
print(X_val.shape, y_val.shape)

print("\nTest:")
print(X_test.shape, y_test.shape)
# ======================================================
# Convert to PyTorch Tensors
# ======================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
).view(-1, 1)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)

print("\nTensor conversion complete.")

print("Training tensor:", X_train.shape)
print("Validation tensor:", X_val.shape)
print("Test tensor:", X_test.shape)
# ======================================================
# Create DataLoaders
# ======================================================

BATCH_SIZE = 32

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoaders created.")

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

# ======================================================
# Transformer Model
# ======================================================

INPUT_SIZE = len(FEATURES)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
DROPOUT = 0.2


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Linear(INPUT_SIZE, 64)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, OUTPUT_SIZE)

    def forward(self, x):

        x = self.input_projection(x)
        x = self.transformer(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x


model = TransformerModel()

print("\nTransformer Model Created\n")

print(model)
# ======================================================
# Training Configuration
# ======================================================

LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
best_val_loss = float("inf")
patience_counter = 0

print("\nTraining configuration ready.")
# ======================================================
# Training Loop
# ======================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train: {train_loss:.6f} | "
        f"Validation: {val_loss:.6f}"
    )

    # ------------------------------------
    # Early Stopping
    # ------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            "saved_models/best_transformer_model.pth"
        )

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break

# ======================================================
# Load Best Model
# ======================================================

model.load_state_dict(
    torch.load("saved_models/best_transformer_model.pth")
)

print("\nBest model loaded.")

# ======================================================
# Test Evaluation
# ======================================================

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        predictions.extend(
            outputs.cpu().numpy().flatten()
        )

        actuals.extend(
            y_batch.cpu().numpy().flatten()
        )

predictions = np.array(predictions)
actuals = np.array(actuals)

# ======================================================
# Inverse Scaling (Per State)
# ======================================================

predictions_original = np.zeros(len(predictions))
actuals_original = np.zeros(len(actuals))

for state in test_meta["state"].unique():

    scaler = scalers[state]

    mask = test_meta["state"] == state

    state_predictions = predictions[mask]
    state_actuals = actuals[mask]

    dummy_pred = np.zeros((len(state_predictions), len(FEATURES)))
    dummy_true = np.zeros((len(state_actuals), len(FEATURES)))

    dummy_pred[:, 0] = state_predictions
    dummy_true[:, 0] = state_actuals

    predictions_original[mask] = scaler.inverse_transform(dummy_pred)[:, 0]
    actuals_original[mask] = scaler.inverse_transform(dummy_true)[:, 0]

predictions = np.maximum(predictions_original, 0)
actuals = actuals_original
# ======================================================
# Metrics
# ======================================================

rmse = np.sqrt(
    mean_squared_error(actuals, predictions)
)

mae = mean_absolute_error(
    actuals,
    predictions
)


print("\n==============================")
print("Test Results")
print("==============================")

print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")


# ======================================================
# Save Predictions
# ======================================================

results_df = test_meta.copy()

results_df["actual"] = actuals
results_df["predicted"] = predictions

results_df.to_csv(
    "results/transformer_predictions.csv",
    index=False
)

print("\nPredictions saved:")
print(results_df.head())
# ======================================================
# Metrics by State
# ======================================================

state_metrics = []

for state in results_df["state"].unique():

    state_df = results_df[
        results_df["state"] == state
    ]

    rmse = np.sqrt(
        mean_squared_error(
            state_df["actual"],
            state_df["predicted"]
        )
    )

    mae = mean_absolute_error(
        state_df["actual"],
        state_df["predicted"]
    )

    state_metrics.append({
        "state": state,
        "RMSE": rmse,
        "MAE": mae
    })

metrics_df = pd.DataFrame(state_metrics)

print("\nMetrics by State")
print(metrics_df)

metrics_df.to_csv(
    "results/transformer_metrics.csv",
    index=False
)
# ======================================================
# Plot Predictions by State
# ======================================================

for state in results_df["state"].unique():

    state_df = results_df[
        results_df["state"] == state
    ].copy()

    plt.figure(figsize=(12, 5))

    plt.plot(
        state_df["year_month"],
        state_df["actual"],
        label="Actual"
    )

    plt.plot(
        state_df["year_month"],
        state_df["predicted"],
        label="Predicted"
    )

    plt.title(f"Transformer Predictions - {state}")
    plt.xlabel("Month")
    plt.ylabel("Dengue Cases")
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"results/transformer_predictions_{state}.png"
    )

    plt.close()

print("\nState plots saved.")
# ======================================================
# Plot Predictions
# ======================================================

plt.figure(figsize=(12,6))

plt.plot(
    actuals,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.title("Transformer Forecast")

plt.xlabel("Test Samples")

plt.ylabel("Monthly Dengue Cases")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/transformer_predictions.png",
    dpi=300
)

plt.show()