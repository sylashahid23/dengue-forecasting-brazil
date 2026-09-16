import os
import random
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# ======================================================
# Reproducibility
# ======================================================

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ======================================================
# Configuration
# ======================================================

# Script is inside:
# diss/models/
#
# Dataset is inside:
# diss/

DATA_FILE = "../Brazil_Final_Dataset.csv"

# Results and saved models are inside:
# diss/results/
# diss/saved_models/

SAVE_DIR = "../saved_models"
RESULTS_DIR = "../results"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


FEATURES = [
    "dengue_cases",
    "temperature",
    "humidity",
    "precipitation"
]

TARGET = "dengue_cases"

SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10


# ======================================================
# Load Dataset
# ======================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

df["year_month"] = pd.to_datetime(df["year_month"])

df = df.sort_values(
    ["state", "year_month"]
).reset_index(drop=True)

print(f"Dataset loaded: {len(df)} rows")


# ======================================================
# Train / Validation / Test Split
# ======================================================

train_df = df[
    df["year_month"] < "2022-01-01"
].copy()

val_df = df[
    (df["year_month"] >= "2022-01-01")
    & (df["year_month"] < "2023-01-01")
].copy()

test_df = df[
    df["year_month"] >= "2023-01-01"
].copy()


print("\nData split:")
print(f"Training:   {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows")
print(f"Testing:    {len(test_df)} rows")


# ======================================================
# Scaling
# Fit scaler ONLY on training data for each state
# ======================================================

print("\nScaling data...")

scalers = {}

for state in train_df["state"].unique():

    scaler = MinMaxScaler()

    train_mask = train_df["state"] == state
    val_mask = val_df["state"] == state
    test_mask = test_df["state"] == state

    # Convert columns to float before assigning scaled values.
    train_df[FEATURES] = train_df[FEATURES].astype(float)
    val_df[FEATURES] = val_df[FEATURES].astype(float)
    test_df[FEATURES] = test_df[FEATURES].astype(float)

    train_df.loc[train_mask, FEATURES] = (
        scaler.fit_transform(
            train_df.loc[train_mask, FEATURES]
        )
    )

    val_df.loc[val_mask, FEATURES] = (
        scaler.transform(
            val_df.loc[val_mask, FEATURES]
        )
    )

    test_df.loc[test_mask, FEATURES] = (
        scaler.transform(
            test_df.loc[test_mask, FEATURES]
        )
    )

    scalers[state] = scaler


# ======================================================
# Sequence Generation
# ======================================================

def create_sequences(
    train_history_df,
    eval_df,
    sequence_length
):

    X = []
    y = []
    metadata = []

    for state in train_history_df["state"].unique():

        history = train_history_df[
            train_history_df["state"] == state
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
                values[
                    i:i + sequence_length
                ]
            )

            y.append(
                target[
                    i + sequence_length
                ]
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


def create_training_sequences(
    dataframe,
    sequence_length
):

    X = []
    y = []

    for state in dataframe["state"].unique():

        state_df = dataframe[
            dataframe["state"] == state
        ]

        values = state_df[FEATURES].values
        target = state_df[TARGET].values

        for i in range(
            len(state_df) - sequence_length
        ):

            X.append(
                values[
                    i:i + sequence_length
                ]
            )

            y.append(
                target[
                    i + sequence_length
                ]
            )

    return (
        np.array(X),
        np.array(y)
    )


print("\nCreating sequences...")

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


print(f"Training sequences:   {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")
print(f"Testing sequences:    {len(X_test)}")


# ======================================================
# Convert to PyTorch tensors
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


# ======================================================
# Model Configuration
# ======================================================

INPUT_SIZE = len(FEATURES)

HIDDEN_SIZE = 64
NUM_LAYERS = 2

OUTPUT_SIZE = 1

DROPOUT = 0.2


# ======================================================
# LSTM Model
# ======================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            OUTPUT_SIZE
        )


    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return self.fc(
            hidden[-1]
        )


# ======================================================
# Transformer Model
# ======================================================

class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.input_projection = nn.Linear(
            INPUT_SIZE,
            64
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.2,
                batch_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        )

        self.fc = nn.Linear(
            64,
            OUTPUT_SIZE
        )


    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        x = x[:, -1, :]

        return self.fc(x)


# ======================================================
# Moving Average
# ======================================================

class MovingAverage(nn.Module):

    def __init__(
        self,
        kernel_size
    ):

        super().__init__()

        self.kernel_size = kernel_size

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )


    def forward(self, x):

        front = x[:, 0:1, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1
        )

        end = x[:, -1:, :].repeat(
            1,
            (self.kernel_size - 1) // 2,
            1
        )

        x = torch.cat(
            [front, x, end],
            dim=1
        )

        x = self.avg(
            x.permute(0, 2, 1)
        )

        return x.permute(0, 2, 1)


# ======================================================
# Series Decomposition
# ======================================================

class SeriesDecomposition(nn.Module):

    def __init__(
        self,
        kernel_size=25
    ):

        super().__init__()

        self.moving_avg = (
            MovingAverage(kernel_size)
        )


    def forward(self, x):

        trend = self.moving_avg(x)

        seasonal = x - trend

        return seasonal, trend


# ======================================================
# DLinear Model
# ======================================================

class DLinearModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.decomposition = (
            SeriesDecomposition(
                kernel_size=25
            )
        )

        self.linear_seasonal = nn.Linear(
            SEQUENCE_LENGTH,
            1
        )

        self.linear_trend = nn.Linear(
            SEQUENCE_LENGTH,
            1
        )


    def forward(self, x):

        seasonal, trend = (
            self.decomposition(x)
        )

        seasonal = seasonal.permute(
            0, 2, 1
        )

        trend = trend.permute(
            0, 2, 1
        )

        seasonal = self.linear_seasonal(
            seasonal
        )

        trend = self.linear_trend(
            trend
        )

        out = seasonal + trend

        return out[:, 0, :]


# ======================================================
# Training Function
# ======================================================

def train_model(
    model,
    model_name
):

    set_seed(SEED)

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

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    patience_counter = 0

    save_path = os.path.join(
        SAVE_DIR,
        f"best_{model_name}_ensemble_model.pth"
    )


    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0


        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()


        train_loss /= len(train_loader)


        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        model.eval()

        val_loss = 0.0


        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_loss += loss.item()


        val_loss /= len(val_loader)


        print(
            f"{model_name} | "
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Validation: {val_loss:.6f}"
        )


        # ----------------------------------------------
        # Save best model
        # ----------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            patience_counter = 0

            torch.save(
                model.state_dict(),
                save_path
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"{model_name}: early stopping."
                )

                break


    # ----------------------------------------------
    # Load best model
    # ----------------------------------------------

    model.load_state_dict(
        torch.load(
            save_path,
            map_location="cpu"
        )
    )

    return model


# ======================================================
# Prediction Function
# ======================================================

def predict_scaled(model):

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model.eval()

    predictions = []


    with torch.no_grad():

        for X_batch, _ in test_loader:

            outputs = model(
                X_batch
            )

            predictions.extend(
                outputs.cpu()
                .numpy()
                .flatten()
            )


    return np.array(
        predictions
    )


# ======================================================
# Train Base Models
# ======================================================

print("\n")
print("======================================")
print("Training LSTM")
print("======================================")


lstm_model = train_model(
    LSTMModel(),
    "lstm"
)


print("\n")
print("======================================")
print("Training Transformer")
print("======================================")


transformer_model = train_model(
    TransformerModel(),
    "transformer"
)


print("\n")
print("======================================")
print("Training DLinear")
print("======================================")


dlinear_model = train_model(
    DLinearModel(),
    "dlinear"
)


# ======================================================
# Predictions from Base Models
# ======================================================

print("\nGenerating predictions...")


lstm_scaled = predict_scaled(
    lstm_model
)

transformer_scaled = predict_scaled(
    transformer_model
)

dlinear_scaled = predict_scaled(
    dlinear_model
)


# ======================================================
# Inverse Scaling
# ======================================================

def inverse_scale_predictions(
    predictions
):

    output = np.zeros(
        len(predictions)
    )


    for state in test_meta[
        "state"
    ].unique():

        scaler = scalers[state]

        mask = (
            test_meta["state"] == state
        )

        dummy = np.zeros(
            (
                mask.sum(),
                len(FEATURES)
            )
        )

        dummy[:, 0] = (
            predictions[mask]
        )

        output[mask] = (
            scaler.inverse_transform(
                dummy
            )[:, 0]
        )


    # Dengue cases cannot be negative.

    return np.maximum(
        output,
        0
    )


lstm_original = (
    inverse_scale_predictions(
        lstm_scaled
    )
)

transformer_original = (
    inverse_scale_predictions(
        transformer_scaled
    )
)

dlinear_original = (
    inverse_scale_predictions(
        dlinear_scaled
    )
)

actual_original = (
    inverse_scale_predictions(
        y_test.numpy().flatten()
    )
)


# ======================================================
# Pairwise Ensembles
# Equal-weight average
# ======================================================

ensembles = {

    "LSTM_Transformer":
        (
            lstm_original
            + transformer_original
        ) / 2,

    "LSTM_DLinear":
        (
            lstm_original
            + dlinear_original
        ) / 2,

    "Transformer_DLinear":
        (
            transformer_original
            + dlinear_original
        ) / 2
}


# ======================================================
# Evaluate Pairwise Ensembles
# ======================================================

all_results = []


for name, predictions in ensembles.items():

    rmse = np.sqrt(
        mean_squared_error(
            actual_original,
            predictions
        )
    )

    mae = mean_absolute_error(
        actual_original,
        predictions
    )


    print("\n")
    print("==============================")
    print(name)
    print("==============================")

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"MAE : {mae:.2f}"
    )


    all_results.append({

        "Model": name,

        "RMSE": rmse,

        "MAE": mae

    })


# ======================================================
# Metrics by State
# ======================================================

for name, predictions in ensembles.items():

    rows = []


    for state in test_meta[
        "state"
    ].unique():

        mask = (
            test_meta["state"] == state
        )

        state_actual = (
            actual_original[mask]
        )

        state_predictions = (
            predictions[mask]
        )


        rmse = np.sqrt(
            mean_squared_error(
                state_actual,
                state_predictions
            )
        )

        mae = mean_absolute_error(
            state_actual,
            state_predictions
        )


        rows.append({

            "state": state,

            "RMSE": rmse,

            "MAE": mae

        })


    state_df = pd.DataFrame(
        rows
    )


    print("\n")
    print(
        f"{name} - Metrics by State"
    )

    print(
        state_df
    )


    state_file = os.path.join(
        RESULTS_DIR,
        f"{name.lower()}_state_metrics.csv"
    )

    state_df.to_csv(
        state_file,
        index=False
    )


# ======================================================
# Save Ensemble Results
# ======================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("==============================")
print("Pairwise Ensemble Results")
print("==============================")


print(
    results_df
)


results_file = os.path.join(
    RESULTS_DIR,
    "pairwise_ensemble_metrics.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ======================================================
# Save Predictions
# ======================================================

prediction_df = test_meta.copy()

prediction_df["actual"] = (
    actual_original
)

prediction_df["LSTM_Transformer"] = (
    ensembles["LSTM_Transformer"]
)

prediction_df["LSTM_DLinear"] = (
    ensembles["LSTM_DLinear"]
)

prediction_df["Transformer_DLinear"] = (
    ensembles["Transformer_DLinear"]
)


prediction_file = os.path.join(
    RESULTS_DIR,
    "pairwise_ensemble_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


# ======================================================
# Finished
# ======================================================

print("\n")
print("======================================")
print("DONE")
print("======================================")

print(
    f"Metrics saved to: {results_file}"
)

print(
    f"Predictions saved to: {prediction_file}"
)

print(
    f"Models saved to: {SAVE_DIR}"
)