import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "../Brazil_Final_Dataset.csv"
OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 12
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
SEED = 42

FEATURES = [
    "dengue_cases",
    "temperature",
    "humidity",
    "precipitation"
]

TARGET = "dengue_cases"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Helper functions
# ============================================================

def create_sequences(data, sequence_length):
    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def train_model(model, train_loader, val_loader):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X_batch).squeeze(-1)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                predictions = model(X_batch).squeeze(-1)

                loss = criterion(
                    predictions,
                    y_batch
                )

                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Validation: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    return model


def predict_model(model, X):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        predictions = model(
            X_tensor
        ).squeeze(-1)

    return predictions.cpu().numpy()


def inverse_predictions(predictions, scaler):

    dummy = np.zeros(
        (len(predictions), len(FEATURES))
    )

    dummy[:, 0] = predictions

    return scaler.inverse_transform(
        dummy
    )[:, 0]


# ============================================================
# Transformer Model
# ============================================================

class TransformerModel(nn.Module):

    def __init__(
        self,
        input_size=4,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            d_model,
            1
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        last_output = x[:, -1, :]

        return self.fc(last_output)


# ============================================================
# DLinear Model
# ============================================================

class MovingAverage(nn.Module):

    def __init__(self, kernel_size):

        super().__init__()

        self.kernel_size = kernel_size

        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def forward(self, x):

        pad = (
            self.kernel_size - 1
        ) // 2

        front = x[:, 0:1, :].repeat(
            1,
            pad,
            1
        )

        end = x[:, -1:, :].repeat(
            1,
            pad,
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


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):

        super().__init__()

        self.moving_avg = MovingAverage(
            kernel_size
        )

    def forward(self, x):

        moving_mean = self.moving_avg(x)

        residual = x - moving_mean

        return residual, moving_mean


class DLinearModel(nn.Module):

    def __init__(
        self,
        seq_len=12,
        pred_len=1,
        enc_in=4,
        kernel_size=25
    ):

        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.decomposition = SeriesDecomp(
            kernel_size
        )

        self.Linear_Seasonal = nn.Linear(
            seq_len,
            pred_len
        )

        self.Linear_Trend = nn.Linear(
            seq_len,
            pred_len
        )

    def forward(self, x):

        seasonal_init, trend_init = (
            self.decomposition(x)
        )

        seasonal_init = seasonal_init.permute(
            0, 2, 1
        )

        trend_init = trend_init.permute(
            0, 2, 1
        )

        seasonal_output = (
            self.Linear_Seasonal(
                seasonal_init
            )
        )

        trend_output = (
            self.Linear_Trend(
                trend_init
            )
        )

        output = (
            seasonal_output +
            trend_output
        )

        output = output.permute(
            0, 2, 1
        )

        return output[:, -1, 0].unsqueeze(-1)


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df["year_month"] = pd.to_datetime(
    df["year_month"]
)

df = df.sort_values(
    ["state", "year_month"]
).reset_index(drop=True)

print(df.shape)


# ============================================================
# Train / Validation / Test split
# ============================================================

train_df = df[
    df["year_month"] < "2022-01-01"
].copy()

val_df = df[
    (df["year_month"] >= "2022-01-01") &
    (df["year_month"] < "2023-01-01")
].copy()

test_df = df[
    df["year_month"] >= "2023-01-01"
].copy()

print()
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ============================================================
# Scaling
# ============================================================

scalers = {}

train_df[FEATURES] = train_df[
    FEATURES
].astype(float)

val_df[FEATURES] = val_df[
    FEATURES
].astype(float)

test_df[FEATURES] = test_df[
    FEATURES
].astype(float)


for state in df["state"].unique():

    scaler = MinMaxScaler()

    train_mask = (
        train_df["state"] == state
    )

    val_mask = (
        val_df["state"] == state
    )

    test_mask = (
        test_df["state"] == state
    )

    train_df.loc[
        train_mask,
        FEATURES
    ] = scaler.fit_transform(
        train_df.loc[
            train_mask,
            FEATURES
        ]
    )

    val_df.loc[
        val_mask,
        FEATURES
    ] = scaler.transform(
        val_df.loc[
            val_mask,
            FEATURES
        ]
    )

    test_df.loc[
        test_mask,
        FEATURES
    ] = scaler.transform(
        test_df.loc[
            test_mask,
            FEATURES
        ]
    )

    scalers[state] = scaler


print()
print("Per-state scaling complete.")


# ============================================================
# Sequence generation
# ============================================================

X_train_all = []
y_train_all = []

X_val_all = []
y_val_all = []

X_test_all = []

val_states = []
val_dates = []

test_states = []
test_dates = []


for state in df["state"].unique():

    train_state = train_df[
        train_df["state"] == state
    ].sort_values("year_month")

    val_state = val_df[
        val_df["state"] == state
    ].sort_values("year_month")

    test_state = test_df[
        test_df["state"] == state
    ].sort_values("year_month")


    # Training sequences
    train_values = train_state[
        FEATURES
    ].values

    X_train, y_train = create_sequences(
        train_values,
        SEQUENCE_LENGTH
    )

    X_train_all.append(X_train)
    y_train_all.append(y_train)


    # Validation sequences
    val_input = pd.concat(
        [
            train_state.tail(
                SEQUENCE_LENGTH
            ),
            val_state
        ]
    )

    val_values = val_input[
        FEATURES
    ].values

    X_val, y_val = create_sequences(
        val_values,
        SEQUENCE_LENGTH
    )

    X_val_all.append(X_val)
    y_val_all.append(y_val)

    val_states.extend(
        [state] * len(y_val)
    )

    val_dates.extend(
        val_state["year_month"].tolist()
    )


    # Test sequences
    test_input = pd.concat(
        [
            val_state.tail(
                SEQUENCE_LENGTH
            ),
            test_state
        ]
    )

    test_values = test_input[
        FEATURES
    ].values

    X_test, _ = create_sequences(
        test_values,
        SEQUENCE_LENGTH
    )

    X_test_all.append(X_test)

    test_states.extend(
        [state] * len(X_test)
    )

    test_dates.extend(
        test_state["year_month"].tolist()
    )


X_train = np.concatenate(
    X_train_all
)

y_train = np.concatenate(
    y_train_all
)

X_val = np.concatenate(
    X_val_all
)

y_val = np.concatenate(
    y_val_all
)

X_test = np.concatenate(
    X_test_all
)


print()
print("Sequence generation complete.")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# DataLoaders
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(
        X_train,
        dtype=torch.float32
    ),
    torch.tensor(
        y_train,
        dtype=torch.float32
    )
)

val_dataset = TensorDataset(
    torch.tensor(
        X_val,
        dtype=torch.float32
    ),
    torch.tensor(
        y_val,
        dtype=torch.float32
    )
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


# ============================================================
# Train Transformer
# ============================================================

print()
print("=" * 60)
print("Training Transformer")
print("=" * 60)

transformer_model = TransformerModel().to(DEVICE)

transformer_model = train_model(
    transformer_model,
    train_loader,
    val_loader
)

print("Transformer best model loaded.")


# ============================================================
# Train DLinear
# ============================================================

print()
print("=" * 60)
print("Training DLinear")
print("=" * 60)

dlinear_model = DLinearModel(
    seq_len=SEQUENCE_LENGTH,
    pred_len=1,
    enc_in=len(FEATURES)
).to(DEVICE)

dlinear_model = train_model(
    dlinear_model,
    train_loader,
    val_loader
)

print("DLinear best model loaded.")


# ============================================================
# Generate predictions
# ============================================================

print()
print("Generating validation predictions...")

transformer_val_scaled = predict_model(
    transformer_model,
    X_val
)

dlinear_val_scaled = predict_model(
    dlinear_model,
    X_val
)

print("Generating test predictions...")

transformer_test_scaled = predict_model(
    transformer_model,
    X_test
)

dlinear_test_scaled = predict_model(
    dlinear_model,
    X_test
)


# ============================================================
# Inverse scaling
# ============================================================

transformer_val = np.zeros(
    len(transformer_val_scaled)
)

dlinear_val = np.zeros(
    len(dlinear_val_scaled)
)

transformer_test = np.zeros(
    len(transformer_test_scaled)
)

dlinear_test = np.zeros(
    len(dlinear_test_scaled)
)


for i, state in enumerate(val_states):

    transformer_val[i] = inverse_predictions(
        [transformer_val_scaled[i]],
        scalers[state]
    )[0]

    dlinear_val[i] = inverse_predictions(
        [dlinear_val_scaled[i]],
        scalers[state]
    )[0]


for i, state in enumerate(test_states):

    transformer_test[i] = inverse_predictions(
        [transformer_test_scaled[i]],
        scalers[state]
    )[0]

    dlinear_test[i] = inverse_predictions(
        [dlinear_test_scaled[i]],
        scalers[state]
    )[0]


transformer_val = np.maximum(
    transformer_val,
    0
)

dlinear_val = np.maximum(
    dlinear_val,
    0
)

transformer_test = np.maximum(
    transformer_test,
    0
)

dlinear_test = np.maximum(
    dlinear_test,
    0
)


# ============================================================
# Actual validation and test values
# ============================================================

val_actual = []
test_actual = []


for state in df["state"].unique():

    val_state_original = df[
        (df["state"] == state) &
        (df["year_month"] >= "2022-01-01") &
        (df["year_month"] < "2023-01-01")
    ].sort_values("year_month")

    test_state_original = df[
        (df["state"] == state) &
        (df["year_month"] >= "2023-01-01")
    ].sort_values("year_month")

    val_actual.extend(
        val_state_original[
            TARGET
        ].astype(float).tolist()
    )

    test_actual.extend(
        test_state_original[
            TARGET
        ].astype(float).tolist()
    )


val_actual = np.array(
    val_actual
)

test_actual = np.array(
    test_actual
)


# ============================================================
# Train stacking meta-model
# ============================================================

print()
print("=" * 60)
print("Training Stacking Meta-Model")
print("=" * 60)

meta_X = np.column_stack(
    [
        transformer_val,
        dlinear_val
    ]
)

meta_y = val_actual

meta_model = LinearRegression()

meta_model.fit(
    meta_X,
    meta_y
)


print()
print("Meta-model coefficients:")

print(
    "Transformer coefficient:",
    meta_model.coef_[0]
)

print(
    "DLinear coefficient:",
    meta_model.coef_[1]
)

print(
    "Intercept:",
    meta_model.intercept_
)


# ============================================================
# Apply stacking model to test data
# ============================================================

test_meta_X = np.column_stack(
    [
        transformer_test,
        dlinear_test
    ]
)

stacking_predictions = meta_model.predict(
    test_meta_X
)

stacking_predictions = np.maximum(
    stacking_predictions,
    0
)


# ============================================================
# Overall metrics
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        test_actual,
        stacking_predictions
    )
)

mae = mean_absolute_error(
    test_actual,
    stacking_predictions
)


print()
print("=" * 60)
print("Transformer + DLinear Stacking Results")
print("=" * 60)

print(
    f"RMSE: {rmse:.2f}"
)

print(
    f"MAE : {mae:.2f}"
)


# ============================================================
# State-level metrics
# ============================================================

state_results = []

start = 0

for state in df["state"].unique():

    n = len(
        test_df[
            test_df["state"] == state
        ]
    )

    end = start + n

    actual_state = test_actual[
        start:end
    ]

    predicted_state = stacking_predictions[
        start:end
    ]

    state_rmse = np.sqrt(
        mean_squared_error(
            actual_state,
            predicted_state
        )
    )

    state_mae = mean_absolute_error(
        actual_state,
        predicted_state
    )

    state_results.append(
        {
            "state": state,
            "RMSE": state_rmse,
            "MAE": state_mae
        }
    )

    start = end


state_results_df = pd.DataFrame(
    state_results
)


print()
print("State-level results:")

print(
    state_results_df.to_string(
        index=False
    )
)


# ============================================================
# Save predictions
# ============================================================

output = pd.DataFrame(
    {
        "state": test_states,
        "year_month": test_dates,
        "actual": test_actual,
        "transformer_predicted": transformer_test,
        "dlinear_predicted": dlinear_test,
        "stacking_predicted": stacking_predictions
    }
)

output.to_csv(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_predictions.csv",
    index=False
)


# ============================================================
# Save overall metrics
# ============================================================

metrics_output = pd.DataFrame(
    {
        "Model": [
            "Transformer + DLinear Stacking"
        ],
        "RMSE": [
            rmse
        ],
        "MAE": [
            mae
        ]
    }
)

metrics_output.to_csv(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_metrics.csv",
    index=False
)


# ============================================================
# Save state-level metrics
# ============================================================

state_results_df.to_csv(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_state_metrics.csv",
    index=False
)


# ============================================================
# Finish
# ============================================================

print()
print("=" * 60)
print("Results saved successfully.")
print("=" * 60)

print(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_predictions.csv"
)

print(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_metrics.csv"
)

print(
    OUTPUT_DIR /
    "transformer_dlinear_stacking_state_metrics.csv"
)