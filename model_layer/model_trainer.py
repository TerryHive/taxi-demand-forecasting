import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

def load_data():
    """Load and prepare the training data."""
    X = np.load("data/model_input/X.npy")
    y = np.load("data/model_input/y.npy")
    grid_codes = np.load("data/model_input/grid_codes.npy")
    
    # Reshape for LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, grid_codes

def build_model(input_shape, num_grids, embedding_dim=8):
    """
    Builds an LSTM model with spatial embedding input.
    
    Args:
        input_shape: Shape of time series input (window_size, 1)
        num_grids: Number of unique grid_ids
        embedding_dim: Size of embedding vector

    Returns:
        Keras Model
    """
    # Input 1: Time series
    time_input = Input(shape=input_shape, name="time_series_input")
    lstm_out = LSTM(64)(time_input)

    # Input 2: Grid ID (as integer)
    grid_input = Input(shape=(1,), name="grid_id_input")
    embedding = Embedding(input_dim=num_grids, output_dim=embedding_dim)(grid_input)
    embedding_flat = Dense(embedding_dim, activation='relu')(embedding[:, 0, :])  # Flatten 1D

    # Combine both
    combined = Concatenate()([lstm_out, embedding_flat])
    output = Dense(1)(combined)

    model = Model(inputs=[time_input, grid_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

    return model

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load and split data
    print("Loading data...")
    X, y, grid_codes = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test, grid_train, grid_test = train_test_split(
        X, y, grid_codes, test_size=0.2, random_state=42
    )

    print("\nData shapes:")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Number of unique grids: {len(np.unique(grid_codes))}")

    # Build the model
    print("\nBuilding model...")
    model = build_model(
        input_shape=(X.shape[1], 1),
        num_grids=int(grid_codes.max() + 1)
    )
    model.summary()

    # Train the model
    print("\nTraining model...")
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        [X_train, grid_train], y_train,
        validation_data=([X_test, grid_test], y_test),
        epochs=20,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    # Plot training history
    plot_training_history(history)
    print("\nTraining history plot saved to: models/training_history.png")

    # Evaluate
    print("\nEvaluating model...")
    loss, mae = model.evaluate([X_test, grid_test], y_test, verbose=0)
    print(f"Final Test MAE: {mae:.2f}")

    # Predict on test set
    y_pred = model.predict([X_test, grid_test])

    # Save predictions
    np.save("data/processed/y_true.npy", y_test)
    np.save("data/processed/y_pred.npy", y_pred)
    print("Saved predictions to data/processed/y_true.npy and y_pred.npy")

    # Save model
    model_path = models_dir / "lstm_pickup_forecaster.h5"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main() 