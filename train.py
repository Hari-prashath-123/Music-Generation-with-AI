import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    
    # Add LSTM layers
    model.add(LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    
    return model

def train(model, network_input, network_output, epochs=10, batch_size=64):
    """ train the neural network """
    
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.keras"
    
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    
    model.fit(
        network_input, 
        network_output, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=callbacks_list
    )
    
    # Save the final model
    model.save('music_gen_model.keras')
    print("Training complete. Model saved to music_gen_model.keras")

if __name__ == '__main__':
    print("Loading preprocessed data...")
    try:
        with open('data.pkl', 'rb') as filepath:
            data = pickle.load(filepath)
            
        network_input = data['network_input']
        network_output = data['network_output']
        n_vocab = data['n_vocab']
        
        print(f"Data loaded successfully. Vocabulary size: {n_vocab}")
        
        print("Building model...")
        model = create_network(network_input, n_vocab)
        model.summary()
        
        print("Starting training (this may take a while)...")
        # Training for just 10 epochs for demonstration; increase for better results
        train(model, network_input, network_output, epochs=10)
        
    except FileNotFoundError:
        print("Error: data.pkl not found. Please run preprocess.py first.")
