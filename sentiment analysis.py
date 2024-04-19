import pandas as pd
import tensorflow as tf
import numpy as np

# Load data from CSV
data = pd.read_csv('sentiment.csv')

# Text data and labels
texts = data['text'].tolist()
labels = data['sentiment'].values

# Manual tokenization
word_index = {}
sequences = []
for text in texts:
    words = text.lower().split()
    sequence = []
    for word in words:
        if word not in word_index:
            word_index[word] = len(word_index) + 1
        sequence.append(word_index[word])
    sequences.append(sequence)

# Padding sequences
max_length = max(len(sequence) for sequence in sequences)
padded_sequences = []
for sequence in sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_sequences.append(padded_sequence)

# Convert to numpy array
padded_sequences = np.array(padded_sequences)

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 16, input_length=max_length),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=15, verbose=1)


# Test the model
test_texts = ["The price was too high for the quality", "The interface is user-friendly", "I'm satisfied"]
test_sequences = []
for text in test_texts:
    words = text.lower().split()
    sequence = []
    for word in words:
        if word in word_index:
            sequence.append(word_index[word])
    test_sequences.append(sequence)

# Padding test sequences
padded_test_sequences = []
for sequence in test_sequences:
    padded_sequence = sequence[:max_length] + [0] * (max_length - len(sequence))
    padded_test_sequences.append(padded_sequence)

# Convert to numpy array
padded_test_sequences = np.array(padded_test_sequences)

# Make predictions
predictions = model.predict(padded_test_sequences)

# Print predicted sentiments
for i, text in enumerate(test_texts):
    print(f"Text: {text}, Predicted Sentiment: {np.argmax(predictions[i])}")


# Evaluate the model
evaluation = model.evaluate(padded_sequences, labels, verbose=0)

# Extract loss and accuracy
loss = evaluation[0]
accuracy = evaluation[1]

# Print loss and accuracy
print("Loss:", loss)
print("Accuracy:", accuracy)