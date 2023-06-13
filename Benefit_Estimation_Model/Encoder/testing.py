import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

def testing(test_x,total_costs_test,max_length):

    # Pad the sequences
    test_x = pad_sequences(test_x, maxlen=max_length, padding='post', truncating='post', dtype='float32')
    test_x_tensors = []
    for i in range(0, len(test_x)):
        test_x_tensors.append(tf.convert_to_tensor(test_x[i]))
    test_x_tensors = tf.convert_to_tensor(test_x_tensors)


    ground_truth_labels = []
    total_costs_test = torch.stack(total_costs_test)
    for item in total_costs_test:
        ground_truth_labels.append([label.item() for label in item])
    ground_truth_labels = np.array(ground_truth_labels)

    model = tf.keras.models.load_model('Benefit_Estimation_Model/Encoder/model(16245Q)')
    ##### Step 6 - Use model to make predictions
    test_x_tensors = np.reshape(test_x_tensors, (test_x_tensors.shape[0], 1, test_x_tensors.shape[1]))
    test_x_tensors = tf.convert_to_tensor(test_x_tensors, dtype=tf.float32)
    # Predict results on test data
    pred_test = model.predict(test_x_tensors)
    predictions = []
    for i in range(0, len(pred_test)):
        predictions.append(pred_test[i][0])

    print('Prediction:', predictions)

    # Prepare the x-axis values (indices)
    indices = np.arange(len(predictions))

    # Plot the predicted outputs and the labels
    plt.figure(figsize=(10, 6))
    plt.plot(indices, predictions, color='red', label='Predicted Outputs')
    plt.plot(indices, ground_truth_labels, color='blue', label='Labels')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title('Predicted Outputs vs Labels')
    plt.legend()
    plt.show()
    ##### Step 7 - Print Performance Summary
    print("")
    print('-------------------- Model Summary --------------------')
    model.summary()  # print model summary
    print("")
    print('-------------------- Weights and Biases --------------------')
    print("Too many parameters to print but you can use the code provided if needed")
    print("")
    # for layer in model.layers:
    #    print(layer.name)
    #    for item in layer.get_weights():
    #        print("  ", item)
    # print("")

    # Print the last value in the evaluation metrics contained within history file
    print('-------------------- Evaluation on Training Data --------------------')
    # for item in history.history:
    #     print("Final", item, ":", history.history[item][-1])
    print("")

    # Evaluate the model on the test data using "evaluate"
    print('-------------------- Evaluation on Test Data --------------------')
    #test_x_numpy = test_x.numpy().astype(np.float32)
    results = model.evaluate(test_x_tensors,ground_truth_labels)
    print("Evaluation===========================================")
    # print(results)
    print("Test Loss:", results[1])
    print("Test Accuracy:", results[2])
    # print("")
    # predictions = np.array(predictions).astype(np.float32)
    #
    # metric = tf.keras.metrics.R2Score()
    # metric.update_state(ground_truth_labels, predictions)
    # result = metric.result()
    # result.numpy()

