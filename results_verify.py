import numpy as np
import csv
import matplotlib.pyplot as plt
test_result = np.load(r"C:\Users\xms01\Desktop\Effectiveness_DL_TSF\Baseline_model\LTSF-Linear-main\results\sin_linear_noise1_192_1000_DLinear_custom_ftM_sl192_ll48_pl1000_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0\pred.npy")
def read_column_as_array(filepath, column_index):
    """
    Read a specific column from a CSV file into a NumPy array using the csv module.

    Parameters:
    - filename (str): The path to the CSV file.
    - column_index (int): The index of the column to read.

    Returns:
    - np.array: The data from the specified column as a NumPy array.
    """
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        # Skip the header if necessary
        next(reader)
        # Extract the specified column
        column_data = [float(row[column_index]) for row in reader]
    return column_data


data = read_column_as_array(r"C:\Users\xms01\Desktop\Effectiveness_DL_TSF\Baseline_model\LTSF-Linear-main\dataset\sin_linear_noise_1.csv", 1)
plt.plot(data[-1000:])
plt.plot(test_result[-1])
plt.grid(True)  # Optional: add a grid for better readability
plt.show()
