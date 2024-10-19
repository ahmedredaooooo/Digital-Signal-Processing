import tkinter as tk  # Import the Tkinter module and alias it as 'tk'
from idlelib.pyparse import trans
from os import write
from tkinter import filedialog, font
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline



# Variables to store the file paths
first_signal_path = ""
second_signal_path = ""

def prompt():
    # Create a new popup window
    input_window = tk.Toplevel(root)
    input_window.title("Enter a Number")

    # Create a label and a text entry box
    tk.Label(input_window, text="Enter a number:").pack(padx=10, pady=10)
    entry = tk.Entry(input_window)
    entry.pack(padx=10, pady=10)

    # Variable to store the input value
    user_input = tk.DoubleVar()

    def submit_input():
        # Get the number entered and store it in user_input
        try:
            user_input.set(int(entry.get()))
            input_window.destroy()  # Close the popup window after input is taken
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    # Create a button to submit the input
    tk.Button(input_window, text="Submit", command=submit_input).pack(pady=10)

    # Wait for the popup window to be closed before continuing
    input_window.wait_window()

    return user_input.get()


def ReadSignalFile(file_name):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices,expected_samples


def select_file(button_id):
    global first_signal_path, second_signal_path

    # Open file dialog
    file_path = filedialog.askopenfilename(
        initialdir=".",
        title="Select a Text File",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
    )

    # Check which button was pressed and store the file path accordingly
    if file_path:
        if button_id == 1:
            first_signal_path = file_path
            messagebox.showinfo("File 1 Selected", f"First file selected: {first_signal_path}")
        elif button_id == 2:
            second_signal_path = file_path
            messagebox.showinfo("File 2 Selected", f"Second file selected: {second_signal_path}")
        else:
            return file_path
    else:
        messagebox.showwarning("No File Selected", "Please select a file.")

def standardize_signals():
    if first_signal_path == "":
        select_file(1)
    if second_signal_path == "":
        select_file(2)
    x1, y1 = ReadSignalFile(first_signal_path)
    x2, y2 = ReadSignalFile(second_signal_path)

    i, j = 0, 0
    x1.append(int(1e9))
    x2.append(int(1e9))
    while i < len(x1) and j < len(x2):
        if x1[i] < x2[j]:
            x2.insert(j, x1[i])
            y2.insert(j, 0.0)
        elif x1[i] > x2[j]:
            x1.insert(i, x2[j])
            y1.insert(i, 0.0)
        else:
            i += 1
            j += 1
    x1.pop()
    x2.pop()
    return x1, y1, y2

def save_display(file_name, result):
    # Open the file in write mode and print the result to it
    with open(f"Task1 testcases and testing functions/output/{file_name}.txt", "w") as file:
        file.write(f"0\n0\n{len(result)}\n")
        for x, y in result:
            file.write(f"{int(x)} {int(y)}\n")
    display_signals(f"Task1 testcases and testing functions/output/{file_name}.txt")

def add_signals():
    x1, y1, y2 = standardize_signals()
    res = [y1[i] + y2[i] for i in range(len(y1))]
    save_display("add", list(zip(x1, res)))

def scale_signal(A):
    if first_signal_path == "":
        select_file(1)
    x, y = ReadSignalFile(first_signal_path)
    res = [y[i] * A for i in range(len(y))]
    save_display("mul", list(zip(x, res)))

def subtract_signals():
    x1, y1, y2 = standardize_signals()
    res = [y1[i] - y2[i] for i in range(len(y1))]
    save_display("subtract", list(zip(x1, res)))

def shift_signal(k):
    if first_signal_path == "":
        select_file(1)
    x, y = ReadSignalFile(first_signal_path)
    for i in range(len(x)):
        x[i] -= k
    if k < 0:
        save_display("delay", list(zip(x, y)))
    else:
        save_display("advance", list(zip(x, y)))

def folding_signal():
    if first_signal_path == "":
        select_file(1)
    x, y = ReadSignalFile(first_signal_path)
    x.reverse()
    y.reverse()
    for i in range(len(x)):
        x[i] = -x[i]
    save_display("folding", list(zip(x, y)))

def display_signals(file_name):
    if file_name == "":
        file_name = select_file(0)
    x, y = ReadSignalFile(file_name)
    x = np.array(x)
    y = np.array(y)
    # Create spline interpolation
    spline = UnivariateSpline(x, y, s=0.5)  # Adjust s for smoothness

    # Generate a smooth range of x values
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = spline(x_smooth)

    # Plot the smoothed curve
    plt.figure(figsize=(8, 5))
    plt.plot(x_smooth, y_smooth, color='blue', lw=3, label='Smooth Curve')

    # Optionally plot original data points
    plt.scatter(x, y, color='red', s=100, label='Data Points')

    # Customizing the plot
    plt.title('Curved Signal Representation', fontsize=16, fontweight='bold')
    plt.xlabel('X Axis', fontsize=12)
    plt.ylabel('Y Axis', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


root = tk.Tk()
root.title("Signal Processing")
root.geometry("1700x1350")  # Set the window size

# Create a custom font
custom_font = font.Font(family="Helvetica", size=12, weight="bold")

# Button styles
button_style = {
    'bg': '#4CAF50',  # Green background
    'fg': 'white',    # White text
    'font': custom_font,
    'padx': 10,
    'pady': 10,
    'borderwidth': 2,
    'relief': 'raised'
}

# Create buttons with custom styles
button1 = tk.Button(root, text="Choose signal 1", command=lambda: select_file(1), **button_style)
button1.place(x=240, y=100)  # Position on the left

button2 = tk.Button(root, text="Choose signal 2", command=lambda: select_file(2), **button_style)
button2.place(x=1240, y=100)  # Position on the right

# Add and Subtract buttons in the middle
button3 = tk.Button(root, text="Add", command=add_signals, **button_style)
button3.place(x=640, y=250)  # Centered between the two buttons

button5 = tk.Button(root, text="Subtract ", command=subtract_signals, **button_style)
button5.place(x=940, y=250)  # Below Add

# Multiply and Shift buttons below Add with y-padding
button4 = tk.Button(root, text="Multiply ", command=lambda: scale_signal(prompt()), **button_style)
button4.place(x=240, y=350)  # Below "Choose signal 1"

button6 = tk.Button(root, text="  Shift  ", command=lambda: shift_signal(prompt()), **button_style)
button6.place(x=240, y=420)  # Added y-padding

# Fold and Display buttons below Multiply with y-padding
button7 = tk.Button(root, text="  Fold   ", command=folding_signal, **button_style)
button7.place(x=240, y=490)  # Added y-padding

button8 = tk.Button(root, text=" Display ", command=lambda: display_signals(""), **button_style)
button8.place(x=740, y=600)  # Added y-padding

root.mainloop()
