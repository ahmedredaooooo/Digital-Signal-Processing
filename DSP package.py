import tkinter as tk  # Import the Tkinter module and alias it as 'tk'
from idlelib.pyparse import trans
from os import write
from tkinter import filedialog, font
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, BarycentricInterpolator
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
def Choose_Disc_Or_Cont(): #GPT
    # Create a new popup window
    input_window = tk.Toplevel(root)
    input_window.title("e5tar ya 7biby e5tar")

    # Create a label to prompt the user
    tk.Label(input_window, text="إختار يا حبيبي إختار").pack(padx=10, pady=10)

    # Variable to store the user's choice
    user_choice = tk.IntVar(value=0)  # Default to 0 if no option is selected

    # Create two radio buttons for the user to choose from
    tk.Radiobutton(input_window, text="Continuous", variable=user_choice, value=1).pack(anchor='w', padx=10)
    tk.Radiobutton(input_window, text="Discrete", variable=user_choice, value=2).pack(anchor='w', padx=10)

    def submit_input():
        if user_choice.get() == 0:
            messagebox.showerror("No Selection", "Please select an option.")
        else:
            input_window.destroy()  # Close the popup window after a valid choice is made

    # Create a button to submit the input
    tk.Button(input_window, text="Submit", command=submit_input).pack(pady=10)

    # Wait for the popup window to be closed before continuing
    input_window.wait_window()

    if user_choice.get() == 1:
        display_signals_continues("")
    else:
        display_signals_discrete("")

def Generate_Signal(): #copy From Above Function
    # Create a new popup window
    input_window = tk.Toplevel(root)
    input_window.title("et2mar ya 7pp et2mar")

    # Create a label to prompt the user
    tk.Label(input_window, text="إتأمر يا حبيبي إتأمر").pack(padx=10, pady=10)

    # Variable to store the user's choice
    wave_type = tk.IntVar(value=0)  # Default to 0 if no option is selected

    # Create two radio buttons for the user to choose from
    tk.Radiobutton(input_window, text="Sine", variable=wave_type, value=1).pack(anchor='w', padx=10)
    tk.Radiobutton(input_window, text="Cosine", variable=wave_type, value=2).pack(anchor='w', padx=10)

    A = tk.DoubleVar()
    theta = tk.DoubleVar()
    f = tk.DoubleVar()
    fs = tk.DoubleVar()

    tk.Label(input_window, text="Enter Amplitude (A):").pack(padx=10, pady=10)
    AEntry = tk.Entry(input_window)
    AEntry.pack(padx=10, pady=10)

    tk.Label(input_window, text="Enter phase shift theta (Θ):").pack(padx=10, pady=10)
    thetaEntry = tk.Entry(input_window)
    thetaEntry.pack(padx=10, pady=10)

    tk.Label(input_window, text="Enter analog frequency (f):").pack(padx=10, pady=10)
    FEntry = tk.Entry(input_window)
    FEntry.pack(padx=10, pady=10)

    tk.Label(input_window, text="Enter sampling frequency (fs):").pack(padx=10, pady=10)
    FsEntry = tk.Entry(input_window)
    FsEntry.pack(padx=10, pady=10)

    def submit_input():
        if wave_type.get() == 0:
            messagebox.showerror("No Selection", "Please select an option.")
        try:
            A.set(float(AEntry.get()))
            theta.set(float(thetaEntry.get()))
            f.set(float(FEntry.get()))
            fs.set(float(FsEntry.get()))
            if fs.get() < 2.0 * f.get():
                messagebox.showerror("Invalid Fs", "Note that the discrete signal will contain aliasing :(")
            input_window.destroy()  # Close the popup window after a valid choice is made
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    # Create a button to submit the input
    tk.Button(input_window, text="Submit", command=submit_input).pack(pady=10)

    # Wait for the popup window to be closed before continuing
    input_window.wait_window()

    drawSignal(wave_type.get(), A.get(), theta.get(), f.get(), fs.get())

def drawSignal(wave_type, A, theta, f, fs):
    dt = 1.0 / fs
    hyperDataSize = 12
    x = []
    for i in range(hyperDataSize):
        x.append(i * dt)
    plot_continuous_and_discrete_sine_wave(wave_type, A, f, theta, np.array(x))

    func = np.cos
    if wave_type == 1:
        func = np.sin
    n = []
    y = []
    for i in range(hyperDataSize):
        n.append(i)
        y.append(round(float(A * func(2 * np.pi * f / fs * i + theta)), ndigits=3))
    result = list(zip(n, y))
    with open(f"Task2 testcases and testing functions/output/GeneratedDiscWave.txt", "w") as file:
        file.write(f"0\n0\n{len(result)}\n")
        for n, y in result:
            file.write(f"{n} {y}\n")
    display_signals_discrete(f"Task2 testcases and testing functions/output/GeneratedDiscWave.txt")

def plot_continuous_and_discrete_sine_wave(wave_type, amplitude, frequency, phase_shift, x):
    func = np.cos
    if (wave_type == 1):
        func = np.sin

    # Generate x values for the continuous wave
    x_continuous = np.linspace(x[0], x[-1], 1000)  # Create 1000 points for a smooth curve
    # Calculate the y values for the continuous wave using the chosen function (sin or cos)
    y_continuous = amplitude * func(2 * np.pi * frequency * x_continuous + phase_shift)
    # Generate discrete x values matching the input list x
    x_discrete = np.linspace(x[0], x[-1], len(x))  # Use the length of x to create discrete points
    # Calculate the y values for the discrete points using the same function (sin or cos)
    y_discrete = amplitude * func(2 * np.pi * frequency * x_discrete + phase_shift)


    spline_interp = CubicSpline(x, np.array(y_discrete))
    x_pred = np.linspace(x.min(), x.max(), 1000)
    y_pred = spline_interp(x_pred)


    # Plot the continuous sine wave as a smooth curve
    plt.figure(figsize=(10, 6))  # Set the figure size for the plot
    plt.plot(x_continuous, y_continuous, label='Continuous Wave', color='blue', linewidth=2)

    plt.plot(x_pred, y_pred, label='Sampled Wave', color='orange', linewidth=2)

    # Plot the discrete sine wave as a stem plot (vertical lines with markers at the points)
    plt.stem(x_discrete, y_discrete, linefmt='green', markerfmt='ro', basefmt='k', label='Discrete Signal')

    # Set the x-axis ticks to display integer values only
    plt.xticks(np.arange(x.min(), x.max(), 1))  # Adjust the tick marks based on the step size in x

    # Customizing the plot appearance
    plt.axhline(0, color='black', linewidth=1)  # Draw a horizontal line at y = 0 (x-axis)
    plt.axvline(0, color='black', linewidth=1)  # Draw a vertical line at x = 0 (y-axis)
    plt.grid(True)  # Display a grid on the plot
    plt.title('Continuous and Discrete Sine Wave Plot', fontsize=16, fontweight='bold')  # Set the plot title
    plt.xlabel('Time', fontsize=12)  # Label for the x-axis
    plt.ylabel('Amplitude', fontsize=12)  # Label for the y-axis
    plt.legend()  # Display the legend for different wave representations

    # Show the plot
    plt.show()  # Render the plot


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
                V1=float(L[0])                      # changed to float as we need it generateSignal()
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
    display_signals_continues(f"Task1 testcases and testing functions/output/{file_name}.txt")

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

def display_signals_continues(file_name):
    if file_name == "":
        file_name = select_file(0)
    x, y = ReadSignalFile(file_name)
    x = np.array(x)
    y = np.array(y)

    # Create a polynomial interpolation function (Barycentric)
    spline_interp = CubicSpline(x, y)

    # Generate new x values for plotting
    x_new = np.linspace(x.min(), x.max(), 1000)  # Generate 300 points
    y_new = spline_interp(x_new)

    plt.figure(figsize=(8, 5))
    # Plot the continuous signal using the smooth curve
    plt.plot(x_new, y_new, color='blue', label='Continuous Signal', linewidth=2)

    # Plot the original data points
    plt.scatter(x, y, color='red', s=100, label='Data Points', alpha=0.6)

    # Set the x-axis ticks to display only integer values
    plt.xticks(np.arange(x.min(), x.max(), x[1] - x[0]))


    # Customizing the plot
    plt.title('Continuous Signal Representation', fontsize=16, fontweight='bold')
    plt.xlabel('X Axis', fontsize=12)
    plt.ylabel('Y Axis', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def display_signals_discrete(file_name):
    if file_name == "":
        file_name = select_file(0)

    x, y = ReadSignalFile(file_name)
    x = np.array(x)
    y = np.array(y)

    plt.figure(figsize=(8, 5))
    plt.stem(x, y, linefmt='blue', markerfmt='ro', basefmt='k', label='Discrete Signal')

    # Set the x-axis ticks to display only integer values
    plt.xticks(np.arange(x.min(), x.max(), x[1] - x[0]))

    # Customizing the plot
    plt.title('Discrete Signal Representation', fontsize=16, fontweight='bold')
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

button8 = tk.Button(root, text=" Display ", command=Choose_Disc_Or_Cont, **button_style)
button8.place(x=740, y=600)  # Added y-padding

button9 = tk.Button(root, text=" Generate Signal ", command=Generate_Signal, **button_style)
button9.place(x=520, y=600)  # Added y-padding

root.mainloop()