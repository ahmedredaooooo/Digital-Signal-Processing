import bisect
import cmath
import math
import tkinter as tk  # Import the Tkinter module and alias it as 'tk'
from tkinter import filedialog, font
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from scipy.interpolate import CubicSpline, BarycentricInterpolator


# Variables to store the file paths
first_signal_path = ""
second_signal_path = ""

def prompt(txt = "Enter a number:"):
    # Create a new popup window
    input_window = tk.Toplevel(root)
    input_window.title("Enter a Number")

    # Create a label and a text entry box
    tk.Label(input_window, text=txt).pack(padx=10, pady=10)
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

def display_signals_discrete(file_name, xAxis = 'X Axis', yAxis = 'Y Axis'):
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
    plt.xlabel(xAxis, fontsize=12)
    plt.ylabel(yAxis, fontsize=12)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def to_binary(n, numOfBits):
    binary = ""
    while n > 0:
        binary += str(n & 1)
        n >>= 1
    while len(binary) < numOfBits:
        binary += '0'
    binary = binary[::-1]
    return binary

def quantize_signal(file_name):
    if file_name == "":
        file_name = select_file(0)

    # Create a new popup window
    input_window = tk.Toplevel(root)
    input_window.title("et2mar ya 7pp et2mar")

    # Create a label to prompt the user
    tk.Label(input_window, text="bits OR LEVELS ?").pack(padx=10, pady=10)
    # Variable to store the user's choice
    lvlsORbit = tk.IntVar(value=0)  # Default to 0 if no option is selected

    # Create two radio buttons for the user to choose from
    tk.Radiobutton(input_window, text="levels (إختار ديه عشان متتعبناش)", variable=lvlsORbit, value=1).pack(anchor='w', padx=10)
    tk.Radiobutton(input_window, text="bits (كده حضطر احسب اللفلز)", variable=lvlsORbit, value=2).pack(anchor='w', padx=10)

    _lvls = tk.IntVar()
    lvls = 0
    txt = "Enter a Number:"
    tk.Label(input_window, text= txt).pack(padx=10, pady=10)
    entry = tk.Entry(input_window)
    entry.pack(padx=10, pady=10)

    def submit_input():
        if lvlsORbit.get() == 0:
            messagebox.showerror("No Selection", "Please select an option.")
        try:
            if lvlsORbit.get() == 1:
                _lvls.set(int(entry.get()))
            elif lvlsORbit.get() == 2:
                _lvls.set(1 << int(entry.get()))
            if lvlsORbit.get() != 0:
                input_window.destroy()  # Close the popup window after a valid choice is made
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    # Create a button to submit the input
    tk.Button(input_window, text="Submit", command=submit_input).pack(pady=10)

    # Wait for the popup window to be closed before continuing
    input_window.wait_window()
    lvls = _lvls.get()

    x, y = ReadSignalFile(file_name)
    binOflvl, levels = [], []
    mnY, mxY = np.min(y), np.max(y)
    # if the given is bits

    bits = math.ceil(math.log2(lvls))
    delta = 1.0 * (mxY - mnY) / lvls
    levels.append(mnY + delta / 2)
    binOflvl.append(to_binary(0, bits))
    for i in range(1, lvls):
        levels.append(levels[i - 1] + delta)
        binOflvl.append(to_binary(i, bits))

    n = x
    Xn = y
    eps = 1e-8
    interval_index, Xqn, EQn, EQ2n, encoded_values = [], [], [], [], []
    for i in range(len(y)):
        idx = bisect.bisect_right(levels, y[i])
        if idx == len(levels):
            idx -= 1
        elif idx != 0:
            if y[i] - levels[idx - 1] <= delta / 2 + eps:
                idx -= 1
        encoded_values.append(binOflvl[idx])
        interval_index.append(idx + 1)
        Xqn.append(levels[idx])
        EQn.append(Xqn[-1] - Xn[i])
        EQ2n.append(EQn[-1] ** 2)
   # Test1
    result = list(zip(encoded_values, Xqn))
    with open(f"Task3 testcases and testing functions/output/Quan1.txt", "w") as file:
        file.write(f"0\n0\n{len(result)}\n")
        for x, y in result:
            file.write(f"{x} {round(y, 2)}\n")
    printTable(n, Xn, interval_index, encoded_values, Xqn, EQn, EQ2n)
    display_quantized_signal(n, Xn, Xqn)
   # Test2
    result = list(zip(interval_index, encoded_values, Xqn, EQn))
    with open(f"Task3 testcases and testing functions/output/Quan2.txt", "w") as file:
        file.write(f"0\n0\n{len(result)}\n")
        for w, x, y, z in result:
            file.write(f"{w} {x} {round(y, 3)} {round(z, 3)}\n")
    printTable(n, Xn, interval_index, encoded_values, Xqn, EQn, EQ2n)
    display_quantized_signal(n, Xn, Xqn)

def display_quantized_signal(x, y, _y):
    x = np.array(x)
    y = np.array(y)
    _y = np.array(_y)

    # Create a polynomial interpolation function (Barycentric)
    spline_interp = CubicSpline(x, y)

    # Generate new x values for plotting
    x_new = np.linspace(x.min(), x.max(), 1000)  # Generate 1000 points
    y_new = spline_interp(x_new)

    plt.figure(figsize=(10, 6))

    # Plot the quantized signal as a ladder graph
    plt.step(x, _y, where='post', color='green', label='Quantized Signal', linewidth=2)

    # Plot the continuous signal using the smooth curve
    plt.plot(x_new, y_new, color='blue', label='Continuous Signal', linewidth=2)

    # Plot the original data points
    plt.scatter(x, y, color='red', s=50, label='Original Data Points', alpha=0.6)

    # Set the x-axis ticks to display only integer values
    plt.xticks(np.arange(x.min(), x.max() + 1, 1))  # Adjust the range for integer ticks

    # Customizing the plot
    plt.title('Continuous and Quantized Signal Representation', fontsize=16, fontweight='bold')
    plt.xlabel('X Axis', fontsize=12)
    plt.ylabel('Y Axis', fontsize=12)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def printTable(n, Xn, interval_index, encoded_values, Xqn, EQn, EQ2n):
    # Create data as a list of rows
    data = list(zip(n, Xn, interval_index, encoded_values, Xqn, EQn, EQ2n))
    # Define column headers
    columns = ["n", "Xn", "Interval Index", "encoded_values", "Xqn", "EQn", "EQ2n"]
    # Print the table using tabulate
    print(tabulate(data, headers=columns, tablefmt="fancy_grid"))
    average_power_error = sum(EQ2n) / len(n)
    print(f"average_power_error = {round(average_power_error, 6)}")


# %%Task4
def compare_signals(calculated_file, expected_file, task):
    calculated_x, calculated_y = ReadSignalFile(calculated_file)
    expected_x, expected_y = ReadSignalFile(expected_file)

    # check lengthes
    if len(calculated_x) != len(expected_x) or len(calculated_y) != len(expected_y):
        print(f"Test case for {task} failed, the lengths don't match")
        return
    for cx, ex, cy, ey in zip(calculated_x, expected_x, calculated_y, expected_y):
        if cx != ex or cy != ey:
            print(f"Test case for {task} failed, the values don't match")
            return
    print(f"Congratulations Test case for {task} passed :) .")


def derivative_signal():
    file_name = select_file(0)
    x, y = ReadSignalFile(file_name)

    d1 = []
    for i in range(1, len(x)):
        d1.append(y[i] - y[i - 1])
    d2 = []
    for i in range(1, len(d1)):
        d2.append(d1[i] - d1[i - 1])

    with open(f"Task4 testcases and testing functions/output/1st_derivative_out.txt", "w") as file:
        file.write(f"0\n0\n{len(d1)}\n")
        for i in range(len(d1)):
            file.write(f"{i} {d1[i]}\n")
    with open(f"Task4 testcases and testing functions/output/2nd_derivative_out.txt", "w") as file:
        file.write(f"0\n0\n{len(d2)}\n")
        for i in range(len(d2)):
            file.write(f"{i} {d2[i]}\n")
            

def moving_average(WS):
    file_name = select_file(0)
    x, y = ReadSignalFile(file_name)

    for i in range(1, len(y)):
        y[i] += y[i - 1]
    y.append(0)
    nx = list(range(len(x) - WS + 1))
    ny = []

    for i in range(len(nx)):
        ny.append(round((y[i + WS - 1] - y[i - 1]) / WS, ndigits=3))

    with open(f"Task4 testcases and testing functions/output/MovingAvg_out.txt", "w") as file:
        file.write(f"0\n0\n{len(nx)}\n")
        for i in range(len(nx)):
            file.write(f"{nx[i]} {int(ny[i]) if ny[i].is_integer() else ny[i]}\n")


def convolution():
    messagebox.showinfo("signal 1", "select 1st signal")
    file_name = select_file(0)
    x1, y1 = ReadSignalFile(file_name)
    x1 = list(map(int, x1))

    messagebox.showinfo("signal 2", "select 2nd signal")
    file_name = select_file(0)
    x2, y2 = ReadSignalFile(file_name)
    x2 = list(map(int, x2))

    mnx = x1[0]
    mxx = x1[-1]
    x = [0] * len(x1)
    for i, vx in zip(x1, y1):
        x[i] = vx

    mnh = x2[0]
    mxh = x2[-1]
    h = [0] * len(x2)
    for i, vh in zip(x2, y2):
        h[i] = vh

    mn = mnx + mnh
    mx = mxx + mxh
    y = [0] * (mx - mn + 1)
    for n in range(mn, mx + 1):
        for k in range(max(mnx, n - mxh), min(mxx, n - mnh) + 1):
            y[n] += x[k] * h[n - k]

    print(mn, mx, y)

    with open(f"Task4 testcases and testing functions/output/Conv_output.txt", "w") as file:
        file.write(f"0\n0\n{len(y)}\n")
        for i in range(mn, mx + 1):
            file.write(f"{i} {int(y[i])}\n")

def DFT_IDFT(b, fs):
    # b = 0 : DFT, b = 1 : IDFT
    file_name = select_file(0)
    x, y = ReadSignalFile(file_name)

    if b == 1:
        y = [cmath.rect(magnitude, angle) for magnitude, angle in zip(x, y)]
    print(y)
    N = len(y)
    xk = []
    c = pow(-1, 1 - b) * 2 * math.pi / N
    for k in range(N):
        t = complex(0, 0)
        for n in range(N):
            theta = c * n * k
            t = t + y[n] * complex((math.cos(theta)), (math.sin(theta)))
            t = complex(round(t.real, 6), round(t.imag, 6))
        t = t * pow(N, -b)
        xk.append(t)

    if b == 0:
        y = [cmath.polar(c) for c in xk]
    else:
        y = [int(c.real) for c in xk]
    if b == 0:
        with open(f"Task5 testcases and testing functions/output/Signal_DFT.txt", "w") as file:
            file.write(f"1\n0\n{len(xk)}\n")
            for m, a in y:
                file.write(f"{m} {a}\n")
    else:
        with open(f"Task5 testcases and testing functions/output/Signal_IDFT.txt", "w") as file:
            file.write(f"0\n0\n{len(xk)}\n")
            for i, r in enumerate(y):
                file.write(f"{i} {r}\n")
    if b == 1:
        display_signals_discrete("Task5 testcases and testing functions/output/Signal_IDFT.txt")
    else:
        w = 2 * math.pi * fs / N
        freqs = []
        for i in range(1, N + 1):
            freqs.append(i * w)
        with open(f"Task5 testcases and testing functions/output/Signal_DFT_magnitude.txt", "w") as file:
            file.write(f"1\n0\n{len(xk)}\n")
            for i in range(N):
                m, a = y[i]
                file.write(f"{freqs[i]} {m}\n")
        with open(f"Task5 testcases and testing functions/output/Signal_DFT_phase_shift.txt", "w") as file:
            file.write(f"1\n0\n{len(xk)}\n")
            for i in range(N):
                m, a = y[i]
                file.write(f"{freqs[i]} {a}\n")
        display_signals_discrete("Task5 testcases and testing functions/output/Signal_DFT_magnitude.txt", "frequency", "Magnitude")
        display_signals_discrete("Task5 testcases and testing functions/output/Signal_DFT_phase_shift.txt", "frequency", "Phase-Shift")
    print(xk)


#%% correlation
def correlation(x1=None, y1=None, x2=None, y2=None):
    if x1 == None:
        x1, y1, x2, y2 = list(), list(), list(), list()
    if len(x1) == 0:
        messagebox.showinfo("signal 1", "select 1st signal")
        file_name = select_file(0)
        x1, y1 = ReadSignalFile(file_name)
        x1 = list(map(int, x1))

        messagebox.showinfo("signal 2", "select 2nd signal")
        file_name = select_file(0)
        x2, y2 = ReadSignalFile(file_name)
        x2 = list(map(int, x2))

    # clac
    # P12
    N = len(y1)
    p_dominator, tmp = 0, 0
    for i in y1: p_dominator += i ** 2
    for i in y2: tmp += i ** 2
    p_dominator *= tmp
    p_dominator = math.sqrt(p_dominator)
    p_dominator /= N
    r = [0] * N
    p = r.copy()
    for i in range(N):
        for j in range(N): r[i] += y1[j] * y2[(i + j) % N]
        r[i] /= N
        p[i] = r[i] / p_dominator
    print(r)
    print(p)
    return p, p.index(max(p))
# correlation()

def time_delay(fs):
    # when calling from button enter call the prompt to enter fs
    _, mx_idx = correlation()
    print(mx_idx, mx_idx / fs)
# time_delay(100)


def ReadXOrYFile(file_name):
    XOrY=[]
    with open(file_name, 'r') as f:
        line = '1'
        while line:
            # process line
            L = line.strip()
            if len(L.split(' '))==1:
                L=line.split(' ')
                V1=float(L[0])                      # changed to float as we need it generateSignal()
                XOrY.append(V1)
                line = f.readline()
            else:
                break
    return XOrY

def template_matching():
    file_name = select_file(0)
    test = ReadXOrYFile(file_name)
    down = [[0.0] * len(test)] * 5
    corr_up, corr_down, up = down.copy(), down.copy(), down.copy()
    print(len(down))
    print(len(down[0]))
    print(np.array(down).shape)

    for i in range(5):
        down[i] = ReadXOrYFile(f'correlation testcases and testing functions/point3 Files/Class 1/down{i + 1}.txt')
    for i in range(5):
        up[i] = ReadXOrYFile(f'correlation testcases and testing functions/point3 Files/Class 2/up{i + 1}.txt')
    for i in range(5):
        corr_up[i], _ = correlation([0], test, [0], up[i])
        corr_down[i], _ = correlation([0], test, [0], down[i])
    mx_up, mx_down = [0.0] * 5, [0.0] * 5
    for i in range(5):
        mx_up[i] = max(corr_up[i])
        mx_down[i] = max(corr_down[i])
    if sum(mx_up) > sum(mx_down):
        print("up")
    else:
        print("down")
template_matching()

# %% GUI
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

button10 = tk.Button(root, text=" Quantize signal ", command=lambda : quantize_signal(""), **button_style)
button10.place(x=960, y=600)  # Added y-padding

# Task4 Text Box
task4_label = tk.Label(root, text=" Task4 ", font=("Helvetica", 14), bg="lightgray", fg="black")
task4_label.place(x=1270, y=300)  # Positioning on the right

# Task4 Buttons
button11 = tk.Button(root, text=" Derivative ", command=derivative_signal, **button_style)
button11.place(x=202, y=573)

button12 = tk.Button(root, text=" Moving Average ", command=lambda: moving_average(int(prompt("Enter Window size:"))), **button_style)
button12.place(x=278, y=728)

button13 = tk.Button(root, text=" Convolution ", command=convolution, **button_style)
button13.place(x=1225, y=189)

button14 = tk.Button(root, text="  DFT   ", command=lambda: DFT_IDFT(0, int(prompt("Enter sampling frequency"))), **button_style)
button14.place(x=313, y=222)

button15 = tk.Button(root, text="  IDFT   ", command=lambda: DFT_IDFT(1, 0), **button_style)
button15.place(x=789, y=222)

root.mainloop()