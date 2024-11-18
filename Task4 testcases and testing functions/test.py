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
    print(f"Congratulations Test case for {task} passed ;) .")

compare_signals("output/1st_derivative_out.txt",
                "Derivative testcases/1st_derivative_out.txt",
                "Calculating 1st Derivative")
compare_signals("output/2nd_derivative_out.txt",
                "Derivative testcases/2nd_derivative_out.txt",
                "Calculating 2nd Derivative")
print()

compare_signals("output/MovingAvg_out.txt",
                "Moving Average testcases/MovingAvg_out1.txt",
                "Calculating Moving Average (smoothed signal)")
