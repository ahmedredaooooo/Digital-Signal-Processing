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

def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")

# x1, y1 = ReadSignalFile("output/out.txt")
# Compare_Signals("Testcase 1\LPFCoefficients.txt", x1, y1)
# Compare_Signals("Testcase 3\HPFCoefficients.txt", x1, y1)
# Compare_Signals("Testcase 5/BPFCoefficients.txt", x1, y1)
# Compare_Signals("Testcase 7/BSFCoefficients.txt", x1, y1)
# x1, y1 = ReadSignalFile("../Task4 testcases and testing functions/output/Conv_output.txt")
# Compare_Signals("Testcase 2/ecg_low_pass_filtered.txt", x1, y1)
# Compare_Signals("Testcase 4/ecg_high_pass_filtered.txt", x1, y1)
# Compare_Signals("Testcase 6/ecg_band_pass_filtered.txt", x1, y1)
# Compare_Signals("Testcase 8/ecg_band_stop_filtered.txt", x1, y1)

