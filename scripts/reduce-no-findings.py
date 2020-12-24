import sys

file = open('Data_Entry_2017.csv', 'r+')
new_csv = ""

print("Before reducing No Findings")

no_findings = file.read().count("No Finding")
file.seek(0)

before_edema = file.read().count("Edema")
file.seek(0)

before_cardiomegaly= file.read().count("Cardiomegaly")
file.seek(0)

print("No Finding:", no_findings)

max_NO_FINDING = 8000
no_findings_count = 0
header = True

line = file.readline()

while line != "":
    # add header line with no processing
    if header:
        new_csv += line
        header = False

    if "No Finding" in line:
        no_findings_count += 1
        if no_findings_count <= max_NO_FINDING:
            new_csv += line
    else:
        # other finding, no filtering needed
        new_csv += line

    line = file.readline()


file.close()

print("\nAfter reducing No Findings")
no_findings = new_csv.count("No Finding")
print("No Finding:", no_findings)

after_edema = new_csv.count("Edema")
after_cardiomegaly = new_csv.count("Cardiomegaly")

assert before_edema == after_edema
assert before_cardiomegaly == after_cardiomegaly

outputFile = open('Data_Entry_2017_w_reduced_no_finding.csv', 'w')
outputFile.write(new_csv)
outputFile.flush()
outputFile.close()


