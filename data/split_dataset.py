import copy

# split a dataset to test and validation datasets

file_name = "./nyuAndSunPartlyResize.odgt"  # full dataset file
validation_path = "./validation10FoldPartlyResize.odgt"  # output validation file
train_path = "./testWithout10FoldPartlyResize.odgt"  # output test file

with open(file_name, 'r') as f:
    lines = f.readlines()
    length = len(lines)
    step = 10
    validation_lines = []
    test_lines = copy.deepcopy(lines)
    for idx in range(0, length, 10):
        validation_lines.append(lines[idx])
        test_lines.remove(lines[idx])

    with open(validation_path, "w") as outfile:
        for validation_line in validation_lines:
            outfile.write(validation_line)

    with open(train_path, "w") as outfile:
        for test_line in test_lines:
            outfile.write(test_line)
