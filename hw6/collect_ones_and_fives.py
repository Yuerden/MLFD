train = open("ZipDigits.train", "r")
test = open("ZipDigits.test", "r")
my_train = open("zip.train", "w")
my_test = open("zip.test", "w")

ones = 0
fives = 0
for x in train:
    # print(x[0] + x[1])
    if x[0] == "1":
        my_train.write(x)
        ones+=1
    if x[0] == "5":
        my_train.write(x)
        fives += 1
print(f"Ones in train: {ones}")
print(f"Ones in train: {fives}")

ones = 0
fives = 0
for x in test:
    if x[0] == "1":
        my_test.write(x)
        ones+=1
    if x[0] == "5":
        my_test.write(x)
        fives += 1
print(f"Ones in test: {ones}")
print(f"Ones in test: {fives}")