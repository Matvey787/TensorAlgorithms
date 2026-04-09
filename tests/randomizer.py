from random import randint
import sys

def generate_numbers():
    if len(sys.argv) < 4:
        print("Usage: python script.py <filename> <min> <max> <amount>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        min_val = int(sys.argv[2])
        max_val = int(sys.argv[3])
        amount = int(sys.argv[4])
    except ValueError:
        print("Error: min, max, and amount must be integers")
        sys.exit(1)
    
    if min_val > max_val:
        print("Error: min cannot be greater than max")
        sys.exit(1)
    
    if amount <= 0:
        print("Error: amount must be positive")
        sys.exit(1)
    
    with open(filename, "w", encoding="utf-8") as wFile:
        numbers = [str(amount)]
        for i in range(amount):
            numbers.append(str(randint(min_val, max_val)))
        
        wFile.write(' '.join(numbers))

if __name__ == "__main__":
    generate_numbers()