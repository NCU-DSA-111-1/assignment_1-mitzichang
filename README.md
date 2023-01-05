# A simple Neural Network that learns to predict the XOR logic gates

-------------------------------------
# Neural Network in C

This is assignment 1

1. Enter number of neurons in Layer[1]~[4]
## Enter number of neurons in layer[1]: 2
## Enter number of neurons in layer[2]: 4
## Enter number of neurons in layer[3]: 4
## Enter number of neurons in layer[4]: 1

2. Enter the Inputs and Desired Outputs for training example[0]~[3]
## Enter the Inputs for training example[0]:0 0 Enter the Desired Outputs for training example[0]: 0

## Enter the Inputs for training example[1]:0 1 Enter the Desired Outputs for training example[1]: 1

## Enter the Inputs for training example[2]:1 0 Enter the Desired Outputs for training example[2]: 1

## Enter the Inputs for training example[3]:1 1 Enter the Desired Outputs for training example[3]: 0

3. Enter the number for training times
## Enter the number for training(usually 20000):20000

## < Training Time: 19891 >
## Input: (0.00,0.00)      Output: 0
## Input: (0.00,1.00)      Output: 1
## Input: (1.00,0.00)      Output: 1
## Input: (1.00,1.00)      Output: 0

4. Enter input to test
## < You can start testing! >
## Enter input to test: 0 0

5. Output showed
## Output: 0


-------------------------------------
# Compile
make
gcc main.c -lm layer.c neuron.c
# Run
./a.out
