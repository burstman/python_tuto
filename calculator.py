def addition(num1, num2):
    return num1+num2


def substract(num1, num2):
    return num1-num2


def multiply(num1, num2):
    return num1*num2


def divide(num1, num2):
    if num2 == 0:
        return "Error! division by zero"
    return num1/num2


operation = {"+": addition, "-": substract,
             "*": multiply, "/": divide}


def calculator():
    num1 = float(input("Prompt the first number"))
    print("Availble opration: ")
    for symbol in operation:
        print(symbol)
    should_continue = True
    while should_continue:
        op_symbol = input("Prompt the operation")
        num2 = float(input("Prompt the second number"))
        calculation_function = operation.get(op_symbol)
        if calculation_function:
            answer=calculation_function(num1, num2)
            print(f"{num1} {op_symbol} {num2} = {answer}")
            choice = input(f"Type 'y' to continue calculating with {answer}, or 'n' to start a new calculation: ")
            if choice.lower() == 'y':
                num1 = answer
            else:
                should_continue = False
                calculator()  # Start a new calculation
        else:
            print("Invalid operation symbol. Please try again.")
            
calculator()