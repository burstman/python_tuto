print("*****Welcome to Python Pizza Deliveries*****")
size = input("What is the size of pizza you want: ")
add_pepperoni = input("Do you want to add pepperoni to your pizza ?: ")
extra_cheese = input("Do you want extra cheese to your pizza?: ")
# Variable inzialization
price = 0
input_size_pizza = True
input_extra = True

# input condition for the extra 
if add_pepperoni not in ["Y", "y", "N", "n"] or extra_cheese not in ["Y", "y", "N", "n"]:
    input_extra_pizza = False

#condition for the size of pizza
if size in ["L", "l"]:        
    price += 25
    if add_pepperoni in ["Y", "y"]:
        price += 3
    elif extra_cheese in ["Y", "y"]:
        price += 1
elif size in ["M", "m"]:
    price += 20
    if add_pepperoni in ["Y", "y"]:
        price += 3
    elif extra_cheese in ["Y", "y"]:
        price += 1
elif size in ["S", "s"]:
    price += 15
    if add_pepperoni in ["Y", "y"]:
        price += 2
    elif extra_cheese in ["Y", "y"]:                   
        price += 1
else:
    input_size_pizza = False

if input_size_pizza and input_extra:
    print(f"Your total bill is {price} $")
else:
    if not input_size_pizza:
        print("please prompt the right size of pizza")
    elif not input_extra:
        print("please prompt the right extra")
