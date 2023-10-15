#variable initialization
shoppinglist = []
action = ""
inputError = False
#loop for manuplation of the shpping list stop when action = quit

while action != "quit":
    print("--------------------------------------------------")
    action = input("Please use:\n [add] to add item in the shopping list\n [remove] to remove item from shopping list\n\
 [view] to view item from shopping list\n [quit] to quit from shopping list\n")
    print("--------------------------------------------------")
    if action not in ("add", "remove", "view", "quit"): #condition inputError, if the input is not in the tuple inputError will be true
        inputError = True
    #add item condition Max 10 item
    if action == "add" and inputError == False:
        print("[q] to quit adding, you can only add 10 items")
        for i in range(1, 11):
            add = input("add item"+str(i)+": ")
            if add == "q":
                break
            shoppinglist.append(add)
    #view item condition list all items with their indexes recpectivelly
    elif action == "view" and inputError == False:
        for index, item in enumerate(shoppinglist):
            print(f"item {index+1}:{item} ")
    #remove only one item at time 
    elif action == "remove" and inputError == False:
        remove = input("Prompt the name of the item to remove: ")
        if remove not in shoppinglist:
            print("Item does not exist the shopping list")
        else:
            shoppinglist.remove(remove)
            print("Item succefully removed ")    
    #force user to put respective command
    if inputError == True:
        print("Error: Please promt the right command")
