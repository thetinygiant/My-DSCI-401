# def flatten(*list):
    # if type(list[0]) == type([]):
        # return list_held_here + list[0]
	# else:
	    # lists_held_here = []
	    # list([]) == flatten([])
		# rest = flatten(list[1:])
	# return list[0] +flatten(list)
		
# print(flatten([1,2,3],[1,2,4],[[3,4],[2]]))




# #Need to find if the list is first
# #find whether the list is a list or a list of lists.
# #"rest" looks like [1,2,3] + ft[[
# #ft([]) = ([])

# def power_set(list):
	# if len(list) == 1:
		# return map(lambda x: [x], list[0])
	# else:
		# rest = power_set(list[1:]), len(list) - 1
# return map(lambda x: list[0] + x, + rest)

# print(power_set([1,2,3]))
	
	
def flatten(list):
    if len(list) == 0:
        return list                 #This is the base case
    rest = list[1:]                 #Puting the rest of the list into a variable
    if type(list[0]) == type([]):   #Testing if the input is a list
        return flatten(list[0]) + flatten(rest)    #What to do if the input is a list 
    else:
	    return [list[0]] + flatten(rest)    #What happens if the input is not a list
		
print(flatten([[1,2,3],[1,2,4],[[3,4],[2]]]))  


                                 
#Problem 2:  "Powerset"+
def power_set(list):         #Subset  []     so our base case will be len(list) == 0,   Notice how the length of the list are declining by one.  
	if list == ([]):      #        [3],[2],[1]
		return [[]]
	else:
		rest = power_set(list[1:])
		return map(lambda x: [list[0]] + x, rest) + rest			
print(power_set([1,2,3]))


def all_perms(lst):
	if len(lst) == 1:
		return [lst]
	else:
		rest = all_perms(lst[1:])
		perms = []
		for sp in rest:
			for i in range (len(sp)+1):
				np = list(sp)
				np.insert(i,lst[0])
				perms.append(np)  
		return perms
print(all_perms([1,2,3]))

def spiral(n,end_corner):
	d_vect = [[1,0],[0,1],[-1,0],[0,-1]]     #directional vectors
	max_num = n**2-1
	number_list = range(max_num+1)   #Creating list of numbers from 0 to n**2-1
	for vect in d_vect:
		current_dir = d_vect.index(vect)
		for number in number_list:               		
			if not type(n) == int or not type(n) == float(n):        #Test whether n is and integer
				return "n must be a positive integer"
			else:
				if end_corner == 1:                    # from num 0 the next move for # of end corner should correspond w index value of d_vect
					d_vect.index(2)
					if end_corner == 2:
						d_vect.index(3)
						if end_corner == 3:
							d_vect.index(1)
							if end_corner == 4:
								d_vect.index(0)	
        row = 0
	    column = 0
	    current_pos = [row,column]
		
		#What I want to do:
		# I can figure out which corner the spiral will end on based on which direction vector I choose.  
		# I need to start there and then interate throught the list of d_vects
		#in order to be able to create a spiral I need to know when to create a stopping point.  I should hit a corner when the number is greater than current_n +1
		#my array for n = 4 end corner = 4 should look like this:
		
		
		
		#		  (1,0),(0,1),(0,1),(0,1)	                  
		#		  (1,0),(1,0),(0,1),(-1,0)                    
		#		  (1,0),(0,0),(-1,0),(-1,0)
		#		 (0,-1),(0,-1),(-1,0),(-1,0)
						     # having trouble implementing interation of direction vectors to achieve this with the number_list i've created which ranges from 0 to n**2-1
print(spiral(8,2))
				