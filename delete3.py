# importing the multiprocessing module 
import multiprocessing 
from multiprocessing import Process, Queue
from time import time

def getSumForFirstHalf(start , end, res_dict): 
	firstHalfSum = 0
	for x in range(start, end):
		firstHalfSum += x
	res_dict['firstHalfSum'] = firstHalfSum

def getSumForThirdHalf(start , end, res_dict): 
	thirdHalfSum = 0
	for x in range(start, end):
		thirdHalfSum += x
	res_dict['thirdHalfSum'] = thirdHalfSum

def getSumForFourthHalf(start , end, res_dict): 
	fourthHalfSum = 0
	for x in range(start, end):
		fourthHalfSum += x
	res_dict['fourthHalfSum'] = fourthHalfSum
	
		
def getSumForSecondHalf(start , end, res_dict): 
	secondHalfSum = 0
	for x in range(start, end):
		secondHalfSum += x
	res_dict['secHalfSum'] = secondHalfSum

def main():
	mgr = multiprocessing.Manager()
	results_dict = mgr.dict()
	
	p1 = multiprocessing.Process(target=getSumForFirstHalf, args=(1, 25_000_001,results_dict )) 
	p2 = multiprocessing.Process(target=getSumForSecondHalf, args=(25_000_001,50_000_001,results_dict )) 
	p3 = multiprocessing.Process(target=getSumForThirdHalf, args=(50_000_001, 75_000_001,results_dict )) 
	p4 = multiprocessing.Process(target=getSumForFourthHalf, args=(75_000_001,100_000_001,results_dict )) 
	
	
	startTime = time()

	# starting process 1 
   
	
	p1.start() 
	# starting process 2 
	p2.start() 
	p3.start()
	p4.start()

	#wait until process 1 is finished 
	p1.join() 
	# wait until process 2 is finished 
	p2.join() 
	p3.join()
	p4.join()

	print(results_dict.values())
	#merge firsthaldsum and sechalfsum

	finalsum = 0
	for key in results_dict:
		finalsum += results_dict[key]
	
	print(f"final sum {finalsum}")

	end_time=time()
	totalTime = end_time - startTime
	print(totalTime) 
	
	

if __name__ == "__main__": 
	main()





# import os
# from time import time

# def main():
# 	#print(os.cpu_count())
# 	sum = 0
	
# 	start_time=time()    
	
# 	for x in range(1,100_000_001):
# 		sum += x
		

	
# 	print(f"final sum {sum}")
# 	end_time=time()
# 	totalTime = end_time - start_time
# 	print(totalTime)
	
   
	

# if __name__ == "__main__": 
# 	main()



import multiprocessing, os
from time import time

def getSumForHalf(start, end, res_dict):
    halfSum = 0
    for x in range(start, end):
        halfSum += x
    res_dict[start] = halfSum

def main():
    # print(os.cpu_count())
    # num_halves = 5  # Number of halves
    # num_halves = os.cpu_count()  # Number of halves
    num_halves = 10  # Number of halves


    mgr = multiprocessing.Manager()
    results_dict = mgr.dict()

    processes = []
    start = 1
    end = 100_000_001
    step = (end - start) // num_halves

    startTime = time()

    for i in range(num_halves):
        p = multiprocessing.Process(target=getSumForHalf, args=(start, start + step, results_dict))
        processes.append(p)
        start += step

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    final_sum = sum(results_dict.values())
    # final_sum = (sum(results_dict.values()))/(len(results_dict))

    print(f"Final sum: {final_sum}")

    end_time = time()
    totalTime = end_time - startTime
    print(f"Total time: {totalTime}")

    print(len(results_dict))

if __name__ == "__main__":
    main()



# import multiprocessing
# from time import time

# def getSumForHalf(start, end, res_dict):
#     halfSum = 0
#     for x in range(start, end):
#         halfSum += x
#     res_dict[start] = halfSum

# def main():
#     max_halves = 12  # Maximum number of halves
#     lowest_time = float('inf')
#     lowest_time_halves = 1

#     for num_halves in range(1, max_halves + 1):
#         mgr = multiprocessing.Manager()
#         results_dict = mgr.dict()

#         processes = []
#         start = 1
#         end = 100_000_001
#         step = (end - start) // num_halves

#         startTime = time()

#         for i in range(num_halves):
#             p = multiprocessing.Process(target=getSumForHalf, args=(start, start + step, results_dict))
#             processes.append(p)
#             start += step

#         for p in processes:
#             p.start()

#         for p in processes:
#             p.join()

#         final_sum = sum(results_dict.values())
#         print(f"For {num_halves} halves - Final sum: {final_sum}")

#         end_time = time()
#         totalTime = end_time - startTime
#         print(f"For {num_halves} halves - Total time: {totalTime}")

#         if totalTime < lowest_time:
#             lowest_time = totalTime
#             lowest_time_halves = num_halves

#     print(f"The lowest time was for {lowest_time_halves} halves: {lowest_time} seconds")

# if __name__ == "__main__":
#     main()

