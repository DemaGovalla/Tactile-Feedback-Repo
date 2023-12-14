# importing the multiprocessing module 
import multiprocessing 
from multiprocessing import Process, Queue
from time import time



def testAlgo(X_test, y_test, results_dict): 
	# Perform computation (example: calculating the sum of the array)
	result = sum(X_test)
	
	# Update results_dict with the label and result
	results_dict['firstHalfSum'] = result
	

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
	

	test1 = [1,2,3,4]
	y1 = [1]
	
	p1 = multiprocessing.Process(target=getSumForFirstHalf, args=(1, 25_000_001,results_dict )) 
	p2 = multiprocessing.Process(target=getSumForSecondHalf, args=(25_000_001,50_000_001,results_dict )) 
	p3 = multiprocessing.Process(target=getSumForThirdHalf, args=(50_000_001, 75_000_001,results_dict )) 
	p4 = multiprocessing.Process(target=getSumForFourthHalf, args=(75_000_001,100_000_001,results_dict )) 
	
	p5 = multiprocessing.Process(target=testAlgo, args=(test1, y1, results_dict )) 
	
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
