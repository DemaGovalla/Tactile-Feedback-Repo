
float Resistor1 = A0; 
float Resistor2 = A1;

String dataLabel1 = "Time";
String dataLabel2 = "Force";
String dataLabel3 = "Range";
String dataLabel4 = "CoR";
String dataLabel5 = "Variance";
String dataLabel6 = "STD";
String dataLabel7 = "Work";
String dataLabel8 = "Label";

//String dataLabel3 = "CoF";

bool label = true; 
float Time = 0, Force, Displacement, Work;

float R1, R2, curr1, curr2, percent = 0.05;
// Label should be 1 or greater
int Label = 5, threshold = 1024*percent, samples = 30;; //within x% in either direction

void setup() {

Serial.begin(9600);
pinMode(Resistor1, INPUT);
pinMode(Resistor2, INPUT);

}

void loop() {


while(label){ 
//  //enable headers
//  Serial.print(dataLabel1);
//  Serial.print(",");
//  Serial.print(dataLabel2);
//  Serial.print(",");
////  Serial.print(dataLabel3);
////  Serial.print(",");
////  Serial.print(dataLabel4);
////  Serial.print(",");
////  Serial.print(dataLabel5);
////  Serial.print(",");
////  Serial.print(dataLabel6);
////  Serial.print(",");
//  Serial.println(dataLabel7);


  //enable headers
  Serial.print(dataLabel1);
  Serial.print(",");
  Serial.print(dataLabel2);
  Serial.print(",");
  Serial.print(dataLabel3);
  Serial.print(",");
  Serial.print(dataLabel4);
  Serial.print(",");
  Serial.print(dataLabel5);
  Serial.print(",");
  Serial.print(dataLabel6);
  Serial.print(",");
  Serial.print(dataLabel7);
  Serial.print(",");
  Serial.println(dataLabel8);


  
  label = false;
}




float Work_sum = 0.0, Force_sum = 0.0, Force_ave, Work_ave, variance, std, range, coeffOfRange;
int classes;
float Work[samples]= {}, Force[samples] = {};
float sqDevSum = 0.0;



for (int i=0; i<samples; i++)
{
R1 = analogRead(Resistor1); // Going to A0, yellow/orange, bottom of tweezers
R2 = analogRead(Resistor2); // Going to A1, blue/green, top of tweezers

   Force[i] = R1;

   // use work and the sqrt of work
   // comment out this loop and collect just work and sqrt of work data. 

   //another idea is to takew the variaNCE OF THE WORK AND tjhen the std and use the std. 
   Work[i] = R1*(abs(R1-R2));
   Force_sum += Force[i];
   Work_sum += Work[i];

//  Serial.print(R1,4) // Going to A1, blue/green, top of tweezers 
//  Serial.print(",");
//  Serial.println(sum,4); // Going to A1, blue/green, top of tweezers 
//    Serial.println(R1);
//    Serial.println(R2);
//    Serial.println(Force[i]);
//    Serial.println(Work[i]); 
}


Force_ave = Force_sum/samples;
Work_ave = Work_sum/samples;


float minimum = Force[1], maximum = Force[1];
//Find the Range and Range of Coefficient
for (int i=0; i<samples; i++)
{
    minimum = min(Force[i], minimum );
    maximum = max(Force[i], maximum);



}

// Set up for the STD of the sensors
for (int i=0; i<samples; i++)
 {
  // pow(x, 2) is x squared. Basically used to square the (average - float(arr[i]) computation. 
  sqDevSum += pow((Force_ave - float(Force[i])), 2);
 }

  range = maximum - minimum;
  coeffOfRange = (100*range) / (maximum + minimum);
  variance = sqDevSum/(samples-1);
  std = sqrt(variance);

  Serial.print(Time,4);
  Serial.print(",");
  Serial.print(Force_ave,4);
  Serial.print(",");
  Serial.print(range,2);
  Serial.print(",");
  Serial.print(coeffOfRange,2);
  Serial.print(",");
  Serial.print(variance,2);
  Serial.print(",");
  Serial.print(std,4);
  Serial.print(",");
  Serial.print(Work_ave,2);
  Serial.print(",");
  Serial.println(Label); 
  
  
  Time += 0.05;
  delay(1000);
// 15 sample a second (15Hz) !!! sometimes 16Hz
//  delay(50);

}




//// Calculate Work done
//Force = R1;
//Displacement = R1 - R2;
//Work = abs(Force*Displacement);
//
//// Calculate the Coefficient of friction
//CoF = (sqrt(abs(R1 - R2)))/Weight;
//
//
//// At percent = 0.05, we are looking at +- 50 steps from our previous data. (change data to Time/Work/CoF/Label).
////if((curr1 >=data1+threshold || curr1 <=data1-threshold) || curr2>=data+threshold || curr2<=data2-threshold) ){
////
////  Serial.print(Time,4);
////  Serial.print(",");
////  Serial.print(Work,4);
////  Serial.print(",");
////  Serial.print(CoF ,4); 
////  Serial.print(",");
////  Serial.println(Label); 
//
//
//  Serial.print(Time,4);
//  Serial.print(",");
//  Serial.print(R1,4); // Going to A0, yellow/orange, bottom of tweezers
//  Serial.print(",");
//  Serial.print(R2,4); // Going to A1, blue/green, top of tweezers 
//  Serial.print(",");
//  Serial.print(CoF ,4); 
//  Serial.print(",");
//  Serial.println(Label); 
////
//  Time += 1;
//
// delay(1000);
//
//
////  Time += 0.05;
//////  curr1 = data1;
//////  curr2 = data2;
//////}
//// delay(50);
//
//}
