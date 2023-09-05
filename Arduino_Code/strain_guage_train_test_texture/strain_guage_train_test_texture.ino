
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
float Time = 0, Force, Weight=.2, CoF;

float R1, R2, curr1, curr2, percent = 0.05;
// Label should be 1 or greater
int Label = 3, threshold = 1024*percent, samples = 30;; //within x% in either direction

void setup() {

Serial.begin(9600);
pinMode(Resistor1, INPUT);
pinMode(Resistor2, INPUT);

}

void loop() {


//while(label){ 
//  //enable headers
//  Serial.print(dataLabel1);
//  Serial.print(",");
//  Serial.print(dataLabel2);
//  Serial.print(",");
//  Serial.print(dataLabel3);
//  Serial.print(",");
//  Serial.print(dataLabel4);
//  Serial.print(",");
//  Serial.print(dataLabel5);
//  Serial.print(",");
//  Serial.print(dataLabel6);
//  Serial.print(",");
//  Serial.print(dataLabel7);
//  Serial.print(",");
//  Serial.println(dataLabel8);
//  label = false;
//}




float CoF_sum = 0.0, Force_sum = 0.0, Force_ave, CoF_ave, variance, std, range, coeffOfRange, Weight = 0.2;
int classes;
float CoF[samples]= {}, Force[samples] = {};
float sqDevSum = 0.0;



for (int i=0; i<samples; i++)
{
R1 = analogRead(Resistor1); // Going to A0, yellow/orange, bottom of tweezers
R2 = analogRead(Resistor2); // Going to A1, blue/green, top of tweezers

   Force[i] = abs(R1-R2);
   Force_sum += Force[i];
   CoF[i] = (abs(R1 - R2))/Weight;
   CoF_sum += CoF[i];

//  Serial.print(R1,4) // Going to A1, blue/green, top of tweezers 
//  Serial.print(",");
//  Serial.println(sum,4); // Going to A1, blue/green, top of tweezers 
//    Serial.println(R1);
//    Serial.println(CoF[i]);
    
}

    Serial.print(R1);
    Serial.print(",");
    Serial.println(R2);

Force_ave = Force_sum/samples;
CoF_ave = CoF_sum/samples;


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

//  Serial.print(Time,4);
//  Serial.print(",");
//  Serial.print(Force_ave,4);
//  Serial.print(",");
//  Serial.print(range,2);
//  Serial.print(",");
//  Serial.print(coeffOfRange,2);
//  Serial.print(",");
//  Serial.print(variance,2);
//  Serial.print(",");
//  Serial.print(std,4);
//  Serial.print(",");
//  Serial.println(CoF_ave,2);
//  Serial.print(",");
//  Serial.println(Label); 
  
  
//  Time += 0.05;
  delay(100);

//  delay(50);

}
