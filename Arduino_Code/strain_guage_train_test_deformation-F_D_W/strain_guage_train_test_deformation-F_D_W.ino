
float Resistor1 = A0; 
float Resistor2 = A1;
float FS = A2;

String dataLabel1 = "Time";
String dataLabel2 = "Displacement";
String dataLabel3 = "Force";
String dataLabel4 = "Work";
String dataLabel5 = "Label";

//float cf = .625; // calibration factor
float cf = 8.5; // calibration factor

float FSdata = 0;
float Vout;

bool label = true; 
float Time = 0, Force, Displacement, Work;

float R1, R2;
// Label should be 1 or greater
int Label = 9; //within x% in either direction


void setup() {

Serial.begin(9600);
pinMode(Resistor1, INPUT);
pinMode(Resistor2, INPUT);
pinMode(FS, INPUT);

}

void loop() {

while(label){ 


  //enable headers
  Serial.print(dataLabel1);
  Serial.print(",");
  Serial.print(dataLabel2);
  Serial.print(",");
  Serial.print(dataLabel3);
  Serial.print(",");
  Serial.print(dataLabel4);
  Serial.print(",");
  Serial.println(dataLabel5);


  
  label = false;
}

R1 = analogRead(Resistor1); // Going to A0, blue/green, top of tweezers
R2 = analogRead(Resistor2); // Going to A1, yellow/orange, bottom of tweezers
FSdata = analogRead(FS);



Vout = (FSdata*5.0)/1023;

Force = Vout*cf;
//Force = (FSdata*5.0)/1023;
Displacement = abs(R1-R2);
Work = Force*Displacement;


    Serial.print(R1);
      Serial.print(",");
    Serial.println(R2);
    Serial.println(Force);




//  Serial.print(Time,4);
//  Serial.print(",");
//  Serial.print(Displacement,2);
//  Serial.print(",");
//  Serial.print(Force,2);
//  Serial.print(",");
//  Serial.print(Work,2);
//  Serial.print(",");
//  Serial.println(Label); 
  
  
  Time += 0.5;
//  delay(1000);
// 15 sample a second (15Hz)
  delay(500);

}
