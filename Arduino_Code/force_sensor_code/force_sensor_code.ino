
//#define FS A0
float FS = A0;



String dataLabel1 = "Time";
//String dataLabel2 = "Displacement";
String dataLabel3 = "Force";
//String dataLabel4 = "Work";
//String dataLabel5 = "Label";


bool label = true; 

float FSdata;
float Time = 0, Force;

//, Displacement, Work;

// Label should be 1 or greater
int Label = 9; //within x% in either direction


void setup() {

Serial.begin(115200);
pinMode(FS, INPUT);

}

void loop() {

while(label){ 


  //enable headers
  Serial.print(dataLabel1);
//  Serial.print(",");
//  Serial.print(dataLabel2);
  Serial.print(",");
  Serial.println(dataLabel3);
//  Serial.print(",");
//  Serial.print(dataLabel4);
//  Serial.print(",");
//  Serial.println(dataLabel5);


  
  label = false;
}


//FSdata = analogRead(FS);


Force = abs(1023-analogRead(FS));



//Force = (FSdata*5.0)/1023;


//Work = Force*Displacement;


  Serial.print(Time,4);
  Serial.print(",");
  Serial.println(Force);




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
  delay(20);

}
