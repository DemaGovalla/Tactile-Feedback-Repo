float Resistor1 = A0; 
float Resistor2 = A1;
float FS = A2;

char userInput;

String dataLabel1 = "Time";
String dataLabel2 = "Displacement";
String dataLabel3 = "Force";
String dataLabel4 = "Work";



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



if(Serial.available()> 0){ 


     userInput = Serial.read(); 
           if(userInput == 'g'){                  // if we get expected value 
//
//            data = analogRead(analogPin);    // read the input pin
//            Serial.println(data); 
                 
//            while(label){ 
//            
//            
//              //enable headers
//              Serial.print(dataLabel1);
//              Serial.print(",");
//              Serial.print(dataLabel2);
//              Serial.print(",");
//              Serial.print(dataLabel3);
//              Serial.print(",");
//              Serial.println(dataLabel4);
//            
//            
//            
//              
//              label = false;
//            }

            R1 = analogRead(Resistor1); // Going to A0, blue/green, top of tweezers
            R2 = analogRead(Resistor2); // Going to A1, yellow/orange, bottom of tweezers
            FSdata = analogRead(FS);

            Vout = (FSdata*5.0)/1023;
            
            Force = Vout*cf;
            //Force = (FSdata*5.0)/1023;
            Displacement = abs(R1-R2);
            Work = Force*Displacement;


//    Serial.print(R1);
//      Serial.print(",");
//    Serial.println(R2);
//    Serial.println(Force);




//            Serial.print(Time,4);
//            Serial.print(",");
            Serial.print(Displacement,2);
            Serial.print(",");
            Serial.print(Force,2);
            Serial.print(",");
            Serial.println(Work,2);
            
            
//            Time += 0.05;
//          //  delay(1000);
//          
            delay(500);

           }
}
}
