## MQTT Explorer

* Is a debugging tool for seeing the values of the broker
* Is also a MQTT Client!

https://mqtt-explorer.com/

## MQTT Broker

* Use PowerShell
* Ensure that Docker is installed and running on your machine
* Download the latest image: `docker pull eclipse-mosquitto`
* Check the download with `docker images`
* Move to the this folder (Broker) with power shell (we will mount the mosquitto.conf)
* Start the broker with `docker run -it -p 1883:1883 -v ${PWD}/mosquitto.conf:/mosquitto/config/mosquitto.conf -v ${PWD}/data:/mosquitto/data eclipse-mosquitto`  (ensure that those ports are not in use!,
`${PWD}` is for powershell: if you are using any other system, use the absolute file path!)
* Connect with MQTT Explorer (host: localhost) and check if connection is working!

