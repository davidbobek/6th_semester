# PART 1

1. Docker pull the mosquitto image
```bash
docker pull eclipse-mosquitto
```
2. Mount the .conf file to the docker container
```bash
docker run -it -p 1883:1883 -v /home/bobek/University/6th_semester/Current_Trends_and_Emerging_Technologies/Preparation/Part_1/mosquitto.conf:/mosquitto/config/mosquitto.conf -v /home/bobek/University/6th_semester/Current_Trends_and_Emerging_Technologies/Preparation/Part_1/data:/mosquitto/data eclipse-mosquitto
```

# PART 2

Inside the Influx Grafana folder run
```bash
docker compose up 
```

# Resulting in
```bash
localhost:8086/signin
localhost:3000/login
```
