# Lecture 1

# MQTT

## What is MQTT?
- MQTT stands for Message Queuing Telemetry Transport
- It is a lightweight messaging protocol for small sensors and mobile devices, optimized for high-latency or unreliable networks

## How does it work?
- It is based on the publish-subscribe model
- It is designed to be used with TCP/IP, but any network protocol that provides ordered, lossless, bi-directional connections can support MQTT

## The concept of publish-subscribe
- In the publish-subscribe model, a sender (publisher) sends a message to a topic
- A receiver (subscriber) can subscribe to a topic and receive messages from it

## Quality of Service
- MQTT supports three levels of Quality of Service (QoS)
    - QoS 0: At most once delivery
    - QoS 1: At least once delivery
    - QoS 2: Exactly once delivery

## MQTT Broker
- The MQTT broker is a server that receives all messages from the clients and then routes the messages to the appropriate destination clients

## MQTT Client
- The MQTT client is a device that connects to the MQTT broker and can publish messages to a topic or subscribe to a topic to receive messages    

## MQTT Topics
- Topics are used to filter messages
- A topic is a string that the broker uses to filter messages for each connected client

# INFLUXDB
- InfluxDB is an open-source time series database
- Schema-less design
- TIme series data is indexed by a timestamp and a set of tags
- Retention period: The duration for which data is stored in the database (TTL: Time to Live)   
## Anatomy of InfluxDB
- Bucket: A bucket is a container for time series data (SQL equivalent of a database)
- Measurement: A measurement is a collection of fields and tags (SQL equivalent of a table)
- Field: A field is a key-value pair (SQL equivalent of an unindexed column)
- Tag: A tag is a key-value pair  (SQL equivalent of an indexed column)
- Point: A point is a single data record in InfluxDB (SQL equivalent of a row)

- Tagging: Tagging is the process of adding metadata to the data points
