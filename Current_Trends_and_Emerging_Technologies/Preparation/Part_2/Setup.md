# Setup

* Move to the directory `Influx_Grafana` and run there `docker-compose up`
* Now, http://localhost:8086/signin as well as http://localhost:3000/login should be available
* Credentials for Influx: `admin` `my_password`
* Credentails for Grafana: `admin` `admin` (skip when asked for a new password or remember it :) )


Hint for setup of Grafana (we will do that together):

* Query Language: FLUX
* URL: http://influx:8086/
* Basic auth
* user: admin
* organization: primary