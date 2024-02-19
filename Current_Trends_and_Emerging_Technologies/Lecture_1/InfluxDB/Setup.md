# Setup

* Move to the director `Influx_Grafana` and run there `docker-compose up`
* Now, http://localhost:8086/signin as well as http://localhost:3000/login should be available
* Credentials for Influx: `myuser` `my_password`
* Credentails for Grafana: `admin` `admin` (skip when asked for a new password :) )


Hint for setup of Grafana (we will do that together):

* Query Language: FLUX
* URL: http://influx:8086/
* Basic auth
* organization: primary
* token: my_password
