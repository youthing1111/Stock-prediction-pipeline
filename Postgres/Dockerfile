FROM postgres:13

# set work directory
WORKDIR /Postgres

# copy project
COPY . /Postgres/
#COPY ACB_price.csv /var/lib/postgresql/data/initdata
COPY schema.sql /docker-entrypoint-initdb.d/schema.sql
