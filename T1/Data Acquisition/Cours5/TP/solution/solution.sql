.open covid.sqlite
.mode csv
.import countries.csv countries
.import owid-covid-data.csv covid
.import population.csv population
.schema
.headers on
.mode column
.width 40
.timer on
select sum(new_deaths) as deaths from covid;
-- 0.017
select name,sum(new_deaths) as deaths from covid join countries on iso_code="alpha-3" group by iso_code order by deaths desc limit 10;
-- 0.128
select name,round(1000*sum(new_deaths)/population,2) as deaths from covid join countries on iso_code="alpha-3" join population on population.country=iso_code group by iso_code order by deaths desc limit 10;
-- 0.178
select name,round(1000*sum(new_deaths)/population,2) as deaths from covid join countries on iso_code="alpha-3" join population on population.country=iso_code where region='Europe' group by iso_code order by deaths desc limit 10;
-- 0.203

create index countries_alpha3 on countries("alpha-3");
create index population_country on population(country);
select name,sum(new_deaths) as deaths from covid join countries on iso_code="alpha-3" group by iso_code order by deaths desc limit 10;
-- 0.099
select name,round(1000*sum(new_deaths)/population,2) as deaths from covid join countries on iso_code="alpha-3" join population on population.country=iso_code group by iso_code order by deaths desc limit 10;
-- 0.149
select name,round(1000*sum(new_deaths)/population,2) as deaths from covid join countries on iso_code="alpha-3" join population on population.country=iso_code where region='Europe' group by iso_code order by deaths desc limit 10;
-- 0.172
