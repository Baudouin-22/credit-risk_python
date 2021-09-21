# FROM postgres
# ENV POSTGRES_PASSWORD djoudken
# ENV POSTGRES_DB nano_credit
# COPY nano_credit.sql /docker-entrypoint-initdb.d/
# RUN python manage.py makemigrations
# RUN python manage.py migrate

FROM python:3.7.4
ENV PYTHONUNBUFFERED=1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
EXPOSE 8000
RUN pip install -r requirements.txt
COPY . /code/
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]




# FROM python:3
# ENV PYTHONUNBUFFERED 1
# RUN mkdir /backend
# WORKDIR /backend
# COPY requirements.txt /backend/
# EXPOSE 8000
# RUN pip install -r requirements.txt
# COPY . /backend/
# RUN python manage.py makemigrations
# RUN python manage.py migrate
