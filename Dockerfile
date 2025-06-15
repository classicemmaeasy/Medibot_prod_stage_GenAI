FROM python:3.10.11-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]



# FROM python:3.10.11-slim-buster

# # Set the working directory
# WORKDIR /app

# # Copy only requirements.txt first to leverage Docker cache
# COPY requirements.txt ./

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# # Expose the port your app runs on (adjust if different)
# EXPOSE 8080

# # Define the default command to run the application
# CMD ["python3", "app.py"]
