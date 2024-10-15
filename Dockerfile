# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /newton

# Copy the current directory contents into the container at /newton
COPY . /newton

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for Streamlit
EXPOSE 8501

# Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py"]
