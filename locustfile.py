from locust import HttpUser, TaskSet, task, between
from pathlib import Path

class UserBehavior(TaskSet):
    def on_start(self):
        # Load a sample image once to send with requests
        self.sample_image_path = Path("sample.jpg")
        if not self.sample_image_path.exists():
            raise FileNotFoundError("sample.jpg not found in root directory")

    @task
    def predict(self):
        with open(self.sample_image_path, "rb") as img_file:
            files = {"file": ("sample.jpg", img_file, "image/jpeg")}
            self.client.post("/predict/", files=files)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # wait 1-3 seconds between tasks
