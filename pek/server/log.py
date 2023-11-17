from datetime import datetime

"""
from pathlib import Path
import pandas as pd
import time
"""


class Log:
    GRAY = "\033[90m"
    ENDC = "\033[0m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    PINK = "\033[95m"

    @staticmethod
    def print(s):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Log.GRAY}[{time}]{Log.ENDC} {s}{Log.ENDC}")

    """@staticmethod
    def save(event, job, clientAddress=None, iteration=None, elbowK=None):
        currentDate = datetime.now().strftime("%Y-%m-%d")
        currentTime = datetime.now().strftime("%H:%M:%S")
        timestamp = round(time.time(), 2)

        df = pd.DataFrame(
            [
                {
                    "event": event,
                    "date": currentDate,
                    "time": currentTime,
                    "timestamp": timestamp,
                    "type": job.type,
                    "dataset": job.dataset,
                    "n_runs": job.n_runs,
                    "n_clusters": job.n_clusters if hasattr(job, "n_clusters") else "null",
                    "random_state": job.random_state,
                    "min_n_clusters": job.min_n_clusters if hasattr(job, "min_n_clusters") else "null",
                    "max_n_clusters": job.max_n_clusters if hasattr(job, "max_n_clusters") else "null",
                    "iteration": iteration if iteration is not None else "null",
                    "elbow_k": elbowK if elbowK is not None else "null",
                    "job": job.id,
                    "client": clientAddress if clientAddress is not None else "null",
                }
            ]
        )

        folder = Path(__file__).parent.parent.joinpath("logs")
        folder.mkdir(exist_ok=True, parents=True)
        file = folder.joinpath(f"log {currentDate}.csv")
        df.to_csv(file, mode="a", header=not file.exists(), index=False)
        """
