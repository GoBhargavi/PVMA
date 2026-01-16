import sqlite3
from multiprocessing import Pool
from time import sleep
import datetime
import os


def connect_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    save = conn.commit
    close = conn.close
    return c, save, close


def get_waiting_jobs(last_job_id):
    """Check if there is a job to be done."""
    
    """
        user_id INTEGER NOT NULL,
        job_id varchar(255) NOT NULL UNIQUE,
        status varchar(255),
        start_date varchar(255),
        end_date varchar(255),
        criteria varchar(255)
    """
    
    cursor, _, close = connect_db()
    
    # get all the jobs where status is waiting

    job_query = """
    SELECT user_id, bucket_id, job_id, id
    FROM jobs
    WHERE status = ? AND id > ?
    """

    cursor.execute(job_query, ("waiting", last_job_id))
    
    
    result = [dict(zip(['user_id', 'bucket_id', 'job_id', "id"], row)) for row in cursor.fetchall()]

    close()
    if len(result) == 0:
        return result, last_job_id
    else:
        return result, result[-1]["id"]


def start_job(job):
    """Start a job."""
    try:
        from tourist_trend import main
        bucket_id = job["bucket_id"]
        climate_data = os.path.join(bucket_location,bucket_id,"climate_data.csv")
        google_trends = os.path.join(bucket_location,bucket_id,"google_trends_data.csv")
        statistical_data = os.path.join(bucket_location,bucket_id,"statistical_data.csv")
        filename = os.path.join(output_location, job["job_id"])
        
        final_result = main(statistical_data, climate_data, google_trends,filename)
        
        return {"status": "completed", "result":final_result}
    except Exception as e:
        print(f"[!] Error: ({job['job_id']}) ", e)
        return {"status": "failed"}

def update_job_status(job_id, status):
    """Update the job status."""
    cursor, save, close = connect_db()
    
    if status == "completed":
        end_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "UPDATE jobs SET status = ?, end_date = ? WHERE job_id = ?",
            (status, end_date, job_id),
        )
    else:
        cursor.execute(
            "UPDATE jobs SET status = ? WHERE job_id = ?",
            (status, job_id),
        )
    
    save()
    close()


def before_starting_server():
    cursor, save, close = connect_db()
    
    query = """
    UPDATE jobs SET status = 'waiting' WHERE status == 'running'
    """
    cursor.execute(query)
    
    save()
    close()
    

    
bucket_location = os.path.join(os.getcwd(), "training_data")
output_location = os.path.join(os.getcwd(), "output")

if __name__ == '__main__':
    
    # before starting the server checks if any job is running and update the status to waiting
    before_starting_server()    
    
    process_allowed = 4
    last_job_id = 0
    jobs_queue = []
    
    current_running_job_queue = {}
    # should only run 4 jobs at a time
    with Pool(4) as pool:
        while True:
            completed_job_queue = []
            sleep(5)
            print("[-] Starting a new loop")
            print("[-] Waiting jobs: ", len(jobs_queue))
            print("[-] Running jobs: ", len(current_running_job_queue.keys()))
            
            # check if any job is complete
            for job_id in current_running_job_queue.keys(): # ["sdfjskdf", "ksdjfksdk"]
                job = current_running_job_queue[job_id]
                if job.ready():
                    data = current_running_job_queue[job_id].get()
                    update_job_status(job_id, data['status'])
                    completed_job_queue.append(job_id)
                        
            # remove the completed jobs from the running job queue
            for job_id in completed_job_queue:
                current_running_job_queue.pop(job_id)
    
            print("[-] Current running jobs: ", current_running_job_queue.keys())
            if len(current_running_job_queue.keys()) > process_allowed:
                continue
            
            print("[-] Getting jobs from the database")
            jobs, last_job_id = get_waiting_jobs(last_job_id)
            jobs_queue.extend(jobs)
            
            if len(jobs_queue) == 0:
                continue
            
            # pop the first job from the queue
            job = jobs_queue.pop(0)
            
            print("[-] Starting job: ", job["job_id"])
            # start the job
            running_job = pool.apply_async(start_job, (job,))
            update_job_status(job["job_id"], "running")
            
            # add the job to the running job queue
            current_running_job_queue[job["job_id"]] = running_job
            